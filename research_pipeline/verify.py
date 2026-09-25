"""Verification. Citation existence and citation entailment are separate tests and stay separate.

Existence is structural here: evidence can only come from a work the paper library resolved against
a bibliographic provider and stored with provenance, and the synthesis schema only admits ids of
retrieved evidence. A reference that does not exist has no way in.

Entailment is checked three ways, cheapest first:
  1. quote anchoring  the extracted quote must occur in the stored passage (deterministic)
  2. number check     figures in a claim must occur in the passages it cites (deterministic)
  3. verifier models  each sees ONLY the claim and the passage, never the conversation or the draft

A claim is SUPPORTED only when every verifier accepts the same piece of evidence and every number
it states occurs in that evidence. A claim with a number its evidence does not contain is removed,
whatever the verifiers said. Verifiers that disagree produce DISPUTED, which is shown to the reader
rather than resolved by a vote.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher

from .domains import GENERIC, DomainPolicy
from .llm import Ollama
from .prompts import DEFAULT, Prompts, wrap
from .schema import Check, Claim, Evidence

SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
# A number and, where it has one, the token that follows it ("10 Hz", "10Hz", "10-Hz", ".05").
# Anything can be captured as a unit here; only tokens a domain declared as units count as one.
# The look-behind keeps identifiers such as "CA1" or "Nav1.7" from being read as quantities.
NUMBER_UNIT = re.compile(
    r"(?<![A-Za-z0-9.])(\d+(?:[.,]\d+)?|\.\d+)(?:\s*-?\s*(%|[A-Za-zµΩ][A-Za-z0-9µΩ/·²³°]*))?"
)
WS = re.compile(r"\s+")
WORDS = re.compile(r"[A-Za-z0-9]+(?:'[a-z]+)?")
NEGATIONS = frozenset(
    [
        "no",
        "not",
        "none",
        "neither",
        "nor",
        "never",
        "without",
        "cannot",
        "unchanged",
        "absent",
        "failed",
        "fail",
        "n't",
    ]
)

# A quote shorter than this cannot tie a finding to a passage: "the effect" occurs everywhere.
MIN_QUOTE_WORDS = 5
# Similarity at which a paraphrased quote is replaced by the source's own sentence. Below it the
# quote is rejected. The replacement is additionally refused if it changes negation or numbers.
QUOTE_REPAIR_THRESHOLD = 0.85

VERDICT_NUMBERS = "NUMBER_NOT_IN_SOURCE"


def _norm(text: str) -> str:
    return WS.sub(" ", text).strip().lower()


def canonical_number(value: str) -> str:
    """One spelling per value: "0.50", ".5" and "0,5" are the same number."""
    raw = value.replace(",", ".")
    if raw.startswith("."):
        raw = "0" + raw
    try:
        return format(Decimal(raw).normalize(), "f")
    except InvalidOperation:
        return raw


def _numbers(text: str) -> set[str]:
    return {canonical_number(value) for value, _ in NUMBER_UNIT.findall(text)}


def _negations(text: str) -> set[str]:
    words = {w.lower() for w in WORDS.findall(text)}
    found = words & NEGATIONS
    if any(w.endswith("n't") for w in words):
        found.add("n't")
    return found


def anchor_quote(
    quote: str, passage: str, threshold: float = QUOTE_REPAIR_THRESHOLD
) -> tuple[str | None, float]:
    """Return the passage's own text matching `quote`, or None, and the match ratio.

    Small models paraphrase even when told to copy. An exact match (ignoring whitespace and case)
    is accepted. Otherwise the closest real sentence is substituted if it is near-identical AND says
    the same thing in the two ways a small edit can reverse a finding: it has the same negations and
    the same numbers. What is stored and shown is always text that exists in the source.
    """
    if len(WORDS.findall(quote)) < MIN_QUOTE_WORDS:
        return None, 0.0
    norm_quote, norm_passage = _norm(quote), _norm(passage)
    start = norm_passage.find(norm_quote)
    if start >= 0:
        return _slice_original(passage, start, len(norm_quote)), 1.0
    best, best_ratio = None, 0.0
    for sentence in SENTENCE.split(passage):
        ratio = SequenceMatcher(None, norm_quote, _norm(sentence)).ratio()
        if ratio > best_ratio:
            best, best_ratio = sentence.strip(), ratio
    if best is None or best_ratio < threshold:
        return None, best_ratio
    if _negations(best) != _negations(quote) or _numbers(best) != _numbers(quote):
        return None, best_ratio
    return best, best_ratio


def _slice_original(passage: str, norm_start: int, norm_len: int) -> str:
    """Map a span found in whitespace-normalised text back onto the original passage."""
    out: list[str] = []
    seen, started, prev_space = 0, False, True
    for ch in passage:
        is_space = ch.isspace()
        if is_space and prev_space:
            if started:
                out.append(ch)
            continue
        if seen >= norm_start:
            started = True
        if started:
            out.append(ch)
        seen += 1
        prev_space = is_space
        if started and seen >= norm_start + norm_len:
            break
    return "".join(out).strip()


# Citation markers inside a quoted sentence. JATS-derived text glues numeric references onto the
# preceding word ("responses3-5,10."); author-year styles use "et al." or "(Name, 2019)". A single
# glued number counts only after a word of seven or more letters, so that gene and channel symbols
# such as "trpv1" or "mecp2" are not read as citations. It is a heuristic and is tested as one.
CITES = re.compile(
    r"[a-z]{4,}\)?\d{1,3}(?:\s?[–\-,]\s?\d{1,3})+(?=[\s.,;:)]|$)"
    r"|[a-z]{7,}\)?\d{1,3}(?=[\s.,;:)]|$)"
    r"|\bet al\.?"
    r"|\[\d{1,3}(?:[–\-,\s]+\d{1,3})*\]"
    r"|\([A-Z][A-Za-z\-]+(?: and [A-Z][A-Za-z\-]+)?,? (?:19|20)\d{2}\)"
)


def cites_other_work(quote: str) -> bool:
    """True when the quoted sentence carries a citation, i.e. it reports someone else's result.

    Citation accuracy studies find that a substantial share of "X et al. showed..." statements
    misreport the cited paper, so a sentence like this is a pointer to a source, not a source. The
    model is asked the same question; this deterministic check overrides a "first-hand" answer,
    never the reverse.
    """
    return bool(CITES.search(quote))


def _quantities(text: str, policy: DomainPolicy) -> tuple[set[str], set[tuple[str, str]]]:
    """Numbers in `text`, and the (number, measure) pairs among them carrying a known unit."""
    bare: set[str] = set()
    measured: set[tuple[str, str]] = set()
    for value, unit in NUMBER_UNIT.findall(text):
        number = canonical_number(value)
        bare.add(number)
        measure = policy.measure_for_unit(unit) if unit else None
        if measure is not None:
            measured.add((number, measure.key))
    return bare, measured


def unsupported_numbers(
    claim: str, passages: list[str], policy: DomainPolicy = GENERIC
) -> list[str]:
    """Quantities asserted by the claim that its cited passages do not support.

    Entailment models are weakest on quantities, and a wrong number is the costliest error in a
    methods or dosing context, so this one is checked literally rather than judged.

    A bare number must occur in a cited passage. A number carrying a unit the field declared
    (10 Hz, 120 mmHg, 50 ms) must occur there with a unit of the same measure: "10 Hz" is not
    supported by a passage whose only "10" was a latency in milliseconds. Units are only known
    when a domain declares them; under the generic policy every number is checked as a bare number.
    """
    pool: set[str] = set()
    measured: set[tuple[str, str]] = set()
    for passage in passages:
        found, pairs = _quantities(passage, policy)
        pool |= found
        measured |= pairs

    missing = []
    for value, unit in NUMBER_UNIT.findall(claim):
        number = canonical_number(value)
        if number not in pool:
            missing.append(value)
            continue
        measure = policy.measure_for_unit(unit) if unit else None
        if measure is not None and (number, measure.key) not in measured:
            missing.append(f"{value} {unit}")
    return missing


def judge(
    llm: Ollama, model: str, claim_text: str, source_text: str, prompts: Prompts = DEFAULT
) -> tuple[str, str]:
    if "minicheck" in model.lower():
        # Bespoke-MiniCheck is a trained grounding classifier with its own fixed prompt and a
        # Yes/No answer. It is binary, so it can confirm support but cannot say "partially".
        reply = llm.chat_text("verify", model, f"Document: {source_text}\nClaim: {claim_text}")
        verdict = "SUPPORTED" if reply.lower().startswith("yes") else "NOT_SUPPORTED"
        return verdict, f"minicheck: {reply[:20]}"
    got = llm.chat_json(
        "verify",
        prompts.verify_system,
        f"Claim: {claim_text}\n\nSource passage:\n{wrap(source_text)}",
        prompts.VERIFY_SCHEMA,
        model=model,
    )
    verdict = got.get("verdict", "NOT_SUPPORTED")
    return verdict, f"claim adds: {str(got.get('claim_adds', ''))[:280]}"


def check_claims(
    claims: list[Claim],
    evidence: dict[str, Evidence],
    passages: dict[str, str],
    llm: Ollama,
    verifiers: list[str],
    *,
    prompts: Prompts = DEFAULT,
) -> None:
    """Run every verifier over every (claim, cited passage) pair, then settle each claim.

    The model loop is outermost: with one model resident at a time, swapping per claim would spend
    longer loading weights than checking.
    """
    if not verifiers:
        raise ValueError("at least one verifier model is required")
    for model in verifiers:
        for claim in claims:
            for eid in claim.evidence_ids:
                if eid in evidence:
                    verdict, reason = judge(llm, model, claim.text, passages[eid], prompts)
                    claim.checks.append(Check(model, eid, verdict, reason))
    for claim in claims:
        settle(claim, evidence, passages, verifiers, prompts.policy)


def settle(
    claim: Claim,
    evidence: dict[str, Evidence],
    passages: dict[str, str],
    verifiers: list[str],
    policy: DomainPolicy = GENERIC,
) -> None:
    """Turn a claim's checks into one verdict. Deterministic checks outrank the verifier models."""
    if not verifiers:
        raise ValueError("at least one verifier model is required")
    unknown = [e for e in claim.evidence_ids if e not in evidence]
    if unknown:
        claim.flags.append("unknown_evidence_ids:" + ",".join(unknown))
    cited = [evidence[e] for e in claim.evidence_ids if e in evidence]
    if not cited:
        claim.verdict, claim.supported_by = "NOT_SUPPORTED", []
        return

    missing = unsupported_numbers(claim.text, [passages[e.id] for e in cited], policy)
    if missing:
        claim.flags.append("numbers_not_in_source:" + ",".join(missing))
        claim.verdict, claim.supported_by = VERDICT_NUMBERS, []
        return

    accepted: dict[str, set[str]] = {e.id: set() for e in cited}
    verdicts = set()
    for check in claim.checks:
        verdicts.add(check.verdict)
        if check.verdict == "SUPPORTED" and check.evidence_id in accepted:
            accepted[check.evidence_id].add(check.verifier)

    required = set(verifiers)
    unanimous = [eid for eid, models in accepted.items() if models >= required]
    some = [eid for eid, models in accepted.items() if models]
    if unanimous:
        claim.verdict, claim.supported_by = "SUPPORTED", unanimous
    elif some:
        claim.verdict, claim.supported_by = "DISPUTED", some
    elif "CONTRADICTED" in verdicts:
        claim.verdict = "CONTRADICTED"
    elif "PARTIALLY_SUPPORTED" in verdicts:
        claim.verdict = "PARTIALLY_SUPPORTED"
    else:
        claim.verdict = "NOT_SUPPORTED"
