"""Why an answer is thin, said from the run's own records. No model writes here.

An answer is thin when a sub-question ended with no verified claim, when no claim survived at all,
when every kept claim is weak, or when fewer than three papers carry it. The reasons are read off
what the run recorded, cause by cause, each with what would change it:

- search providers that did not answer, how often, and the reason they gave (a spent budget, a rate
  limit, an outage);
- the fetch budget (PIPELINE_MAX_FETCH) and the candidates it left unread;
- relevant-looking papers with no open-access copy;
- papers read in full that gave no evidence (the searches reached papers outside the question);
- for each sub-question without a claim, where its trail stopped: no candidates, none readable,
  passages that did not address it, or claims the verifier rejected;
- claims that rest on a single paper.

When nothing technical went wrong, it says so: the literature reached may not address the question.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import State

MIN_PAPERS = 3  # fewer distinct papers behind the kept claims than this reads as thin
KEPT = ("SUPPORTED", "DISPUTED")
_COOLDOWN = re.compile(r"^not asked for \d+ s, after:\s*")
_URL = re.compile(r"(unreachable): \S+ (failed)")  # the request path adds nothing


def _hint(why: str) -> str:
    """What would bring a provider back, from the reason it gave."""
    w = why.lower()
    if "remaining $0" in w or "budget" in w:
        return (
            "its budget for the period is spent; it answers again when the budget resets or "
            "is raised"
        )
    if "429" in w or "rate limit" in w or "rate-limit" in w:
        return "it rate-limited the run; an API key or a slower pace raises the limit"
    if any(
        s in w for s in ("unreachable", "timed out", "http error 5", "bad gateway", "502", "503")
    ):
        return "the service was down or too slow; a later run may reach it"
    if "key" in w or "token" in w or "401" in w or "403" in w:
        return "it refused the credentials; check its key"
    return "see its reason"


def _providers(st: State) -> list[str]:
    total = len(st.searches)
    missed: Counter[str] = Counter()
    reasons: dict[str, str] = {}
    for s in st.searches:
        for name, rec in (s.get("providers") or {}).items():
            if rec.get("status") == "unavailable":
                missed[name] += 1
                why = _URL.sub(r"\1, \2", _COOLDOWN.sub("", str(rec.get("why", "")))).strip()
                # keep the most informative reason: the first that is not a bare cooldown
                if why and (name not in reasons or len(why) > len(reasons[name])):
                    reasons[name] = why
    out = []
    for name, n in missed.most_common():
        why = reasons.get(name, "no reason recorded")
        short = why if len(why) <= 160 else why[:157] + "…"
        out.append(
            f"**{name}** did not answer {n} of {total} searches ({short}): {_hint(why)}. "
            "Papers only it would have found were never seen."
        )
    return out


def diagnose(st: State, *, max_fetch: int | None = None) -> list[str]:
    """Bullet lines (without the leading dash) saying why this answer is thin; [] when it is not."""
    plan = st.plan
    if plan is None:
        return []
    kept = [c for c in st.claims if c.verdict in KEPT]
    works = {
        st.evidence[e].work
        for c in kept
        for e in (c.supported_by or c.evidence_ids)
        if e in st.evidence
    }
    asked = [sq for sq in plan.subquestions if sq.kind != "falsification"]
    empty = [sq for sq in asked if not any(c.subquestion == sq.id for c in kept)]
    all_weak = bool(kept) and all(c.label in ("weak", "insufficient") for c in kept)
    if not (not kept or empty or all_weak or len(works) < MIN_PAPERS):
        return []

    out: list[str] = []
    if not kept:
        out.append("No claim survived verification, so there is nothing to build on.")
    else:
        parts = []
        if empty:
            parts.append(
                f"{len(empty)} of {len(asked)} sub-question(s) ended with no verified claim"
            )
        if len(works) < MIN_PAPERS:
            parts.append(f"the kept claims rest on {len(works)} paper(s)")
        if all_weak:
            parts.append("every kept claim is labelled weak")
        out.append("This answer is thin: " + "; ".join(parts) + ". The causes found in this run:")

    out += _providers(st)

    cands = list(st.candidates.values())
    over = [c for c in cands if c.outcome == "over_budget"]
    if over:
        cap = f"{max_fetch} papers" if max_fetch else "a fixed number of papers"
        out.append(
            f"**Fetch budget:** a run reads at most {cap} (PIPELINE_MAX_FETCH); {len(over)} "
            "further "
            "candidate(s) were left unread. Raising it, or asking a narrower question so the best "
            "candidates rank first, reaches more of them."
        )
    closed = [c for c in cands if c.outcome == "not_obtainable"]
    if closed:
        out.append(
            f"**No open-access copy:** {len(closed)} relevant-looking paper(s) could not be read, "
            "so they contributed nothing (listed under Limits)."
        )
    failed = [c for c in cands if c.outcome == "fetch_failed"]
    if failed:
        out.append(
            f"**Fetch errors:** {len(failed)} paper(s) failed to download; a later run may get "
            "them."
        )

    cited = {e.work for e in st.evidence.values()}
    idle = [c for c in cands if c.outcome == "indexed" and c.work and c.work not in cited]
    read = [c for c in cands if c.outcome == "indexed"]
    if read and len(idle) * 2 >= len(read):
        titles = "; ".join(str(c.title)[:70] for c in idle[:4])
        out.append(
            f"**Little usable text:** {len(idle)} of the {len(read)} paper(s) read in full gave no "
            f"passage taken as evidence (for example: {titles}). Either the searches reached work "
            "outside the question, or the papers do not state a result on it in their text; "
            "wording the question in the field's own terms usually helps."
        )

    for sq in empty:
        mine = [c for c in cands if set(c.queries) & set(sq.queries)]
        readable = [c for c in mine if c.outcome == "indexed"]
        judged = len(st.seen.get(sq.id, ()))
        ev = [e for e in st.evidence.values() if e.subquestion == sq.id]
        drafted = [c for c in st.claims if c.subquestion == sq.id]
        if drafted:
            verdicts = Counter(c.verdict.lower().replace("_", " ") for c in drafted)
            stop = (
                f"{len(drafted)} claim(s) were drafted and the verifier rejected all of them ("
                + ", ".join(f"{n} {v}" for v, n in verdicts.most_common())
                + ")"
            )
        elif ev and all(e.direction == "neutral" for e in ev):
            stop = (
                f"{len(ev)} passage(s) were found on it, but all were read as neutral (no result "
                "for or against), and claims are drafted only from passages that report one"
            )
        elif ev:
            stop = f"{len(ev)} passage(s) were taken as evidence but the model drafted no claim"
        elif not mine:
            stop = "its searches returned no candidate papers and no indexed passage addressed it"
        elif not readable:
            stop = f"its searches found {len(mine)} candidate(s) but none could be read in full"
        else:
            seen = f" and {judged} passage(s) judged" if judged else ""
            stop = f"{len(readable)} paper(s) were read{seen}, but no passage addressed it"
        queries = "; ".join(f"“{q}”" for q in sq.queries[:4]) or "none recorded"
        out.append(f"**“{sq.text}”:** {stop}. Searches: {queries}.")

    if kept and len(works) == 1:
        out.append(
            "**One paper:** every kept claim comes from the same paper; nothing was replicated "
            "across sources."
        )
    elif kept and all(
        len({st.evidence[e].work for e in (c.supported_by or c.evidence_ids) if e in st.evidence})
        <= 1
        for c in kept
    ):
        out.append(
            "**Single sources:** each kept claim rests on one paper; no finding was seen in two "
            "independent papers in this run."
        )

    technical = any(
        line.startswith(("**Fetch", "**No open", "**Little usable")) or " did not answer " in line
        for line in out
    )
    if not technical:
        out.append(
            "No search, fetch or verification failure was recorded: the literature this run "
            "reached may simply not address the question. A narrower or field-specific wording, "
            "or a different domain profile, may reach the work that does."
        )
    return out
