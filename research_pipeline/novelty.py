"""Novelty probe: an adversarial prior-art search that READS the nearest papers.

The null hypothesis is: *someone has already done this.* The probe tries to prove it, and it
always writes a prior-art dossier, most of all when nothing is found.

    research-pipeline novelty --id CAND-0007 "the candidate statement" \
        --established "closest established terminology" \
        --queries "phrasing A" "phrasing B" "phrasing C" "phrasing D"

What it adds to a keyword probe: a keyword probe counts ANY search hit as prior art, and a
keyword search nearly always hits something, so it cannot tell "a paper on the same topic" from
"a paper that already states this". Here the nearest
papers are fetched in full and every retrieved passage is put to the pipeline's own verifier with
the candidate statement as the claim. The verifier's question -- does this passage state the claim
itself, with the same population, conditions, direction and numbers -- is exactly the prior-art
question. A passage every verifier accepts, with its quote found in the stored text, is prior art.

Verdicts, strongest first. The words "discovery" and unqualified "novel" are never written.

    PRIOR_ART       a passage states it (all verifiers agree, quote anchored, numbers present)
    PARTLY_KNOWN    a passage states a weaker or narrower version, or verifiers disagree
    INCONCLUSIVE    nothing states it, but an index refused, the run was offline, or too few of
                    the nearest works could be read in full to rule it out
    CANDIDATE       every index answered every formulation, at least MIN_READ of the nearest
                    works were read in full, and no passage states even part of it. This is a
                    statement about the works read, not about the literature.

The dossier is written into the RESEARCH REPOSITORY (novelty/<ID>/), not into this one: the
finding belongs with the corpus it is about. This repository holds the method.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from . import __version__, rerank, verify
from .client_llm import ClientLLM
from .config import Settings
from .papers import PaperLibrary
from .pipeline import Model, Pipeline, Progress, State
from .prompts import wrap
from .schema import Plan, SubQuestion

MIN_QUERIES = 5  # the keyword probe's rule, kept: one phrasing finds one vocabulary
MIN_READ = 3  # nearest works that must be read in full before CANDIDATE may be said
NEAREST = 10  # how many of the nearest works the coverage line accounts for
PASSAGES = 12  # passages put to the verifiers
ID_RULE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")
# Provider statuses that mean the index did not answer. "skipped" is a provider the library chose
# not to call (the web fallback when the indexes answered); it is not a refusal.
REFUSED = ("unavailable", "error")
# Marker files of the research corpus. The dossier is only ever written into a checkout that has
# them, so a mistyped path cannot scatter findings into some other directory.
RESEARCH_MARKERS = ("CITATIONS.json", "FULL_TEXT_READ.yaml")

VERDICTS = {
    "PRIOR_ART": "a paper already states it",
    "PARTLY_KNOWN": "a paper states a weaker or narrower version, or the verifiers disagree",
    "INCONCLUSIVE": "nothing read states it, but the search or the reading was incomplete",
    "CANDIDATE": "no work read states it; unclaimed territory among the works read, "
    "not in the literature",
}


class NoveltyError(ValueError):
    pass


def research_dir(explicit: str | None, env: dict[str, str]) -> Path:
    """The research checkout to write into. Resolve it, or raise with the line that would."""
    raw = explicit or env.get("RESEARCH_REPO", "")
    if not raw:
        raise NoveltyError(
            "no research repository given. Pass --research-dir PATH or set RESEARCH_REPO "
            "(git clone https://github.com/polarizetech/research.git)"
        )
    path = Path(raw).expanduser().resolve()
    missing = [m for m in RESEARCH_MARKERS if not (path / m).is_file()]
    if missing:
        raise NoveltyError(
            f"{path} is not the research repository (missing {', '.join(missing)}); "
            "refusing to write a dossier there"
        )
    return path


def check_inputs(cid: str, statement: str, queries: list[str], established: list[str]) -> list[str]:
    if not ID_RULE.match(cid):
        raise NoveltyError(f"--id {cid!r}: letters, digits, '.', '_' and '-' only")
    if len(statement.split()) < 5:
        raise NoveltyError("the statement is under five words; write the claim out in full")
    if not established:
        raise NoveltyError(
            "--established is required: the closest ESTABLISHED terminology is the phrasing "
            "most likely to find the prior art, and the one most easily left out"
        )
    all_q = list(dict.fromkeys(q.strip() for q in [*established, *queries] if q.strip()))
    if len(all_q) < MIN_QUERIES:
        raise NoveltyError(
            f"{len(all_q)} distinct phrasings; at least {MIN_QUERIES} are required, including "
            "the established terminology"
        )
    return all_q


@dataclass
class Reading:
    """One passage put to the verifiers, with the candidate statement as the claim."""

    passage_id: str
    work: str
    title: str | None
    doi: str | None
    year: int | None
    verdicts: dict[str, str] = field(default_factory=dict)  # verifier -> verdict
    quote: str | None = None  # the passage's own closest sentence, anchored
    claim_adds: list[str] = field(default_factory=list)
    missing_numbers: list[str] = field(default_factory=list)
    secondhand: bool = False
    bearing: str = "none"  # states | partly | contradicts | none


def settle_reading(r: Reading, verifiers: list[str]) -> str:
    """One passage's bearing. Deterministic checks outrank the verifier models, as in verify."""
    vs = [r.verdicts.get(m, "NOT_SUPPORTED") for m in verifiers]
    if all(v == "SUPPORTED" for v in vs) and r.quote and not r.missing_numbers:
        return "states"
    if any(v in ("SUPPORTED", "PARTIALLY_SUPPORTED") for v in vs):
        return "partly"
    if any(v == "CONTRADICTED" for v in vs):
        return "contradicts"
    return "none"


def decide(
    readings: list[Reading],
    refused: dict[str, list[str]],
    search_errors: list[str],
    n_read: int,
    offline: bool,
) -> tuple[str, str]:
    states = [r for r in readings if r.bearing == "states"]
    partly = [r for r in readings if r.bearing == "partly"]
    if states:
        works = len({r.work for r in states})
        return "PRIOR_ART", f"{len(states)} passage(s) in {works} work(s) state it"
    if partly:
        return "PARTLY_KNOWN", (
            f"{len(partly)} passage(s) state a weaker or narrower version, or at least one "
            "verifier accepted a passage the others did not"
        )
    if offline:
        return (
            "INCONCLUSIVE",
            "offline run: no search ran, only the already-indexed corpus was read",
        )
    if search_errors:
        return "INCONCLUSIVE", f"{len(search_errors)} search(es) failed outright"
    if refused:
        return "INCONCLUSIVE", (
            f"{', '.join(sorted(refused))} did not answer every formulation; an index that "
            "refused is not an empty index"
        )
    if n_read < MIN_READ:
        return "INCONCLUSIVE", (
            f"only {n_read} of the nearest works could be read in full (need {MIN_READ}); the "
            "rest may state it"
        )
    return "CANDIDATE", (
        f"every index answered every formulation, {n_read} of the nearest works were read in "
        "full, and no passage states even part of it"
    )


class NoveltyProbe:
    def __init__(
        self,
        settings: Settings,
        progress: Progress | None = None,
        offline: bool = False,
        llm: Model | None = None,
        domain: str | None = None,
    ):
        self.pipe = Pipeline(settings, progress, offline=offline, llm=llm, domain=domain)
        self.s = settings
        self.offline = offline
        self.progress = self.pipe.progress

    # -- the verifier pass -----------------------------------------------------------------
    def _judge(self, model: str, statement: str, text: str) -> dict[str, Any]:
        if "minicheck" in model.lower():
            verdict, reason = verify.judge(self.pipe.llm, model, statement, text, self.pipe.prompts)
            return {"verdict": verdict, "closest_sentence": "", "claim_adds": reason}
        return self.pipe.llm.chat_json(
            "verify",
            self.pipe.prompts.verify_system,
            f"Claim: {statement}\n\nSource passage:\n{wrap(text)}",
            self.pipe.prompts.VERIFY_SCHEMA,
            model=model,
        )

    def read(self, statement: str, passages: list[Any], verifiers: list[str]) -> list[Reading]:
        readings = []
        for p in passages:
            paper = self.pipe.papers.get(p.work) or {}
            readings.append(
                Reading(p.id, p.work, paper.get("title"), paper.get("doi"), paper.get("year"))
            )
        texts = {p.id: p.text for p in passages}
        for model in verifiers:  # one model resident at a time, as in verify.check_claims
            got = self.pipe._parallel(
                [partial(self._judge, model, statement, texts[r.passage_id]) for r in readings]
            )
            for r, g in zip(readings, got, strict=True):
                r.verdicts[model] = g.get("verdict", "NOT_SUPPORTED")
                if adds := str(g.get("claim_adds", "")).strip():
                    r.claim_adds.append(f"{model}: {adds[:280]}")
                if r.quote is None and (closest := g.get("closest_sentence")):
                    r.quote, _ = verify.anchor_quote(closest, texts[r.passage_id])
        for r in readings:
            r.missing_numbers = verify.unsupported_numbers(
                statement, [texts[r.passage_id]], self.pipe.policy
            )
            r.secondhand = bool(r.quote) and verify.cites_other_work(r.quote or "")
            r.bearing = settle_reading(r, verifiers)
        return readings

    # -- the run ---------------------------------------------------------------------------
    async def run(
        self, cid: str, statement: str, queries: list[str], established: list[str]
    ) -> dict[str, Any]:
        all_q = check_inputs(cid, statement, queries, established)
        started = time.time()
        st = State(statement.strip())
        verifiers = self.pipe._resolve_models(st)
        ranker = rerank.build(
            self.s,
            None if isinstance(self.pipe.llm, ClientLLM) else self.pipe.llm,
            self.pipe.prompts,
        )
        sq = SubQuestion("S1", statement.strip(), "evidence", all_q)
        st.plan = Plan(statement.strip(), "novelty_probe", statement.strip(), [sq])

        async with PaperLibrary(self.s.gateway_config, self.s.papers_upstream) as lib:
            if not self.offline:
                await self.pipe.discover(st, lib, all_q)
                await self.pipe.acquire(st, lib)
            pools = await self.pipe.pool(st, lib, [(sq, None)])
        saved = self.s.passages_per_subquestion
        self.s.passages_per_subquestion = max(saved, PASSAGES)
        try:
            ranked = await asyncio.to_thread(self.pipe.select, sq, pools["S1"], ranker)
        finally:
            self.s.passages_per_subquestion = saved
        self.progress("verify", f"{len(ranked)} passages x {len(verifiers)} verifier(s)")
        readings = await asyncio.to_thread(
            self.read, statement.strip(), [p for p, _ in ranked], verifiers
        )

        refused: dict[str, list[str]] = {}
        search_errors = []
        for s in st.searches:
            if "error" in s:
                search_errors.append(f"{s['query']}: {s['error']}")
                continue
            for name, rep in s["providers"].items():
                if rep.get("status") in REFUSED:
                    refused.setdefault(name, []).append(
                        f"{rep.get('status')}: {rep.get('why', '')}"[:160]
                    )
        nearest = sorted(st.candidates.values(), key=lambda c: -c.relevance)[:NEAREST]
        n_read = sum(c.outcome == "indexed" for c in nearest)
        verdict, why = decide(readings, refused, search_errors, n_read, self.offline)

        return {
            "id": cid,
            "statement": statement.strip(),
            "verdict": verdict,
            "verdict_meaning": VERDICTS[verdict],
            "why": why,
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "seconds": round(time.time() - started, 1),
            "method": {
                "tool": "retrieval-augmented-generation novelty",
                "pipeline_version": __version__,
                "prompt_version": self.pipe.prompts.version,
                "models": self.pipe.models,
                "reranker": ranker.name,
                "min_read": MIN_READ,
                "offline": self.offline,
            },
            "formulations": {"established": established, "all": all_q},
            # The keyword probe's rule, kept for comparison: any hit at all.
            "search_level": {
                "any_hits": any(s.get("n_hits", 0) for s in st.searches),
                "hits": sum(s.get("n_hits", 0) for s in st.searches),
            },
            "searches": st.searches,
            "refused": refused,
            "search_errors": search_errors,
            "nearest": [
                {
                    "title": c.title,
                    "year": c.year,
                    "doi": c.ids.get("doi"),
                    "providers": c.providers,
                    "queries": sorted(set(c.queries)),
                    "outcome": c.outcome,
                    "relevance": c.relevance,
                }
                for c in nearest
            ],
            "read_in_full": n_read,
            "readings": [asdict(r) for r in readings],
            "notes": [n for n in st.notes if not n.startswith("Verification used the same model")]
            + (
                [
                    "One model judged every passage. A second verifier from a different model "
                    "family (PIPELINE_VERIFIER_MODEL) would require two to agree before PRIOR_ART."
                ]
                if len(verifiers) == 1
                else []
            ),
            "llm_calls": self.pipe.llm.calls,
        }


# -- the dossier ---------------------------------------------------------------------------
BANNED = re.compile(r"\bdiscover(y|ies|ed)\b|\bnovel\b", re.I)


def _ref(r: dict[str, Any]) -> str:
    ident = f"https://doi.org/{r['doi']}" if r.get("doi") else (r.get("work") or "no DOI")
    return f"{r.get('title') or '(untitled)'} ({r.get('year') or '?'}) — {ident}"


def render(log: dict[str, Any]) -> str:
    """The dossier. Every sentence here is template text or quoted from a source or the input."""
    out = [
        f"# {log['id']} — prior-art dossier",
        "",
        f"_Probed {log['date'][:10]} by {log['method']['tool']} "
        f"(pipeline {log['method']['pipeline_version']}, "
        f"prompts {log['method']['prompt_version']}). "
        f"{len(log['formulations']['all'])} formulations; {log['read_in_full']} of the nearest "
        f"works read in full._",
        "",
        "## Statement under test",
        "",
        f"> {log['statement']}",
        "",
        f"## Verdict: {log['verdict']}",
        "",
        f"{log['verdict_meaning'].capitalize()}. {log['why'].capitalize()}.",
        "",
        "## How to kill this",
        "",
        "The cheapest test that would show this is not new: put the established-terminology "
        "phrasing below to a domain expert or a recent review. If any work listed here, or one "
        "this search could not read, already states it, it dies.",
        "",
    ]
    by = {"states": [], "partly": [], "contradicts": []}
    for r in log["readings"]:
        if r["bearing"] in by:
            by[r["bearing"]].append(r)
    for key, head in (
        ("states", "Passages that state it"),
        ("partly", "Passages that state part of it"),
        ("contradicts", "Passages that state the opposite"),
    ):
        if not by[key]:
            continue
        out += [f"## {head}", ""]
        for r in by[key]:
            verdicts = ", ".join(f"{m}: {v}" for m, v in r["verdicts"].items())
            out.append(f"- {_ref(r)}")
            if r["quote"]:
                out.append(f"  > {r['quote']}")
            else:
                out.append("  - no sentence of the passage could be anchored; read the paper")
            out.append(f"  - verifiers: {verdicts}")
            for adds in r["claim_adds"]:
                out.append(f"  - what the statement adds ({adds})")
            if r["missing_numbers"]:
                out.append(
                    f"  - numbers in the statement not in this passage: "
                    f"{', '.join(r['missing_numbers'])}"
                )
            if r["secondhand"]:
                out.append(
                    "  - this sentence cites another paper: follow the citation, which is "
                    "the earlier source"
                )
        out.append("")
    out += ["## Nearest works", ""]
    for c in log["nearest"]:
        state = {
            "indexed": "read in full",
            "not_obtainable": "NOT READ — no open full text",
            "over_budget": "not read — fetch budget",
            "fetch_failed": "NOT READ — fetch failed",
            "retracted": "retracted",
            "not_attempted": "not read",
        }.get(c["outcome"], c["outcome"])
        out.append(f"- {_ref(c)} — {state} — via {', '.join(c['providers']) or '?'}")
    if not log["nearest"]:
        out.append("_No work was returned on any formulation._")
    out += ["", "## Formulations", ""]
    for q in log["formulations"]["all"]:
        tag = " (established terminology)" if q in log["formulations"]["established"] else ""
        out.append(f"- {q}{tag}")
    out += ["", "## Index coverage", ""]
    names = sorted({n for s in log["searches"] if "providers" in s for n in s["providers"]})
    for n in names:
        answered = sum(
            1
            for s in log["searches"]
            if "providers" in s and s["providers"].get(n, {}).get("status") in ("ok", "cache")
        )
        statuses = {
            s["providers"].get(n, {}).get("status") for s in log["searches"] if "providers" in s
        }
        if statuses <= {"skipped", None}:
            out.append(f"- {n}: not called on these formulations (a fallback provider)")
            continue
        out.append(f"- {n}: answered {answered}/{len(log['searches'])}")
        for why in dict.fromkeys(log["refused"].get(n, [])):
            out.append(f"  - did not answer — {why}")
    for e in log["search_errors"]:
        out.append(f"- search failed — {e}")
    if log["method"]["offline"]:
        out.append("- offline run: no index was searched")
    sl = log["search_level"]
    out += [
        "",
        "## For comparison: the keyword rule",
        "",
        f"The earlier keyword probe counted any hit as prior art. This run returned {sl['hits']} "
        f"hits, so that rule would have said "
        f"{'prior art exists' if sl['any_hits'] else 'no prior art'} without reading any of them.",
        "",
        "## Models",
        "",
    ]
    models = log["method"]["models"]
    out.append(f"- text: {models.get('text', {}).get('name')}")
    for v in models.get("verifiers", []):
        out.append(f"- verifier: {v['name']} ({v['digest'][:19]})")
    out.append(f"- reranker: {log['method']['reranker']}")
    for note in log["notes"]:
        out.append(f"- note: {note}")
    return "\n".join(out) + "\n"


def template_text(md: str, log: dict[str, Any]) -> str:
    """The dossier minus everything quoted from the input or a source: what the banned-word rule
    applies to."""
    for s in [log["statement"], *log["formulations"]["all"]]:
        md = md.replace(s, "")
    for r in log["readings"]:
        for s in (r.get("quote"), r.get("title"), *r.get("claim_adds", [])):
            if s:
                md = md.replace(s, "")
    for c in log["nearest"]:
        if c.get("title"):
            md = md.replace(c["title"], "")
    return md


def write(log: dict[str, Any], research: Path) -> Path:
    md = render(log)
    if hit := BANNED.search(template_text(md, log)):
        raise NoveltyError(f"refusing to write a dossier containing {hit.group(0)!r}")
    out = research / "novelty" / log["id"]
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "prior-art.md").write_text(md)
    stamp = log["date"].replace(":", "")
    (out / "runs" / f"{stamp}.json").write_text(json.dumps(log, indent=1, ensure_ascii=False))
    return out
