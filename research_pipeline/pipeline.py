"""The research run: a fixed sequence of stages, with the model called only inside them.

    plan -> discover -> acquire+index -> retrieve+rerank -> extract (evidence table)
         -> synthesise -> critique (new searches, not rewrites) -> verify -> grade -> render -> log

Discovery, acquisition, indexing and retrieval are the paper library's (paper-fetch, over MCP):
this module tells it the field and the concepts, and it knows how that field is indexed, what was
searched before, and which passages answer. Everything from reading a passage on is here.

Control flow never depends on the model choosing a tool. A ~4B model is unreliable as an agent but
adequate as a component when each call is small, schema-constrained, and checked afterwards.

The model is either a local Ollama model or, behind the MCP server, the client's own model
(client_llm.ClientLLM). Within a stage, independent calls are issued together so that a client can
answer a whole batch per turn; with Ollama they run one at a time, as a single resident model would.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from . import __version__, companions, grading, integrity, safety, verify
from .client_llm import ClientLLM
from .config import Settings
from .domains import Domain, active
from .llm import ModelOutputError, Ollama
from .notes import NoteStore
from .papers import PaperLibrary, PapersError, Passage
from .prompts import Prompts, wrap
from .schema import (
    Candidate,
    Claim,
    Evidence,
    Plan,
    SubQuestion,
    parent_doi,
    source_status,
    title_key,
    to_json,
)
from .upstream import UpstreamError

Progress = Callable[[str, str], None]
Model = Ollama | ClientLLM


@dataclass
class State:
    question: str
    plan: Plan | None = None
    searches: list[dict[str, Any]] = field(default_factory=list)
    candidates: dict[str, Candidate] = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    passages: dict[str, str] = field(default_factory=dict)  # evidence id -> full passage text
    seen: dict[str, set[str]] = field(default_factory=dict)  # subquestion id -> passage ids judged
    claims: list[Claim] = field(default_factory=list)
    insufficient: list[str] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    dropped_passages: list[dict[str, Any]] = field(default_factory=list)
    summary: list[dict[str, Any]] = field(default_factory=list)
    integrity: dict[str, dict[str, Any]] = field(default_factory=dict)  # doi -> Crossref status
    requests: list[dict[str, Any]] = field(
        default_factory=list
    )  # companion tools the plan asked for
    companions: dict[str, dict[str, Any]] = field(default_factory=dict)  # key -> what it returned
    fetches: int = 0  # library fetches attempted this run, against PIPELINE_MAX_FETCH
    notes: list[str] = field(default_factory=list)


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        progress: Progress | None = None,
        offline: bool = False,
        domain: str | Domain | None = None,
        *,
        llm: Model | None = None,
        collection: str | None = None,
    ):
        self.s = settings
        self.offline = offline  # answer from the already-indexed corpus; no discovery, no fetching
        self.progress = progress or (lambda stage, detail: None)
        self.llm: Model = llm or Ollama(settings)
        self.client = isinstance(self.llm, ClientLLM)
        # A client answers a batch per turn; a single resident local model answers one at a time.
        self.concurrency = 32 if self.client else 1
        self.notes = NoteStore(settings.notes_path)
        self.collection = settings.collection if collection is None else collection
        # What the library said about each paper a passage came from (title, doi, retraction,
        # text hash, route), and the state of its passage index at the last retrieval.
        self.papers: dict[str, dict[str, Any]] = {}
        self.index_stats: dict[str, Any] = {}
        self.library_reranker: str | None = None
        self.profile: dict[str, Any] | None = None
        self.companions: dict[str, companions.Companion] = {}  # configured ones, set per run
        self.models: dict[str, Any] = {}
        # The field being researched. Nothing installed means the generic policy, not a failure:
        # the engine answers a question on its own, it just answers it without a field's rules.
        self.domain = domain if isinstance(domain, Domain) else active(domain or settings.domain)
        self.policy = self.domain.policy
        self.prompts = Prompts(self.policy)

    # -- setup -----------------------------------------------------------------------------
    def _resolve_models(self, st: State) -> list[str]:
        name = self.llm.name if isinstance(self.llm, ClientLLM) else self.s.text_model
        text, digest = self.llm.resolve(name)
        self.s.text_model = text
        self.models["text"] = {"name": text, "digest": digest}
        # A client model always verifies its own claims too. A configured local verifier is an
        # independent second check, and every verifier must accept a claim.
        resolved = [(text, digest)] if self.client else []
        resolved += [
            self.llm.resolve(v)
            for v in filter(None, map(str.strip, self.s.verifier_model.split(",")))
        ]
        if not resolved or [n for n, _ in resolved] == [text]:
            resolved = [(text, digest)]
            st.notes.append(
                "Verification used the same model that wrote the claims. It sees only the claim "
                "and the passage, but it shares the writer's blind spots. Set "
                "PIPELINE_VERIFIER_MODEL to a different model family for an independent check."
            )
        self.models["verifiers"] = [{"name": n, "digest": d} for n, d in resolved]
        return [n for n, _ in resolved]

    # -- stage 0-2: plan -------------------------------------------------------------------
    def plan(self, st: State) -> None:
        got = self.llm.chat_json(
            "plan",
            self.prompts.plan_system,
            f"{self.prompts.plan_context()}\n\nResearch question: {st.question}",
            self.prompts.plan_schema(self.s.max_subquestions, self.s.queries_per_subquestion),
        )
        subs = [
            SubQuestion(
                f"S{i}",
                row["question"].strip(),
                "evidence",
                [q.strip() for q in row["queries"] if q.strip()],
            )
            for i, row in enumerate(got["subquestions"], 1)
        ]
        core = got.get("core_question", "").strip() or st.question
        # The falsification pass is judged against the core question, but retrieved with queries
        # written to find null results and failed replications. It always runs.
        subs.append(
            SubQuestion(
                "F1",
                core,
                "falsification",
                [q.strip() for q in got["falsification_queries"] if q.strip()],
            )
        )
        st.plan = Plan(st.question, got["mode"], core, subs)
        # Companion tools the planner asked for. The schema only admits configured ones, but as
        # everywhere else that is checked here too; each tool is asked at most once.
        for row in got.get("tools", []):
            comp = self.companions.get(str(row.get("tool")))
            queries = [q.strip() for q in row.get("queries", []) if q.strip()]
            if comp and queries and comp.key not in {r["tool"] for r in st.requests}:
                st.requests.append(
                    {
                        "tool": comp.key,
                        "queries": queries[: comp.max_queries],
                        "why": str(row.get("why", "")).strip(),
                    }
                )

    async def run_companions(self, st: State, lib: PaperLibrary) -> None:
        """Call each companion tool the plan requested. A companion never ends a run: one that
        fails is recorded as having not answered, which the answer then says."""
        assert st.plan
        targets = [st.question, *(sq.text for sq in st.plan.subquestions if sq.kind == "evidence")]

        async def rank(texts: list[str]) -> list[float]:
            return await lib.relevance(targets, texts)

        for request in st.requests:
            comp = self.companions[request["tool"]]
            self.progress("companion", f"{comp.key}: {'; '.join(request['queries'])}")
            try:
                async with companions.connect(self.s.gateway_config, comp.key) as up:
                    st.companions[comp.key] = await comp.run(up, request["queries"], rank)
            except (UpstreamError, OSError, KeyError, TypeError, ValueError) as exc:
                st.companions[comp.key] = {
                    "queries": request["queries"],
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }

    # -- stage 3-4: discover and normalise -------------------------------------------------
    async def discover(self, st: State, lib: PaperLibrary, queries: list[str]) -> None:
        # Each query is an independent provider round-trip; running them concurrently (bounded, so
        # a slow/rate-limited provider does not stall the rest) only shortens wall-clock time.
        sem = asyncio.Semaphore(self.s.search_concurrency)

        async def _search_one(query: str) -> tuple[str, dict[str, Any] | None, PapersError | None]:
            self.progress("discover", query)
            async with sem:
                try:
                    res = await lib.search(
                        query,
                        limit=self.s.hits_per_query,
                        profile=(self.policy.profile or "") if self.profile else "",
                        collection=self.collection,
                    )
                    return query, res, None
                except PapersError as exc:
                    return query, None, exc

        for query, res, exc in await asyncio.gather(*(_search_one(q) for q in queries)):
            if exc is not None:
                st.searches.append({"query": query, "error": str(exc)})
                continue
            assert res is not None
            silent = [
                name
                for name, rep in res["providers"].items()
                if rep.get("status") in ("unavailable", "error", "skipped")
            ]
            st.searches.append(
                {
                    "query": query,
                    "variants": res.get("variants", []),
                    "providers": res["providers"],
                    "n_hits": len(res["hits"]),
                    "did_not_answer": silent,
                    "search_id": res.get("search_id"),
                    "searched_before": [m["query"] for m in res.get("memory", [])],
                }
            )
            for hit in res["hits"]:
                ids = dict(hit.get("ids") or {})
                title, work = hit.get("title"), hit.get("work")
                doi = (ids.get("doi") or "").lower()
                if doi and (parent := parent_doi(doi)) != doi:
                    # A figure or table listed as its own work: stand in for the article, and drop
                    # the component's own title and ids, which would fetch the figure again.
                    ids, title, work = {"doi": parent}, None, None
                    st.notes.append(f"search listed {doi}, a component of {parent}; using {parent}")
                key = (ids.get("doi") or "").lower() or next(
                    (f"{k}:{ids[k]}" for k in ("pmid", "pmcid", "openalex", "arxiv") if ids.get(k)),
                    title_key(title),
                )
                if not key:
                    continue
                cand = st.candidates.get(key)
                if cand is None:
                    cand = st.candidates[key] = Candidate(
                        key,
                        title,
                        hit.get("year"),
                        ids,
                        hit.get("authors") or [],
                        hit.get("is_oa"),
                        list(hit.get("providers") or []),
                    )
                cand.queries.append(query)
                cand.is_oa = cand.is_oa or hit.get("is_oa")
                cand.title = cand.title or title
                cand.work = cand.work or work
                cand.full_text_in_library = cand.full_text_in_library or bool(
                    hit.get("full_text_in_library")
                )
        self._fold_versions(st)

    @staticmethod
    def _fold_versions(st: State) -> None:
        """A preprint and its published version are one study. Keep the published record."""
        by_title: dict[str, list[Candidate]] = {}
        for cand in st.candidates.values():
            if tk := title_key(cand.title):
                by_title.setdefault(tk, []).append(cand)
        for group in by_title.values():
            if len(group) < 2:
                continue
            group.sort(
                key=lambda c: (source_status(c.ids.get("doi")) != "published", -(c.year or 0))
            )
            keep = group[0]
            for other in group[1:]:
                keep.queries.extend(other.queries)
                keep.providers = sorted(set(keep.providers) | set(other.providers))
                st.candidates.pop(other.key, None)

    # -- acquire: fetch through the library, have it index the full texts -------------------
    async def acquire(self, st: State, lib: PaperLibrary) -> None:
        assert st.plan
        fresh = [c for c in st.candidates.values() if c.outcome in ("not_attempted", "over_budget")]
        if not fresh:
            return
        # Rank what to read by how close its title sits to a sub-question, so a small fetch budget
        # is spent on the most relevant candidates rather than on whatever a provider listed first.
        unscored = [c for c in fresh if not c.relevance]
        if unscored:
            scores = await lib.relevance(
                [sq.text for sq in st.plan.subquestions], [c.title or "" for c in unscored]
            )
            for cand, score in zip(unscored, scores, strict=True):
                cand.relevance = round(float(score), 4)
        # Among them, papers a provider reported as open access go first, and papers every
        # provider reported as closed go last: a fetch of a closed paper spends budget and
        # returns nothing to read. (Observed: all six fetches of a run went to closed papers.)
        fresh.sort(key=lambda c: ({True: 0, None: 1, False: 2}[c.is_oa], -c.relevance))

        # Decide who gets fetched, and consume the budget, before any awaiting: this keeps the
        # same priority order and the same PIPELINE_MAX_FETCH accounting as the sequential version.
        held: list[Candidate] = []
        to_fetch: list[tuple[Candidate, str]] = []
        for cand in fresh:
            if cand.work and cand.full_text_in_library:
                held.append(cand)
                continue
            # PIPELINE_MAX_FETCH bounds the whole run, not each round: every fetch writes into the
            # shared paper library, and the critique round asks for more searches by design.
            if st.fetches >= self.s.max_fetch:
                cand.outcome = "over_budget"
                continue
            ident = (
                cand.work
                or cand.ids.get("doi")
                or cand.ids.get("pmcid")
                or (
                    f"pmid:{cand.ids['pmid']}" if cand.ids.get("pmid") else cand.ids.get("openalex")
                )
            )
            if not ident:
                cand.outcome = "fetch_failed"
                continue
            st.fetches += 1
            to_fetch.append((cand, ident))

        sem = asyncio.Semaphore(self.s.fetch_concurrency)

        async def _fetch_one(cand: Candidate, ident: str) -> None:
            self.progress("fetch", f"{cand.title or ident}"[:90])
            try:
                async with sem:
                    rec = await lib.fetch(ident, collection=self.collection)
                cand.work = rec.get("work") or cand.work or ident
                if rec.get("is_retracted"):
                    cand.outcome = "retracted"
                elif not rec.get("full_text"):
                    cand.outcome = "not_obtainable"
                else:
                    held.append(cand)
            except PapersError as exc:
                cand.outcome = "not_obtainable" if exc.code == "not_found" else "fetch_failed"
                st.notes.append(f"fetch {ident}: {exc}")

        await asyncio.gather(*(_fetch_one(cand, ident) for cand, ident in to_fetch))
        if not held:
            return
        # The library indexes the texts (and embeds them, if it has an embedding model); papers it
        # has already indexed cost nothing.
        works = list(dict.fromkeys(c.work for c in held if c.work))
        self.progress("index", f"{len(works)} paper(s)")
        try:
            report = await lib.index(works)
        except PapersError as exc:
            st.notes.append(f"indexing failed: {exc}")
            for cand in held:
                cand.outcome = "fetch_failed"
            return
        for cand in held:
            cand.outcome = "indexed"
        self.index_stats = report.get("index") or self.index_stats
        self.progress("index", f"{report.get('passages_added', 0)} new passages")

    # -- stage 5: retrieve and rerank ------------------------------------------------------
    async def pool(
        self,
        st: State,
        lib: PaperLibrary,
        targets: list[tuple[SubQuestion, list[str] | None]],
    ) -> dict[str, list[Passage]]:
        """Retrieval for several sub-questions at once, in one library call."""
        plans = {
            sq.id: (
                queries or ([sq.text, *sq.queries] if sq.kind != "falsification" else sq.queries)
            )
            for sq, queries in targets
        }
        flat = list(dict.fromkeys(q for qs in plans.values() for q in qs))
        res = await lib.retrieve(
            flat, collection=self.collection, limit=self.s.candidates_per_subquestion
        )
        self.index_stats = res.get("index") or self.index_stats
        self.library_reranker = res.get("reranker")
        by_query = {
            r["query"]: [Passage.from_json(p) for p in r["passages"]] for r in res["results"]
        }
        used_elsewhere = {e.passage_id for e in st.evidence.values()}
        out: dict[str, list[Passage]] = {}
        for sq, _ in targets:
            found: dict[str, Passage] = {}
            for text in plans[sq.id]:
                for p in by_query.get(text, []):
                    if p.id not in found or p.fused > found[p.id].fused:
                        found[p.id] = p
            judged = st.seen.setdefault(sq.id, set())
            kept: list[Passage] = []
            for p in sorted(found.values(), key=lambda p: -p.fused)[
                : self.s.candidates_per_subquestion
            ]:
                # The falsification pass looks for what the other passes missed, not the same text.
                if p.id in judged or (sq.kind == "falsification" and p.id in used_elsewhere):
                    continue
                if p.work not in self.papers:
                    self.papers[p.work] = p.paper
                    hidden = int(p.paper.get("hidden_chars") or 0)
                    if hidden > safety.HIDDEN_TOLERANCE:
                        st.notes.append(f"{p.work}: {hidden} hidden characters; stripped")
                flags = safety.scan(p.text) + (["retracted"] if p.paper.get("is_retracted") else [])
                if flags:
                    st.dropped_passages.append({"passage_id": p.id, "work": p.work, "flags": flags})
                    continue
                kept.append(p)
            out[sq.id] = kept
        return out

    def select(self, sq: SubQuestion, kept: list[Passage]) -> list[tuple[Passage, float]]:
        """The passages to read for a sub-question, in the library's order. The library ranks
        (its cross-encoder sees passage text only: no year, venue or citation count to be biased
        by); this only caps how many come from one paper and how many are read."""
        out: list[tuple[Passage, float]] = []
        per_paper: dict[str, int] = {}
        for p, score in sorted(((p, p.fused) for p in kept), key=lambda t: -t[1]):
            # One paper repeating itself must not look like several sources.
            if per_paper.get(p.work, 0) >= self.s.max_passages_per_paper:
                continue
            per_paper[p.work] = per_paper.get(p.work, 0) + 1
            out.append((p, float(score)))
            if len(out) >= self.s.passages_per_subquestion:
                break
        return out

    def _parallel(self, jobs: list[Callable[[], Any]]) -> list[Any]:
        """Run independent model calls together, results in order. See the module docstring."""
        if self.concurrency <= 1 or len(jobs) <= 1:
            return [job() for job in jobs]
        with ThreadPoolExecutor(max_workers=min(len(jobs), self.concurrency)) as pool:
            return list(pool.map(lambda job: job(), jobs))

    # Classification: the openings of unclassified papers are read from the library first (async),
    # then only the model calls run concurrently; the note store stays on the stage's own thread.
    async def openings(self, lib: PaperLibrary, works: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for work in dict.fromkeys(works):
            if self._note(work) is not None:
                continue
            try:
                out[work] = "\n\n".join(p.text for p in await lib.passages(work, 0, 4))[:2500]
            except PapersError:
                continue  # left unclassified: graded as "unclear"
        return out

    def _note(self, work: str) -> dict[str, str] | None:
        return self.notes.get(work, self.papers.get(work, {}).get("text_sha256"))

    def _classify_input(self, work: str, opening: str) -> str | None:
        """The classifier's input for a paper, or None if its text was flagged (then noted)."""
        if safety.scan(opening):
            # The classifier would read the injected text; record the paper as unclassified.
            self._set_note(work, "unclear", "", "safety-filter")
            return None
        return f"Title: {self.papers.get(work, {}).get('title')}\n\n{wrap(opening)}"

    def _set_note(self, work: str, study_type: str, population: str, model: str) -> None:
        sha = self.papers.get(work, {}).get("text_sha256") or ""
        self.notes.set(work, study_type, population, model, sha)

    def _classify_call(self, user: str) -> dict[str, Any] | None:
        """The classification, or None when the model gave no usable answer: the paper is then
        graded as "unclear" in this run and classified again in the next."""
        try:
            return self.llm.chat_json(
                "classify_paper", self.prompts.paper_system, user, self.prompts.paper_schema()
            )
        except ModelOutputError:
            return None

    def _classify_record(self, work: str, got: dict[str, Any]) -> None:
        self._set_note(
            work, got["study_type"], got.get("population", "").strip(), self.s.text_model
        )

    def paper_note(self, work: str) -> dict[str, str]:
        return self._note(work) or {"study_type": "unclear", "population": ""}

    # -- stage 6: evidence table -----------------------------------------------------------
    def extract(
        self,
        st: State,
        sq: SubQuestion,
        ranked: list[tuple[Passage, float]],
        openings: dict[str, str] | None = None,
    ) -> None:
        self.extract_many(st, [(sq, ranked)], openings or {})

    def _read(self, sq: SubQuestion, p: Passage) -> dict[str, Any] | None:
        """What the model extracted from a passage, or None when it gave no usable answer."""
        try:
            return self.llm.chat_json(
                "extract",
                self.prompts.extract_system,
                f"Research question: {sq.text}\n\n{wrap(p.text)}",
                self.prompts.EXTRACT_SCHEMA,
            )
        except ModelOutputError:
            return None

    def extract_many(
        self,
        st: State,
        batches: list[tuple[SubQuestion, list[tuple[Passage, float]]]],
        openings: dict[str, str],
    ) -> None:
        """Read every selected passage of several sub-questions in one batch of model calls.

        Papers that have not been classified yet are classified in the same batch. Results are
        applied in order afterwards, so evidence ids do not depend on which call finished first.
        """
        items = [(sq, p, score) for sq, ranked in batches for p, score in ranked]
        for sq, p, _ in items:
            st.seen.setdefault(sq.id, set()).add(p.id)
        unclassified = [
            w for w in dict.fromkeys(p.work for _, p, _ in items) if self._note(w) is None
        ]
        self.progress("extract", f"{len(items)} passages, {len(unclassified)} new papers")
        to_classify = [
            (w, user)
            for w in unclassified
            if w in openings and (user := self._classify_input(w, openings[w]))
        ]
        jobs: list[Callable[[], Any]] = [partial(self._read, sq, p) for sq, p, _ in items]
        jobs += [partial(self._classify_call, user) for _, user in to_classify]
        answers = self._parallel(jobs)
        results = answers[: len(items)]
        for (work, _), got in zip(to_classify, answers[len(items) :], strict=True):
            if got is None:
                st.notes.append(f"{work}: the model gave no study-design answer; graded unclear")
            else:
                self._classify_record(work, got)
        for (sq, p, score), got in zip(items, results, strict=True):
            if got is None:
                # A passage the model could not read is dropped and counted, not guessed at.
                st.dropped_passages.append(
                    {"passage_id": p.id, "work": p.work, "flags": ["no_answer_from_model"]}
                )
                continue
            if not got.get("relevant"):
                continue
            quote, ratio = verify.anchor_quote(got.get("quote", ""), p.text)
            if quote is None:
                st.dropped_passages.append(
                    {
                        "passage_id": p.id,
                        "work": p.work,
                        "flags": [f"quote_not_in_passage:{ratio:.2f}"],
                    }
                )
                continue
            direction = got["direction"]
            secondhand = bool(got.get("secondhand")) or verify.cites_other_work(quote)
            note = self.paper_note(p.work)
            # Only a paper's own reported result is evidential. A passage describing someone else's
            # finding is a pointer to that paper, not a second source for the finding.
            role = (
                "O"
                if secondhand or direction == "neutral" or note["study_type"] == "review"
                else "E"
            )
            eid = f"{sq.id}-E{len([e for e in st.evidence.values() if e.subquestion == sq.id]) + 1}"
            st.evidence[eid] = Evidence(
                eid,
                sq.id,
                p.work,
                p.id,
                p.start,
                p.end,
                direction,
                got["finding"].strip(),
                quote,
                note["study_type"],
                got.get("population", "").strip() or note.get("population", ""),
                secondhand,
                role,
                rerank_score=round(score, 4),
                flags=["quote_repaired"] if ratio < 1.0 else [],
            )
            st.passages[eid] = p.text

    # -- stage 7-8: synthesis from evidence ids --------------------------------------------
    def synthesise(self, st: State, sq: SubQuestion) -> None:
        self.synthesise_many(st, [sq])

    def synthesise_many(self, st: State, sqs: list[SubQuestion]) -> None:
        """Draft claims for several sub-questions in one batch; each sees only its own evidence."""
        drafts = []
        for sq in sqs:
            st.claims = [c for c in st.claims if c.subquestion != sq.id]
            if sq.id in st.insufficient:
                st.insufficient.remove(sq.id)
            rows = [
                e
                for e in st.evidence.values()
                if e.subquestion == sq.id and e.direction != "neutral"
            ]
            if not rows:
                if sq.kind != "falsification":
                    st.insufficient.append(sq.id)
                continue
            drafts.append((sq, rows))
        results = self._parallel([partial(self._draft, sq, rows) for sq, rows in drafts])
        for (sq, rows), got in zip(drafts, results, strict=True):
            self._accept_claims(st, sq, rows, got)

    def _draft(self, sq: SubQuestion, rows: list[Evidence]) -> dict[str, Any]:
        listing = "\n".join(
            f"[{e.id}] ({e.study_type}; {e.population or 'population not stated'}; {e.direction}"
            f"{'; describes other work' if e.secondhand else ''}) {e.finding}"
            for e in rows
        )
        ask = (
            f"Sub-question: {sq.text}\n\nFindings:\n{listing}"
            if sq.kind != "falsification"
            else f"Question under test: {sq.text}\n\nThese findings came from searches for null "
            "results "
            f"and failed replications. State each as a claim.\n\nFindings:\n{listing}"
        )
        return self.llm.chat_json(
            "synthesise",
            self.prompts.synth_system,
            ask,
            self.prompts.synth_schema([e.id for e in rows]),
        )

    def _accept_claims(
        self, st: State, sq: SubQuestion, rows: list[Evidence], got: dict[str, Any]
    ) -> None:
        allowed = {e.id for e in rows}
        for row in got.get("claims", []):
            ids = list(dict.fromkeys(row["evidence_ids"]))
            # The schema restricts ids to retrieved evidence, but decoding constraints are enforced
            # by the runtime, not by this code. An id outside the evidence table is dropped here.
            stray = [i for i in ids if i not in allowed]
            ids = [i for i in ids if i in allowed]
            if stray:
                st.notes.append(f"{sq.id}: synthesis cited unknown evidence id(s) {stray}; dropped")
            if not ids:
                continue
            st.claims.append(
                Claim(
                    f"{sq.id}-C{sum(c.subquestion == sq.id for c in st.claims) + 1}",
                    sq.id,
                    row["text"].strip(),
                    ids,
                )
            )
        if not got.get("claims") and sq.kind != "falsification":
            st.insufficient.append(sq.id)

    # -- stage 9: critique produces searches, not rewrites ---------------------------------
    def critique(self, st: State) -> list[dict[str, Any]]:
        assert st.plan
        problems = []
        for sq in st.plan.subquestions:
            if sq.id in st.insufficient:
                problems.append(f"{sq.id}: no usable evidence was retrieved.")
        for c in st.claims:
            works = {st.evidence[e].work for e in c.evidence_ids}
            if len(works) == 1:
                problems.append(f"{c.id}: rests on a single paper.")
            if all(st.evidence[e].secondhand for e in c.evidence_ids):
                problems.append(f"{c.id}: rests only on second-hand descriptions of other work.")
            designs = {st.evidence[e].study_type for e in c.evidence_ids}
            for rule in self.policy.critique_rules:
                if rule.triggers(designs):
                    problems.append(f"{c.id}: {rule.problem}")
        if len({e.direction for e in st.evidence.values()} & {"affirms", "denies"}) < 2:
            problems.append(
                "All retrieved results point the same way. That is a search result, "
                "not proof that no opposing result exists."
            )
        draft = (
            "\n".join(
                f"[{c.id}] ({len({st.evidence[e].work for e in c.evidence_ids})} paper(s)) {c.text}"
                for c in st.claims
            )
            or "(no claims could be written)"
        )
        ids = [sq.id for sq in st.plan.subquestions if sq.kind != "falsification"]
        subs = "\n".join(
            f"{sq.id}: {sq.text}" for sq in st.plan.subquestions if sq.kind != "falsification"
        )
        got = self.llm.chat_json(
            "critique",
            self.prompts.gap_system,
            f"Question: {st.question}\n\nSub-questions:\n{subs}\n\nDraft claims:\n{draft}\n\n"
            f"Problems already detected:\n" + "\n".join(f"- {p}" for p in problems),
            self.prompts.gap_schema(ids),
        )
        gaps = [g for g in got.get("gaps", []) if g.get("query", "").strip()]
        st.gaps.append({"problems": problems, "gaps": gaps})
        return gaps

    # -- stage 10: verification and grading ------------------------------------------------
    def verify_and_grade(self, st: State, verifiers: list[str]) -> None:
        self.progress("verify", f"{len(st.claims)} claims x {len(verifiers)} verifier(s)")
        verify.check_claims(
            st.claims,
            st.evidence,
            st.passages,
            self.llm,
            verifiers,
            prompts=self.prompts,
            parallel=self._parallel,
        )
        papers = {e.work: self.papers.get(e.work, {}) for e in st.evidence.values()}
        if not self.offline:
            cited = {st.evidence[e].work for c in st.claims for e in c.supported_by}
            self.progress("integrity", f"checking {len(cited)} cited work(s) against Crossref")
            st.integrity = integrity.check_all([d for w in cited if (d := papers[w].get("doi"))])
        for c in st.claims:
            states = {
                st.integrity.get(papers[st.evidence[e].work].get("doi") or "", {}).get("status")
                for e in c.supported_by
            }
            if "retracted" in states:
                # A retracted paper supports nothing, however well the passage matches.
                c.verdict, c.supported_by = "RETRACTED_SOURCE", []
            elif "partially_retracted" in states:
                c.flags.append("a cited source is partially retracted")
            elif "expression_of_concern" in states:
                c.flags.append("a cited source carries an expression of concern")
        opposite = {"affirms": "denies", "denies": "affirms"}
        for c in st.claims:
            mine = {st.evidence[e].direction for e in c.evidence_ids}
            against = {opposite[d] for d in mine if d in opposite}
            own_works = {st.evidence[e].work for e in c.evidence_ids}
            # Contested means another paper's own result points the other way. A paper's summary
            # of earlier positive reports does not contest its own null finding.
            c.contested_by = [
                e.id
                for e in st.evidence.values()
                if e.subquestion == c.subquestion
                and e.role == "E"
                and e.work not in own_works
                and e.direction in against
            ]
            contested = bool(c.contested_by) or {"affirms", "denies"} <= mine or "mixed" in mine
            c.label, reasons = grading.label(c, st.evidence, papers, contested, self.policy)
            c.flags.extend(reasons)

    def summarise(self, st: State, verifier: str) -> None:
        usable = [c for c in st.claims if c.verdict in ("SUPPORTED", "DISPUTED")]
        if not usable:
            return
        listing = "\n".join(
            f"[{c.id}] ({c.label} evidence{', disputed' if c.verdict == 'DISPUTED' else ''}"
            f"{', found by the search for null results' if c.subquestion.startswith('F') else ''})"
            f" {c.text}"
            for c in usable
        )
        got = self.llm.chat_json(
            "summarise",
            self.prompts.summary_system,
            f"Question: {st.question}\n\nVerified claims:\n{listing}",
            self.prompts.summary_schema([c.id for c in usable]),
        )
        by_id = {c.id: c for c in usable}
        rows = got.get("sentences", [])
        bases = [" ".join(by_id[i].text for i in row["claim_ids"] if i in by_id) for row in rows]
        verdicts = self._parallel(
            [
                partial(verify.judge, self.llm, verifier, row["text"], basis, self.prompts)
                for row, basis in zip(rows, bases, strict=True)
            ]
        )
        for row, (verdict, reason) in zip(rows, verdicts, strict=True):
            # A summary sentence that outruns its claims is dropped, not softened.
            st.summary.append(
                {
                    "text": row["text"].strip(),
                    "claim_ids": row["claim_ids"],
                    "verdict": verdict,
                    "reason": reason,
                    "kept": verdict == "SUPPORTED",
                }
            )

    # -- the run ---------------------------------------------------------------------------
    async def run(self, question: str) -> dict[str, Any]:
        started = time.time()
        st = State(question.strip())
        verifiers = self._resolve_models(st)

        async def gather(lib: PaperLibrary, queries: list[str]) -> None:
            if self.offline:
                return
            await self.discover(st, lib, queries)
            await self.acquire(st, lib)

        async def work(
            lib: PaperLibrary, targets: list[tuple[SubQuestion, list[str] | None]]
        ) -> None:
            pools = await self.pool(st, lib, targets)
            ranked = [(sq, self.select(sq, pools[sq.id])) for sq, _ in targets]
            works = [p.work for _, chosen in ranked for p, _ in chosen]
            openings = await self.openings(lib, works)
            await asyncio.to_thread(self.extract_many, st, ranked, openings)
            await asyncio.to_thread(self.synthesise_many, st, [sq for sq, _ in targets])

        async with PaperLibrary(self.s.gateway_config, self.s.papers_upstream) as lib:
            # The field's search half (indexed terms, measures, advice) is the library's profile
            # of the same name; the evidence half (designs, grading, critique) is this policy.
            if self.policy.profile:
                self.profile = await lib.profile(self.policy.profile)
                if self.profile is None:
                    st.notes.append(
                        f"The paper library has no '{self.policy.profile}' profile; "
                        "queries were planned without the field's indexed terms."
                    )
            # Companion tools reach outside the library, so an offline run has none.
            self.companions = {} if self.offline else companions.configured(self.s.gateway_config)
            self.prompts = Prompts(self.policy, self.profile, self.companions)

            self.progress("plan", f"decomposing the question ({self.policy.label})")
            await asyncio.to_thread(self.plan, st)
            plan = st.plan
            assert plan is not None
            if self.offline:
                st.notes.append(
                    "Offline run: answered from the already-indexed corpus; no discovery or "
                    "fetching."
                )

            await self.run_companions(st, lib)
            await gather(
                lib, list(dict.fromkeys(q for sq in plan.subquestions for q in sq.queries))
            )
            # Evidence passes first; the falsification pass then skips passages they already used.
            await work(lib, [(sq, None) for sq in plan.subquestions if sq.kind != "falsification"])
            await work(lib, [(sq, None) for sq in plan.subquestions if sq.kind == "falsification"])
            by_id = {sq.id: sq for sq in plan.subquestions}
            for round_no in range(2, self.s.max_rounds + 1):
                self.progress("critique", f"round {round_no}: looking for gaps and contradictions")
                gaps = await asyncio.to_thread(self.critique, st)
                if not gaps:
                    break
                await gather(lib, [g["query"] for g in gaps])
                await work(
                    lib,
                    [
                        (by_id[sid], [g["query"] for g in gaps if g["subquestion"] == sid])
                        for sid in dict.fromkeys(g["subquestion"] for g in gaps)
                        if sid in by_id
                    ],
                )
                # Newly fetched papers can also bear on the falsification pass.
                await work(lib, [(by_id["F1"], None)])

        await asyncio.to_thread(self.verify_and_grade, st, verifiers)
        await asyncio.to_thread(self.summarise, st, verifiers[0])
        self.models["embedding"] = {
            "name": self.index_stats.get("embedding_model"),
            "via": "paper library",
        }
        if not self.library_reranker:
            st.notes.append(
                "The paper library has no reranking model (PAPER_FETCH_RERANK_MODEL); passages "
                "were read in retrieval order."
            )

        from .render import render  # late import: render depends on this module's State

        answer = render(st, self.papers, self.models, self.policy, self.companions)
        log = {
            "pipeline_version": __version__,
            "prompt_version": self.prompts.version,
            "domain": {
                "name": self.domain.name,
                "version": self.domain.version,
                "module": self.domain.module,
                "policy": self.policy.summary(),
                "profile": self.profile,
            },
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "seconds": round(time.time() - started, 1),
            "question": st.question,
            "model_backend": "mcp-client" if self.client else "ollama",
            "models": self.models,
            "reranker": self.library_reranker,
            "collection": self.collection or None,
            "companions": {
                "available": sorted(self.companions),
                "requested": st.requests,
                "results": st.companions,
            },
            "settings": {
                k: (str(v) if not isinstance(v, (int, str)) else v) for k, v in vars(self.s).items()
            },
            "index": self.index_stats,
            "plan": to_json(st.plan),
            "searches": st.searches,
            "fetches": st.fetches,
            "candidates": to_json(list(st.candidates.values())),
            "evidence": to_json(list(st.evidence.values())),
            "passages": st.passages,
            "papers": {
                work: {k: paper.get(k) for k in ("doi", "title", "year", "text_sha256", "route")}
                for work in {e.work for e in st.evidence.values()}
                if (paper := self.papers.get(work))
            },
            "claims": to_json(st.claims),
            "insufficient_subquestions": st.insufficient,
            "critique": st.gaps,
            "dropped_passages": st.dropped_passages,
            "summary": st.summary,
            "integrity": st.integrity,
            "notes": st.notes,
            "llm_calls": self.llm.calls,
            "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "answer": answer,
        }
        log["run_dir"] = str(write_run(self.s, log))
        return log


def write_run(settings: Settings, log: dict[str, Any]) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", log["question"].lower()).strip("-")[:48] or "run"
    run_dir = settings.runs_dir / f"{log['date'].replace(':', '')}-{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "answer.md").write_text(log["answer"])
    (run_dir / "run.json").write_text(json.dumps(log, indent=1, ensure_ascii=False))
    return run_dir
