"""The research run: a fixed sequence of stages, with the model called only inside them.

    plan -> discover -> acquire+index -> retrieve -> rerank -> extract (evidence table)
         -> synthesise -> critique (new searches, not rewrites) -> verify -> grade -> render -> log

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

import numpy as np

from . import __version__, grading, integrity, rerank, safety, verify
from .client_llm import ClientLLM
from .config import Settings
from .domains import Domain, active
from .index import Passage, PassageIndex
from .llm import Ollama
from .papers import PaperLibrary, PapersError
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
    seen: dict[str, set[int]] = field(default_factory=dict)  # subquestion id -> passage ids judged
    claims: list[Claim] = field(default_factory=list)
    insufficient: list[str] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    dropped_passages: list[dict[str, Any]] = field(default_factory=list)
    summary: list[dict[str, Any]] = field(default_factory=list)
    integrity: dict[str, dict[str, Any]] = field(default_factory=dict)  # doi -> Crossref status
    fetches: int = 0  # library fetches attempted this run, against PIPELINE_MAX_FETCH
    notes: list[str] = field(default_factory=list)


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        progress: Progress | None = None,
        offline: bool = False,
        domain: str | Domain | None = None,
        llm: Model | None = None,
    ):
        self.s = settings
        self.offline = offline  # answer from the already-indexed corpus; no discovery, no fetching
        self.progress = progress or (lambda stage, detail: None)
        self.llm: Model = llm or Ollama(settings)
        self.client = isinstance(self.llm, ClientLLM)
        # A client answers a batch per turn; a single resident local model answers one at a time.
        self.concurrency = 32 if self.client else 1
        self.index = PassageIndex(settings.index_path, settings.embedding_model)
        self.models: dict[str, Any] = {}
        # The field being researched. Nothing installed means the generic policy, not a failure:
        # the engine answers a question on its own, it just answers it without a field's rules.
        self.domain = domain if isinstance(domain, Domain) else active(domain or settings.domain)
        self.policy = self.domain.policy
        self.prompts = Prompts(self.policy)
        # discover()/acquire() fetch and search concurrently, but only one model/DB write at a
        # time: the embedding model is single-resident and the index is one sqlite connection.
        self._index_lock = asyncio.Lock()

    # -- setup -----------------------------------------------------------------------------
    def _resolve_models(self, st: State) -> list[str]:
        name = self.llm.name if isinstance(self.llm, ClientLLM) else self.s.text_model
        text, digest = self.llm.resolve(name)
        self.s.text_model = text
        self.models["text"] = {"name": text, "digest": digest}
        _, embed_digest = self.llm.resolve(self.s.embedding_model)
        self.models["embedding"] = {"name": self.s.embedding_model, "digest": embed_digest}
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

    # -- stage 3-4: discover and normalise -------------------------------------------------
    async def discover(self, st: State, lib: PaperLibrary, queries: list[str]) -> None:
        # Each query is an independent provider round-trip; running them concurrently (bounded, so
        # a slow/rate-limited provider does not stall the rest) only shortens wall-clock time.
        sem = asyncio.Semaphore(self.s.search_concurrency)

        async def _search_one(query: str) -> tuple[str, dict[str, Any] | None, PapersError | None]:
            self.progress("discover", query)
            async with sem:
                try:
                    return query, await lib.search(query, limit=self.s.hits_per_query), None
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
                    "providers": res["providers"],
                    "n_hits": len(res["hits"]),
                    "did_not_answer": silent,
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

    # -- acquire: fetch through the library, index full text -------------------------------
    async def acquire(self, st: State, lib: PaperLibrary) -> None:
        assert st.plan
        fresh = [c for c in st.candidates.values() if c.outcome in ("not_attempted", "over_budget")]
        if not fresh:
            return
        # Rank what to read by how close its title sits to a sub-question, so a small fetch budget
        # is spent on the most relevant candidates rather than on whatever a provider listed first.
        unscored = [c for c in fresh if not c.relevance]
        if unscored:
            targets = [sq.text for sq in st.plan.subquestions]
            vecs = np.asarray(
                await asyncio.to_thread(
                    self.llm.embed, targets + [c.title or "" for c in unscored]
                ),
                dtype=np.float32,
            )
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
            for cand, row in zip(
                unscored, vecs[len(targets) :] @ vecs[: len(targets)].T, strict=True
            ):
                cand.relevance = round(float(row.max()), 4)
        fresh.sort(key=lambda c: -c.relevance)

        # Decide who gets fetched, and consume the budget, before any awaiting: this keeps the
        # same priority order and the same PIPELINE_MAX_FETCH accounting as the sequential version.
        to_fetch: list[tuple[Candidate, str]] = []
        for cand in fresh:
            if cand.work and self.index.has(cand.work):
                cand.outcome = "indexed"
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
        if not to_fetch:
            return

        sem = asyncio.Semaphore(self.s.fetch_concurrency)

        async def _fetch_one(cand: Candidate, ident: str) -> None:
            self.progress("fetch", f"{cand.title or ident}"[:90])
            try:
                async with sem:
                    rec = await lib.fetch(ident)
                cand.work = rec.get("work") or cand.work or ident
                if rec.get("is_retracted"):
                    cand.outcome = "retracted"
                    return
                if not rec.get("full_text"):
                    cand.outcome = "not_obtainable"
                    return
                async with sem:
                    text = await lib.full_text(rec.get("work") or ident)
                # Hidden characters are counted on the raw text: clean() removes them before
                # indexing, so a passage-level scan afterwards could never see them.
                if hidden := safety.scan_hidden(text):
                    st.notes.append(f"{cand.work}: {hidden}; stripped before indexing")
                # PassageIndex is one sqlite connection with no internal lock, and embedding shares
                # the single resident model: serialise this part although fetch runs concurrently.
                async with self._index_lock:
                    n = await asyncio.to_thread(
                        self.index.add, rec, safety.clean(text), self.llm.embed
                    )
                cand.outcome = "indexed"
                self.progress("index", f"{n} passages from {cand.work}")
            except PapersError as exc:
                cand.outcome = "not_obtainable" if exc.code == "not_found" else "fetch_failed"
                st.notes.append(f"fetch {ident}: {exc}")

        await asyncio.gather(*(_fetch_one(cand, ident) for cand, ident in to_fetch))

    # -- stage 5: retrieve and rerank ------------------------------------------------------
    def pool(
        self, st: State, targets: list[tuple[SubQuestion, list[str] | None]]
    ) -> dict[str, list[Passage]]:
        """Hybrid retrieval for several sub-questions at once.

        All embedding happens here in one batch, before any generation: with a single model
        resident in memory, interleaving embedding and chat calls would reload weights every time.
        """
        plans = {
            sq.id: (
                queries or ([sq.text, *sq.queries] if sq.kind != "falsification" else sq.queries)
            )
            for sq, queries in targets
        }
        flat = list(dict.fromkeys(q for qs in plans.values() for q in qs))
        vectors = dict(zip(flat, self.llm.embed(flat), strict=True))
        used_elsewhere = {e.passage_id for e in st.evidence.values()}
        out: dict[str, list[Passage]] = {}
        for sq, _ in targets:
            found: dict[int, Passage] = {}
            for text in plans[sq.id]:
                for p in self.index.search(
                    text, vectors[text], limit=self.s.candidates_per_subquestion
                ):
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
                paper = self.index.paper(p.work) or {}
                flags = safety.scan(p.text) + (["retracted"] if paper.get("is_retracted") else [])
                if flags:
                    st.dropped_passages.append({"passage_id": p.id, "work": p.work, "flags": flags})
                    continue
                kept.append(p)
            out[sq.id] = kept
        return out

    def select(
        self, sq: SubQuestion, kept: list[Passage], ranker: rerank.Reranker
    ) -> list[tuple[Passage, float]]:
        if not kept:
            return []
        self.progress("rerank", f"{sq.id}: {len(kept)} candidates")
        # The reranker sees passage text only: no year, venue or citation count to be biased by.
        ranked = sorted(zip(kept, ranker.score(sq.text, kept), strict=True), key=lambda t: -t[1])
        out: list[tuple[Passage, float]] = []
        per_paper: dict[str, int] = {}
        for p, score in ranked:
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

    # Classification is split so that only the model call runs concurrently: the index is one
    # SQLite connection, and every read and write of it stays on the stage's own thread.
    def _classify_input(self, work: str) -> str | None:
        """The classifier's input for a paper, or None if its text was flagged (then noted)."""
        paper = self.index.paper(work) or {}
        opening = self.index.opening(work)
        if safety.scan(opening):
            # The classifier would read the injected text; record the paper as unclassified.
            self.index.set_note(work, "unclear", "", "safety-filter")
            return None
        return f"Title: {paper.get('title')}\n\n{wrap(opening)}"

    def _classify_call(self, user: str) -> dict[str, Any]:
        return self.llm.chat_json(
            "classify_paper", self.prompts.paper_system, user, self.prompts.paper_schema()
        )

    def _classify_record(self, work: str, got: dict[str, Any]) -> None:
        self.index.set_note(
            work, got["study_type"], got.get("population", "").strip(), self.s.text_model
        )

    def paper_note(self, work: str) -> dict[str, str]:
        if self.index.note(work) is None and (user := self._classify_input(work)) is not None:
            self._classify_record(work, self._classify_call(user))
        return self.index.note(work) or {"study_type": "unclear", "population": ""}

    # -- stage 6: evidence table -----------------------------------------------------------
    def extract(self, st: State, sq: SubQuestion, ranked: list[tuple[Passage, float]]) -> None:
        self.extract_many(st, [(sq, ranked)])

    def _read(self, sq: SubQuestion, p: Passage) -> dict[str, Any]:
        return self.llm.chat_json(
            "extract",
            self.prompts.extract_system,
            f"Research question: {sq.text}\n\n{wrap(p.text)}",
            self.prompts.EXTRACT_SCHEMA,
        )

    def extract_many(
        self, st: State, batches: list[tuple[SubQuestion, list[tuple[Passage, float]]]]
    ) -> None:
        """Read every selected passage of several sub-questions in one batch of model calls.

        Papers that have not been classified yet are classified in the same batch. Results are
        applied in order afterwards, so evidence ids do not depend on which call finished first.
        """
        items = [(sq, p, score) for sq, ranked in batches for p, score in ranked]
        for sq, p, _ in items:
            st.seen.setdefault(sq.id, set()).add(p.id)
        unclassified = list(
            dict.fromkeys(p.work for _, p, _ in items if not self.index.note(p.work))
        )
        self.progress("extract", f"{len(items)} passages, {len(unclassified)} new papers")
        to_classify = [(w, user) for w in unclassified if (user := self._classify_input(w))]
        jobs: list[Callable[[], Any]] = [partial(self._read, sq, p) for sq, p, _ in items]
        jobs += [partial(self._classify_call, user) for _, user in to_classify]
        answers = self._parallel(jobs)
        results = answers[: len(items)]
        for (work, _), got in zip(to_classify, answers[len(items) :], strict=True):
            self._classify_record(work, got)
        for (sq, p, score), got in zip(items, results, strict=True):
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
        papers = {e.work: (self.index.paper(e.work) or {}) for e in st.evidence.values()}
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
        # Reranking by a client model would cost a turn per four passages; it uses ONNX or none.
        ranker = rerank.build(
            self.s, None if isinstance(self.llm, ClientLLM) else self.llm, self.prompts
        )

        self.progress("plan", f"decomposing the question ({self.policy.label})")
        await asyncio.to_thread(self.plan, st)
        plan = st.plan
        assert plan is not None

        async def gather(lib: PaperLibrary | None, queries: list[str]) -> None:
            if lib is None:
                return
            await self.discover(st, lib, queries)
            await self.acquire(st, lib)

        async def body(lib: PaperLibrary | None) -> None:
            await gather(
                lib, list(dict.fromkeys(q for sq in plan.subquestions for q in sq.queries))
            )

            async def work(targets: list[tuple[SubQuestion, list[str] | None]]) -> None:
                pools = await asyncio.to_thread(self.pool, st, targets)
                ranked = [
                    (sq, await asyncio.to_thread(self.select, sq, pools[sq.id], ranker))
                    for sq, _ in targets
                ]
                await asyncio.to_thread(self.extract_many, st, ranked)
                await asyncio.to_thread(self.synthesise_many, st, [sq for sq, _ in targets])

            # Evidence passes first; the falsification pass then skips passages they already used.
            await work([(sq, None) for sq in plan.subquestions if sq.kind != "falsification"])
            await work([(sq, None) for sq in plan.subquestions if sq.kind == "falsification"])
            by_id = {sq.id: sq for sq in plan.subquestions}
            for round_no in range(2, self.s.max_rounds + 1):
                self.progress("critique", f"round {round_no}: looking for gaps and contradictions")
                gaps = await asyncio.to_thread(self.critique, st)
                if not gaps:
                    break
                await gather(lib, [g["query"] for g in gaps])
                await work(
                    [
                        (by_id[sid], [g["query"] for g in gaps if g["subquestion"] == sid])
                        for sid in dict.fromkeys(g["subquestion"] for g in gaps)
                        if sid in by_id
                    ]
                )
                # Newly fetched papers can also bear on the falsification pass.
                await work([(by_id["F1"], None)])

        if self.offline:
            st.notes.append(
                "Offline run: answered from the already-indexed corpus; no discovery or fetching."
            )
            await body(None)
        else:
            async with PaperLibrary(self.s.gateway_config, self.s.papers_upstream) as lib:
                await body(lib)

        await asyncio.to_thread(self.verify_and_grade, st, verifiers)
        await asyncio.to_thread(self.summarise, st, verifiers[0])

        from .render import render  # late import: render depends on this module's State

        answer = render(st, self.index, self.models, self.policy)
        log = {
            "pipeline_version": __version__,
            "prompt_version": self.prompts.version,
            "domain": {
                "name": self.domain.name,
                "version": self.domain.version,
                "module": self.domain.module,
                "policy": self.policy.summary(),
            },
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "seconds": round(time.time() - started, 1),
            "question": st.question,
            "model_backend": "mcp-client" if self.client else "ollama",
            "models": self.models,
            "reranker": ranker.name,
            "settings": {
                k: (str(v) if not isinstance(v, (int, str)) else v) for k, v in vars(self.s).items()
            },
            "index": self.index.stats(),
            "plan": to_json(st.plan),
            "searches": st.searches,
            "fetches": st.fetches,
            "candidates": to_json(list(st.candidates.values())),
            "evidence": to_json(list(st.evidence.values())),
            "passages": st.passages,
            "papers": {
                work: {k: paper.get(k) for k in ("doi", "title", "year", "text_sha256", "route")}
                for work in {e.work for e in st.evidence.values()}
                if (paper := self.index.paper(work))
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
