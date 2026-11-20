"""`ask`: one entry point that routes a request, runs only what it needs, and returns a relay.

The chat model's job is to call `ask` and show `relay` to the person unchanged. The gateway does
the work: a passage retrieval for a question, the research pipeline for an evidence question, and
retrieval plus the coder's tools for a task or code review (routing.py decides which). The relay
says what ran and why, what came back, and its limits, so the chat model has nothing to fill in
from memory, and any chat model can carry the work on.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_mcp import brief, extensions
from research_mcp.extensions import Context, Extension, StepInput, StepSpec
from research_mcp.hub.contract import (
    AUTONOMY,
    CHOICES,
    COMPONENTS,
    MUST_ASK,
    NO_FALLBACK,
    Problem,
    first_sentence,
    needs_decision,
    restart_hint,
)
from research_mcp.hub.health import Hub
from research_mcp.routing import (
    DOI,
    Route,
    Rules,
    _has,
    _looks_like_code,
    classify,
    load_rules,
    profile_override,
)
from research_pipeline import agent_profiles, replies
from research_pipeline.papers import PapersError

# The library's reranker (a bge-reranker cross-encoder) scores each passage as a logit. Passages
# under the floor are not shown, so a request the library cannot answer gets "nothing relevant"
# rather than unrelated text. ask reranks against the whole question, which scores lower than a
# short query: a passage holding the answer scored +0.14 that way and +1.47 as a short query, and
# near-misses sit around -1. Unrelated text scored -4 to -6, so the floor sits between, at -3.
# Applied only when the library reports a reranker: without one the score is a fused rank on
# another scale.
RELEVANCE_FLOOR = -3.0
QUICK_PASSAGES = 5
CODE_PASSAGES = 3
EXCERPT = 320
# Under a client's 60 s MCP timeout, with room for the fallback after it.
RETRIEVE_SECONDS = 40.0
# Code work on the same repository from another session within this window is mentioned.
OVERLAP_SECONDS = 2 * 3600
RETRIEVAL_ERRORS = (PapersError, ValueError, KeyError, OSError, RuntimeError)
# Every quick answer also searches outside the library, alongside the passage retrieval, through
# SearXNG: its science engines (arXiv, PubMed, Europe PMC, Semantic Scholar...) for papers, and the
# general web. Not through the paper library's own search: its tools run one at a time, so a 30 s
# search there held up every retrieval behind it (November 2026). What comes back are leads, never
# evidence: titles and links the answer can point to, especially when the library falls short.
FRESH_PAPERS = 8
FRESH_WEB = 6
# SearXNG answers in a second or two; a search past this is dropped and named.
FRESH_SECONDS = 20.0
# A fetch by title: the top hit is indexed, the other hits of this search are offered.
FETCH_HITS = 3
PMID = re.compile(r"\bpmid:\s*(\d+)", re.IGNORECASE)
FETCH_TRAILING = ".,;:)]}'\""
INDEXED = ("indexed", "already_indexed")

Retrieve = Callable[[str, int, int], Awaitable[dict[str, Any]]]
# Retrieval with one part of the library's ranking off ("reranker" or "embeddings").
PlainRetrieve = Callable[[str, int, int, str], Awaitable[dict[str, Any]]]
Search = Callable[[str, int], Awaitable[dict[str, Any]]]
WebSearch = Callable[[str, int, str], Awaitable[list[dict[str, str]]]]  # query, limit, category
CallTool = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class Halt(Exception):
    """Stop this request with one problem (a decision to ask for, or a failure)."""

    def __init__(self, problem: Problem):
        super().__init__(problem.what_failed)
        self.problem = problem


@dataclass
class Turn:
    """One request's who-and-how: the session, its standing fallback choices, what degraded."""

    session: str
    decisions: dict[str, str]
    problems: list[Problem] = field(default_factory=list)
    rid: str = ""
    notes: list[str] = field(default_factory=list)
    ext_lines: list[tuple[list[str], list[str], list[str], str]] = field(default_factory=list)
    ext_limits: str = ""
    brief_notes: dict[str, str] = field(default_factory=dict)
    autonomy: str = "strict"


def paper_identifiers(text: str) -> list[str]:
    """The DOIs and PMIDs (pmid:...) a request names, in order, without repeats."""
    dois = [m.group(0).rstrip(FETCH_TRAILING) for m in DOI.finditer(text)]
    pmids = [f"pmid:{m.group(1)}" for m in PMID.finditer(text)]
    return list(dict.fromkeys([*dois, *pmids]))


def _without_phrases(text: str, phrases: Sequence[str]) -> str:
    """`text` with each trigger phrase taken out, whole words only, and its spaces tidied."""
    for phrase in phrases:
        text = re.sub(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", " ", text, flags=re.I)
    return " ".join(text.split())


def _pipeline_full(error: str) -> bool:
    """The pipeline turned a run away because it already carries PIPELINE_MAX_RUNS."""
    low = error.lower()
    return "runs" in low and ("limit" in low or "at once" in low)


def _sentence(text: str) -> str:
    """`text` with its first letter capitalised and a full stop, the rest untouched."""
    text = text.strip()
    return (text[:1].upper() + text[1:]).rstrip(".") + "." if text else text


def _blame(message: str) -> str:
    """Which part of retrieval an error message points at."""
    low = message.lower()
    if any(w in low for w in ("embed", "ollama", "bge", "11434")):
        return "embeddings"
    if any(w in low for w in ("rerank", "cross-encoder", "crossencoder")):
        return "reranker"
    return "library"


class Asker:
    def __init__(
        self,
        *,
        retrieve: Retrieve,
        call_tool: CallTool,
        has_tool: Callable[[str], bool],
        rules_paths: tuple[Path, Path | None],
        profiles_dir: Path,
        extensions_dir: Path | None,
        hub: Hub | None = None,
        search: Search | None = None,
        web: WebSearch | None = None,
        plain_retrieve: PlainRetrieve | None = None,
        can_switch: Callable[[str], bool] = lambda _component: True,
        steps: Sequence[tuple[Extension, StepSpec, Context]] = (),
        default_autonomy: str = "delegate",
    ):
        if default_autonomy not in AUTONOMY:
            raise ValueError(f"autonomy must be one of {', '.join(AUTONOMY)}")
        self.default_autonomy = default_autonomy
        self.retrieve = retrieve
        self.call_tool = call_tool
        self.has_tool = has_tool
        self.rules_paths = rules_paths
        self.profiles_dir = profiles_dir
        self.extensions_dir = extensions_dir
        self.hub = hub
        self.search = search
        self.web = web
        self.plain_retrieve = plain_retrieve
        self.can_switch = can_switch
        self.steps = tuple(steps)
        # Requests still working after their reply's deadline: held so they finish.
        self._tasks: set[asyncio.Task[dict[str, Any]]] = set()

    def rules(self) -> Rules:
        # Read on every call, so an edit to config/routing.toml applies without a restart.
        return load_rules(*self.rules_paths)

    def specialist(self, question: str, rules: Rules) -> dict[str, Any]:
        """The specialists for a question: up to three, those the person's own words or profile
        names recruit first, then the best-scoring ones (agent_profiles.route)."""
        own = profile_override(question, rules)
        route = agent_profiles.route(
            question,
            self.profiles_dir,
            "",
            self.extensions_dir,
            requested=[own[0]] if own else (),
        )
        names = list(route.get("specialists") or [])
        guidances = route.get("guidances") or {}
        paths = {p.get("name"): p for p in route.get("profiles", [])}
        reasons = str(route.get("reason") or "").split("; ")
        why = reasons[0] if reasons else ""
        if own and names and names[0] == own[0]:
            why = f"your word '{own[1]}'"
        listed = [
            {
                "name": n,
                "guidance": guidances.get(n, ""),
                "brief": paths.get(n, {}).get("path", ""),
                "repository": paths.get(n, {}).get("repository", ""),
            }
            for n in names
        ]
        first = listed[0] if listed else {}
        out: dict[str, Any] = {
            "name": first.get("name"),
            "why": why,
            "roles": route.get("roles", ["researcher"]),
            "brief": first.get("brief", ""),
            "repository": first.get("repository", ""),
            "guidance": first.get("guidance", ""),
            "specialists": listed,
            "reason": route.get("reason", ""),
        }
        if route.get("unknown_requested"):
            out["unknown"] = route["unknown_requested"]
        return out

    def profiles(self) -> list[dict[str, str]]:
        """Every specialist profile a question can recruit."""
        return agent_profiles.catalogue(self.profiles_dir, self.extensions_dir)

    # -- failures and fallbacks ----------------------------------------------------------------
    def _retry(self, component: str) -> int | None:
        if self.hub is None:
            return None
        breaker = self.hub.breakers[component]
        return breaker.retry_after() or round(breaker.cooldown)

    def _broken(
        self, turn: Turn, component: str, what_failed: str, fallback: str | None = None
    ) -> None:
        """A component failed. Fall back if this session chose that; otherwise stop and say why.

        The first failure of a component in a session asks how to handle it; the answer holds for
        the rest of the session (decided 2026-10-28)."""
        retry = self._retry(component)
        if component in NO_FALLBACK:
            raise Halt(
                Problem(
                    "failed",
                    component,
                    what_failed,
                    recommendation=f"restart the gateway: `{restart_hint()}`",
                    retry_after=retry,
                )
            )
        choice = turn.decisions.get(component)
        if choice is None and turn.autonomy == "delegate" and component not in MUST_ASK:
            # delegate: take the labelled fallback, unless the component must ask
            choice = "fallback"
        if choice is None:
            raise Halt(needs_decision(component, what_failed, retry))
        if choice == "stop":
            raise Halt(
                Problem(
                    "failed",
                    component,
                    what_failed,
                    recommendation="this session chose to stop when this fails; ask again later, "
                    f'or pass decisions={{"{component}": "fallback"}} to continue without it.',
                    retry_after=retry,
                )
            )
        turn.problems.append(
            Problem(
                "degraded",
                component,
                what_failed,
                fallback_used=fallback or COMPONENTS[component].fallback,
                decided="this session's choice",
            )
        )

    def _notice(self, turn: Turn, found: dict[str, Any]) -> None:
        """Label a part of ranking the library dropped by itself since it last worked."""
        assert self.hub is not None
        seen = self.hub.seen
        if found.get("reranker"):
            seen["reranker"] = str(found["reranker"])
            self.hub.breakers["reranker"].success()
        elif seen.get("reranker"):
            turn.problems.append(
                Problem(
                    "degraded",
                    "reranker",
                    f"The library returned passages without its reranker ({seen['reranker']}).",
                    fallback_used=COMPONENTS["reranker"].fallback,
                    decided="detected",
                )
            )
        if found.get("retrieval") == "hybrid":
            seen["retrieval"] = "hybrid"
            self.hub.breakers["embeddings"].success()
        elif seen.get("retrieval") == "hybrid" and found.get("retrieval") == "lexical":
            turn.problems.append(
                Problem(
                    "degraded",
                    "embeddings",
                    "The library returned lexical results only: its embeddings did not answer.",
                    fallback_used=COMPONENTS["embeddings"].fallback,
                    decided="detected",
                )
            )

    async def _evidence(self, turn: Turn, query: str, limit: int) -> tuple[str, dict[str, Any]]:
        """("evidence", passages), or after a failure the session's fallback ("search", hits)."""
        hub = self.hub
        if hub is None:
            return "evidence", await self.retrieve(query, limit, 1800)
        paused = next(
            (c for c in ("library", "embeddings", "reranker") if not hub.breakers[c].allow()), None
        )
        if paused:
            breaker = hub.breakers[paused]
            component = paused
            error = (
                f"It is skipped for {breaker.retry_after()} s more after {breaker.count} failures "
                f"in a row (last: {breaker.last_error})"
            )
        else:
            try:
                # One of a few at once (the reranker shares the CPU); the time limit starts once
                # this request's turn comes.
                async with hub.lanes.retrieval.slot(turn.rid):
                    found = await asyncio.wait_for(
                        self.retrieve(query, limit, 1800), RETRIEVE_SECONDS
                    )
            except TimeoutError:
                component = "reranker" if hub.seen.get("reranker") else "library"
                error = f"Retrieval took longer than {RETRIEVE_SECONDS:.0f} s."
            except RETRIEVAL_ERRORS as exc:
                component = _blame(str(exc))
                error = f"Retrieval failed: {type(exc).__name__}: {exc}"[:400]
            else:
                hub.breakers["library"].success()
                self._notice(turn, found)
                return "evidence", found
            hub.breakers[component].failure(error)
        if (
            component in ("reranker", "embeddings")
            and self.plain_retrieve is not None
            and self.can_switch(component)
        ):
            self._broken(turn, component, error)
            try:
                found = await asyncio.wait_for(
                    self.plain_retrieve(query, limit, 1800, component), RETRIEVE_SECONDS
                )
                hub.breakers["library"].success()
                return "evidence", found
            except (TimeoutError, *RETRIEVAL_ERRORS) as exc:
                turn.problems.pop()
                error += f" Retrieval without it failed too ({type(exc).__name__})."
                component = "library"
        spec = COMPONENTS["library"]
        if component == "library":
            self._broken(turn, "library", error)
        else:
            self._broken(
                turn,
                component,
                error,
                fallback=f"{spec.fallback} (the library cannot switch off its {component} yet)",
            )
            turn.problems[-1].impact = spec.impact
        if self.search is None:
            raise Halt(Problem("failed", "library", error + " No scholarly search is connected."))
        try:
            return "search", await asyncio.wait_for(self.search(query, limit), RETRIEVE_SECONDS)
        except (TimeoutError, *RETRIEVAL_ERRORS) as exc:
            raise Halt(
                Problem(
                    "failed",
                    "library",
                    f"{error} The scholarly search fallback failed too ({type(exc).__name__}).",
                    recommendation="check `hub__status`; the library may need a gateway restart: "
                    f"`{restart_hint()}`",
                    retry_after=self._retry("library"),
                )
            ) from exc

    async def _fresh(self, query: str, papers: bool) -> dict[str, Any]:
        """A fresh paper search (SearXNG's science engines) and a web search, run at once. Each
        part that fails or is not set up is named in `not_run`; neither ever stops the answer."""
        out: dict[str, Any] = {"papers": [], "web": [], "not_run": []}
        if self.web is None:
            out["not_run"].append("web and paper search: no SearXNG instance is configured")
            return out
        web = self.web
        jobs = {"web": ("general", FRESH_WEB)}
        if papers:
            jobs["papers"] = ("science", FRESH_PAPERS)
        got = await asyncio.gather(
            *(asyncio.wait_for(web(query, n, cat), FRESH_SECONDS) for cat, n in jobs.values()),
            return_exceptions=True,
        )
        for key, value in zip(jobs, got, strict=True):
            label = "paper search" if key == "papers" else "web search"
            if isinstance(value, BaseException):
                why = "timed out" if isinstance(value, TimeoutError) else type(value).__name__
                out["not_run"].append(f"{label}: {why}")
            else:
                out[key] = value
        return out

    async def _duplicate(self, turn: Turn, question: str) -> dict[str, Any] | None:
        """Attach to a research request on the go that asks the same thing; else None. A near
        match is noted on the turn and the new run starts anyway."""
        hub = self.hub
        if hub is None:
            return None
        others = hub.ledger.research_on_the_go(exclude=turn.rid)
        row, score, how = await hub.overlap.closest(question, others)
        if row is None:
            return None
        if score >= hub.overlap.attach:
            return {
                "attached_to": row["id"],
                "run_id": row.get("run_id"),
                "status": "working",
                "similarity": score,
                "how": how,
                "session": row.get("session"),
                "question": row.get("question"),
            }
        if score >= hub.overlap.warn:
            other = str(row.get("question"))[:120]
            turn.notes.append(
                f"A similar research request is on the go (`{row['id']}`, {how} {score}, session "
                f"{row.get('session') or 'unknown'}): \u201c{other}\u201d. This one runs as well."
            )
        return None

    async def _start_run(self, turn: Turn, args: dict[str, Any]) -> dict[str, Any]:
        """Start a run. When the pipeline is full or memory is short, wait in line (oldest
        first) and try again, for at most the configured wait."""
        hub = self.hub
        line = hub.lanes.research if hub is not None else None
        waited = time.monotonic()
        try:
            while True:
                mine = line is not None and turn.rid in line
                ahead = mine and line is not None and line[0] != turn.rid
                low_memory = hub is not None and hub.lanes.short_of_memory()
                if not ahead and not low_memory:
                    try:
                        run = await self.call_tool("pipeline__research_start", args)
                    except Exception as exc:  # a dead upstream surfaces as many types
                        if hub is not None:
                            hub.breakers["upstream"].failure(str(exc))
                        self._broken(
                            turn,
                            "upstream",
                            f"The research pipeline did not answer: {type(exc).__name__}.",
                        )
                        raise AssertionError("unreachable") from exc
                    if not (run.get("error") and _pipeline_full(str(run["error"]))):
                        return run
                    if hub is None or line is None:
                        return run
                assert hub is not None
                assert line is not None
                if not mine:
                    line.append(turn.rid)
                if time.monotonic() - waited > hub.research_wait:
                    raise Halt(
                        Problem(
                            "refused",
                            "upstream",
                            f"The research run waited {hub.research_wait / 60:.0f} min for a "
                            "free slot and did not start.",
                            impact="nothing started; the other runs carry on",
                            recommendation="check `hub__status` for stuck runs, then ask again.",
                            retry_after=60,
                        )
                    )
                await asyncio.sleep(hub.research_retry)
        finally:
            if line is not None and turn.rid in line:
                line.remove(turn.rid)

    async def _research(self, turn: Turn, question: str, strictness: str) -> dict[str, Any]:
        attached = await self._duplicate(turn, question)
        if attached is not None:
            return attached
        if not self.has_tool("pipeline__research_start"):
            why = ""
            if self.hub is not None:
                why = self.hub.upstreams()[1].get("pipeline", "")
            self._broken(
                turn,
                "upstream",
                "The research pipeline upstream is not connected" + (f" ({why})." if why else "."),
            )
        args: dict[str, Any] = {"question": question}
        if strictness:
            args["strictness"] = strictness
        hub = self.hub
        if hub is not None and hub.settings.endpoint_url:
            used, passed = await hub.first_usable("writing")
            ladder = hub.provider_ladder("writing")
            if used is None:
                raise Halt(
                    Problem(
                        "failed",
                        "provider",
                        "No provider on the ladder can run the model steps: "
                        + "; ".join(passed)
                        + ".",
                        recommendation="top up a provider, or start Ollama for the local model.",
                        retry_after=self._retry("provider"),
                    )
                )
            if ladder and used != ladder[0]:
                label = hub.config.get("providers", {}).get(used, {}).get("label", used)
                self._broken(
                    turn,
                    "provider",
                    _sentence("; ".join(passed)),
                    fallback=f"{label}, the next usable rung of the ladder",
                )
        if hub is not None:
            level = strictness or hub.settings.strictness
            if level != "explore":
                check = await hub.check("verifier")
                if check["state"] != "up":
                    self._broken(turn, "verifier", f"The verifier is down: {check['detail']}.")
                    args["strictness"] = "explore"
            corpus = await hub.check("corpus")
            if corpus["state"] != "up":
                self._broken(turn, "corpus", _sentence(corpus["detail"]))
        run = await self._start_run(turn, args)
        if run.get("error"):
            error = str(run["error"])
            if _pipeline_full(error):
                raise Halt(
                    Problem(
                        "refused",
                        "upstream",
                        f"The research run did not start: {error}",
                        impact="nothing started; the other runs carry on",
                        recommendation="wait for a run to finish (`hub__status` lists them), "
                        "then ask again.",
                        retry_after=60,
                    )
                )
            raise Halt(
                Problem(
                    "failed",
                    "upstream",
                    f"The research run did not start: {error}",
                    impact="nothing started",
                )
            )
        return run

    # -- a paper into the index, without the pipeline --------------------------------------------
    async def _index(self, identifiers: list[str]) -> list[dict[str, Any]]:
        """rag__index_paper on these identifiers: one result row each, saying what happened."""
        try:
            out = await self.call_tool("rag__index_paper", {"identifiers": identifiers})
        except Exception as exc:  # a dead upstream surfaces as many types
            raise Halt(
                Problem(
                    "failed",
                    "library",
                    f"The paper library did not index the paper: {type(exc).__name__}.",
                    recommendation=f"check `hub__status`; the library may need a restart: "
                    f"`{restart_hint()}`",
                )
            ) from exc
        return list(out.get("results") or [])

    async def _find_title(self, query: str) -> list[dict[str, Any]]:
        """The top hits of a title search in the paper library (at most FETCH_HITS)."""
        try:
            found = await self.call_tool("papers__search", {"query": query, "limit": FETCH_HITS})
        except Exception as exc:  # a dead upstream surfaces as many types
            raise Halt(
                Problem(
                    "failed",
                    "library",
                    f"The paper search did not answer: {type(exc).__name__}.",
                )
            ) from exc
        # An upstream tool replies with its envelope; the hits are in its `data`.
        return list((found.get("data") or {}).get("hits") or [])[:FETCH_HITS]

    async def _fetch(
        self, route: Route, *, rules: Rules, progress: list[str], results: dict[str, Any]
    ) -> None:
        """Get papers into the passage index without the research pipeline: the DOIs and PMIDs the
        request names, or else the top hit of a title search (the other hits are offered)."""
        ids = paper_identifiers(route.question)
        alternatives: list[dict[str, Any]] | None = None
        if ids:
            papers = [
                {
                    "title": r.get("title"),
                    "doi": r.get("doi") or r.get("identifier"),
                    "index": r,
                }
                for r in await self._index(ids)
            ]
        else:
            query = _without_phrases(route.question, rules.fetch)
            hits = await self._find_title(query) if query else []
            if not hits:
                results["fetch"] = {"papers": [], "note": "No paper was named or matched."}
                progress.append("Fetch: no DOI, PMID or matching title, so nothing was indexed")
                return
            top = hits[0]
            top_ids = top.get("ids") or {}
            key = top_ids.get("doi") or top_ids.get("openalex")  # the work id, without a DOI
            rows = await self._index([key]) if key else []
            papers = [
                {
                    "title": top.get("title"),
                    "doi": top_ids.get("doi"),
                    "index": rows[0] if rows else {"status": "no DOI or work id to index"},
                }
            ]
            alternatives = [
                {"title": h.get("title"), "doi": (h.get("ids") or {}).get("doi")} for h in hits[1:]
            ]
        fetch: dict[str, Any] = {"papers": papers}
        if alternatives is not None:
            fetch["alternatives"] = alternatives
            fetch["alternatives_note"] = "if this isn't the paper you meant"
        results["fetch"] = fetch
        done = sum(p["index"].get("status") in INDEXED for p in papers)
        progress.append(
            f"Fetch: indexed {done} of {len(papers)} paper(s) with rag__index_paper; "
            "no pipeline run"
        )

    def _overlap(self, turn: Turn, repo: str, touches: list[str]) -> str:
        """Recent code work on the same repository from another session (decided: worktrees)."""
        if self.hub is None or not repo:
            return ""
        for row in self.hub.ledger.recent(OVERLAP_SECONDS, repo=repo, route="code"):
            if row.get("session") == turn.session:
                continue
            theirs = set(row.get("touches") or [])
            if touches and theirs and not theirs & set(touches):
                continue
            minutes = max(1, round((time.time() - row["created"]) / 60))
            return (
                f"Another session ({row.get('session') or 'unknown'}) asked for code work on "
                f"{repo} {minutes} min ago: \u201c{row.get('question', '')[:120]}\u201d. "
                "Work in your own git worktree so both changes can merge."
            )
        return ""

    # -- the request ---------------------------------------------------------------------------
    async def ask(
        self,
        question: str,
        repo: str = "",
        mode: str = "",
        strictness: str = "",
        *,
        session: str = "",
        project: str = "",
        decisions: dict[str, str] | None = None,
        caller: dict[str, str] | None = None,
        autonomy: str = "",
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question is required")
        caller = caller or {}
        for component, choice in (decisions or {}).items():
            if component not in COMPONENTS or component in NO_FALLBACK:
                raise ValueError(f"decisions: unknown component {component!r}")
            if choice not in CHOICES:
                raise ValueError(f"decisions: choice must be one of {', '.join(CHOICES)}")
        if autonomy and autonomy not in AUTONOMY:
            raise ValueError(f"autonomy must be one of {', '.join(AUTONOMY)}")
        started = time.monotonic()
        rules = self.rules()
        route = classify(question, rules, mode, repo)
        spec = self.specialist(route.question, rules)
        if route.mode == "fetch":  # a fetch names no specialist
            spec = {**spec, "name": None, "specialists": [], "guidance": ""}
        sid = session.strip() or caller.get("session", "") or "unknown"
        turn = Turn(sid, {}, autonomy=autonomy or self.default_autonomy)
        rid = uuid.uuid4().hex[:12]
        touches = [t.strip("`'\"(),:;!?") for t in question.split() if _looks_like_code(t)]
        if self.hub is not None:
            for component, choice in (decisions or {}).items():
                self.hub.ledger.decide(sid, component, choice)
            turn.decisions = self.hub.ledger.decisions(sid)
            rid = self.hub.ledger.start(
                route.question,  # without a routing prefix, so duplicates compare like with like
                rid=rid,
                client=caller.get("client", ""),
                session=sid,
                session_inferred=not session.strip(),
                project=project.strip()
                or (repo.rsplit("/", maxsplit=1)[-1] if repo else "")
                or caller.get("project", ""),
                repo=repo,
                route=route.mode,
                specialist=spec["name"] or "",
                touches=touches,
            )
            self.hub.refresh_usage_soon()
        progress = [f"Route: {route.mode} ({route.reason})"]
        progress.append(_specialist_line(spec))
        results: dict[str, Any] = {}
        # The project the request comes from, for the early extension steps (project rules).
        where = (
            project.strip()
            or (repo.rsplit("/", maxsplit=1)[-1] if repo else "")
            or caller.get("project", "")
        )
        await self._extension_steps(
            turn,
            route,
            rules=rules,
            project=where,
            repo=repo,
            progress=progress,
            results=results,
            stage="early",
        )
        early = copy.deepcopy(results)
        turn.rid = rid
        work = self._work(
            turn,
            route,
            spec=spec,
            rules=rules,
            repo=repo,
            project=where,
            strictness=strictness,
            touches=touches,
            progress=progress,
            results=results,
            started=started,
        )
        if self.hub is None:
            return await work
        # Every reply comes within the deadline; work that takes longer carries on and leaves
        # its result in the ledger for hub__result.
        task = asyncio.ensure_future(work)
        self._tasks.add(task)
        task.add_done_callback(self._settled(rid))
        done, _ = await asyncio.wait({task}, timeout=self.hub.deadline)
        if task in done:
            return task.result()
        pending = self._pending(rid, route, spec=spec, progress=progress, started=started)
        if early:
            pending["results"] = early
        if turn.brief_notes:
            pending["brief_notes"] = dict(turn.brief_notes)
        return pending

    def _settled(self, rid: str) -> Callable[[asyncio.Task[dict[str, Any]]], None]:
        """Record a request that failed after its reply had already gone."""

        def done(task: asyncio.Task[dict[str, Any]]) -> None:
            self._tasks.discard(task)
            if task.cancelled() or self.hub is None:
                return
            exc = task.exception()
            if exc is not None:
                self.hub.ledger.finish(
                    rid,
                    "failed",
                    problems=[{"what_failed": f"{type(exc).__name__}: {exc}"[:500]}],
                )

        return done

    def _pending(
        self,
        rid: str,
        route: Route,
        *,
        spec: dict[str, Any],
        progress: list[str],
        started: float,
    ) -> dict[str, Any]:
        """The reply at the deadline: queued (with position and estimate) or still working."""
        assert self.hub is not None
        hub = self.hub
        seconds = round(time.monotonic() - started, 1)
        lane = hub.lanes.retrieval
        problem: Problem | None = None
        after = f"The request carries on: call `hub__result` with id `{rid}` for its answer."
        if (pos := lane.position(rid)) is not None:
            wait = lane.estimate(rid) or 0
            held = " Memory is low, so heavy work waits." if lane.held_by_memory else ""
            problem = Problem(
                "queued",
                "queue",
                f"Retrieval is busy ({lane.limit} at a time): position {pos} in line, about "
                f"{wait:.0f} s.{held}",
                recommendation=after,
                retry_after=max(5, round(wait)),
            )
        elif rid in hub.lanes.research:
            pos = list(hub.lanes.research).index(rid) + 1
            typical = hub.ledger.typical_seconds("research", 600.0)
            wait = typical * pos / max(1, hub.settings.max_runs)
            why = (
                "Memory is low, so research waits."
                if hub.lanes.short_of_memory()
                else f"The research pipeline is full ({hub.settings.max_runs} runs at once)."
            )
            problem = Problem(
                "queued",
                "queue",
                f"{why} Position {pos} in line, about {wait / 60:.0f} min.",
                recommendation=after,
                retry_after=max(30, round(wait)),
            )
        status = problem.status if problem else "working"
        footer = _footer(rid, hub.meters.line())
        got = [problem.short() if problem else f"Still working after {seconds} s."]
        nxt = ["The request keeps running on the gateway."]
        you = [f"Ask for the answer when you want it (request `{rid}`)."]
        text = replies.short(got, nxt, you)
        full = replies.join(
            problem.plain() if problem else f"**Still working after {seconds} s.** {after}",
            f"**What the research tool did so far** ({seconds} s)\n"
            + "\n".join(f"- {p}" for p in progress),
            "**Limits:** Nothing is concluded yet; the answer comes through `hub__result`.",
            footer and f"_{footer}_",
        )
        hub.ledger.finish(rid, status, problems=[problem.as_dict()] if problem else None)
        out: dict[str, Any] = {
            "request_id": rid,
            "status": status,
            "route": route.as_dict(),
            "specialist": spec,
            "progress": list(progress),
            "seconds": seconds,
            "relay": text,
            "relay_full": full,
            "summary": {"results": got, "next": nxt, "for_you": you},
            "relay_instructions": (
                "Show `relay` to the person exactly as written. Show `relay_full` only when the "
                "person asks for more detail. Call hub__result with request_id when the person "
                "wants the answer."
            ),
        }
        if problem is not None:
            out["error"] = problem.as_dict()
        return out

    async def _work(
        self,
        turn: Turn,
        route: Route,
        *,
        spec: dict[str, Any],
        rules: Rules,
        repo: str,
        project: str,
        strictness: str,
        touches: list[str],
        progress: list[str],
        results: dict[str, Any],
        started: float,
    ) -> dict[str, Any]:
        rid = turn.rid if self.hub is not None else ""
        try:
            await self._run(
                turn,
                route,
                spec=spec,
                rules=rules,
                repo=repo,
                project=project,
                strictness=strictness,
                touches=touches,
                progress=progress,
                results=results,
            )
        except Halt as halt:
            seconds = round(time.monotonic() - started, 1)
            problems = [halt.problem, *turn.problems]
            return self._reply(
                rid,
                route,
                spec=spec,
                progress=progress,
                results=results,
                seconds=seconds,
                problems=problems,
                ext_lines=turn.ext_lines,
                ext_limits=turn.ext_limits,
                brief_notes=turn.brief_notes,
                autonomy=turn.autonomy,
            )
        seconds = round(time.monotonic() - started, 1)
        return self._reply(
            rid,
            route,
            spec=spec,
            progress=progress,
            results=results,
            seconds=seconds,
            problems=turn.problems,
            ext_lines=turn.ext_lines,
            ext_limits=turn.ext_limits,
            brief_notes=turn.brief_notes,
            autonomy=turn.autonomy,
        )

    async def _run(
        self,
        turn: Turn,
        route: Route,
        *,
        spec: dict[str, Any],
        rules: Rules,
        repo: str,
        project: str,
        strictness: str,
        touches: list[str],
        progress: list[str],
        results: dict[str, Any],
    ) -> None:
        if route.mode == "fetch":  # no pipeline, no passages, no fresh search: just the index
            await self._fetch(route, rules=rules, progress=progress, results=results)
            return
        # A code task gets passages only when it has research content: the library holds papers,
        # not code, so "fix the failing test" would get unrelated passages and cost seconds.
        low = route.question.lower()
        if route.mode == "quick" and any(_has(low, t) for t in rules.profiles_list):
            results["profiles"] = self.profiles()
            progress.append(f"Profiles: listed {len(results['profiles'])}; no retrieval")
            return
        research_words = [p for p in rules.research if p in low]
        skip = await self._extension_steps(
            turn,
            route,
            rules=rules,
            project=project,
            repo=repo,
            progress=progress,
            results=results,
            stage="before_retrieval",
        )
        # A step may ask for retrieval to be skipped in quick mode (a sky request: papers don't
        # hold clock times).
        context = (route.mode == "quick" and not skip) or bool(spec["name"] or research_words)
        if route.mode == "quick" and skip and not context:
            progress.append(f"Retrieval: skipped ({skip[0]})")
        if route.mode == "code" and not context:
            progress.append("Retrieval: skipped (no research content in this code task)")
        if route.mode in ("quick", "code") and context:
            limit = QUICK_PASSAGES if route.mode == "quick" else CODE_PASSAGES
            t = time.monotonic()
            fresh: dict[str, Any] | None = None
            spent = 0.0

            async def evidence() -> tuple[str, dict[str, Any]]:
                nonlocal spent
                got = await self._evidence(turn, route.question, limit)
                spent = time.monotonic() - t
                return got

            if route.mode == "quick":
                (kind, found), fresh = await asyncio.gather(
                    evidence(), self._fresh(route.question, papers=True)
                )
            else:
                kind, found = await evidence()
            if kind == "evidence":
                found = apply_relevance_floor(found)
            results[kind] = found
            if kind == "evidence":
                progress.append(
                    f"Retrieved {found.get('n', 0)} passage(s) ({found.get('retrieval', '?')}) in "
                    f"{spent:.1f} s"
                )
                low = found.get("below_floor")
                if low:
                    progress.append(
                        f"Left out {low['n']} passage(s) under the relevance floor "
                        f"(reranker score < {low['floor']:g}; best {low['best_score']:.2f})"
                    )
            else:
                progress.append(
                    f"Searched the scholarly indexes instead: {len(found.get('results', []))} "
                    f"hit(s) in {spent:.1f} s"
                )
            if fresh is not None:
                if kind == "search":
                    fresh["papers"] = []  # the fallback above already listed the indexes' titles
                results["fresh"] = fresh
                progress.append(
                    f"Fresh search outside the library: {len(fresh['papers'])} paper(s) from the "
                    f"science engines, {len(fresh['web'])} web result(s) in "
                    f"{time.monotonic() - t:.1f} s"
                    + (f"; not run: {'; '.join(fresh['not_run'])}" if fresh["not_run"] else "")
                )
        if route.mode == "research":
            run = await self._research(turn, route.question, strictness)
            results["research"] = run
            if run.get("attached_to"):
                progress.append(
                    f"Attached to request {run['attached_to']} ({run['how']}): no second run"
                )
            else:
                progress.append(
                    f"Started research run {run.get('run_id', '?')} ({run.get('status', '?')})"
                )
            progress.extend(turn.notes)
        await self._extension_steps(
            turn,
            route,
            rules=rules,
            project=project,
            repo=repo,
            progress=progress,
            results=results,
            stage="after",
        )
        if route.mode == "code":
            note = (
                "The gateway does not make code changes itself yet. The assistant makes the "
                "change: in a local checkout when it has one, otherwise as a pull request "
                "through repos__propose_files. Read every file before changing it."
            )
            if self.hub is not None:
                github = await self.hub.github()
                if github["state"] != "up":
                    self._broken(turn, "github", f"GitHub is not usable: {github['detail']}.")
                    note = (
                        "The gateway does not make code changes itself yet. GitHub is not "
                        "usable, so the assistant edits a local checkout only; no pull request "
                        "until it is back. Read every file before changing it."
                    )
            results["coder"] = {
                "runs_on": "client",
                "tools": ["repos__read_file", "repos__search_code", "repos__propose_files"],
                "note": note,
            }
            overlap = self._overlap(turn, repo, touches)
            if overlap:
                results["coder"]["overlap"] = overlap
            progress.append("Coder: recruited (the assistant edits; the gateway gave context)")

    async def _extension_steps(
        self,
        turn: Turn,
        route: Route,
        *,
        rules: Rules,
        project: str,
        repo: str,
        progress: list[str],
        results: dict[str, Any],
        stage: str,
    ) -> list[str]:
        """The extension steps, on any route: each step whose trigger words the request names runs
        on a copy of the results so far. Its result goes under its own key and its lines join the
        reply; a step that fails is named, and nothing else in the reply changes
        (research_mcp/extensions.py). Only steps of the given stage run; the reasons the
        before_retrieval steps give for skipping retrieval are returned."""
        skip: list[str] = []
        for ext, step, ctx in self.steps:
            if step.stage != stage:
                continue
            words = rules.extensions.get(ext.name, {}).get(step.name, ())
            if not extensions.matches(step, route.question, words):
                continue
            step_input = StepInput(
                question=route.question,
                mode=route.mode,
                project=project,
                repo=repo,
                results=copy.deepcopy(results),
                context=ctx,
            )
            try:
                out = await step.run(step_input)
            except Exception as exc:  # noqa: BLE001 - one step's failure must not fail the ask
                failure = f"{type(exc).__name__}: {exc}"
                results.setdefault("extensions", {}).setdefault(ext.name, {})[step.name] = {
                    "error": failure
                }
                progress.append(f"{ext.prefix}/{step.name}: failed ({failure})")
                continue
            if out is None:
                continue
            results.setdefault("extensions", {}).setdefault(ext.name, {})[step.name] = out.result
            if ext.brief_key and out.brief is not None:
                results[ext.brief_key] = out.brief
                turn.brief_notes[ext.brief_key] = ext.brief_note
            if out.limits and not turn.ext_limits:
                turn.ext_limits = out.limits
            if out.skip_retrieval:
                skip.append(out.skip_retrieval)
            progress.extend(out.progress or [f"{ext.prefix}: ran"])
            turn.ext_lines.append((out.found, out.next, out.for_you, out.relay))
        return skip

    def _reply(
        self,
        rid: str,
        route: Route,
        *,
        spec: dict[str, Any],
        progress: list[str],
        results: dict[str, Any],
        seconds: float,
        problems: list[Problem],
        ext_lines: Sequence[tuple[list[str], list[str], list[str], str]] = (),
        ext_limits: str = "",
        brief_notes: dict[str, str] | None = None,
        autonomy: str = "",
    ) -> dict[str, Any]:
        worst = next((p for p in problems if p.status != "degraded"), None)
        status = worst.status if worst else ("degraded" if problems else "done")
        if status == "done" and "research" in results:
            status = "working"
        usage = self.hub.meters.line() if self.hub is not None else ""
        got, nxt, you = _short(
            route, spec, results, problems, rid, ext_lines=ext_lines, ext_limits=ext_limits,
            autonomy=autonomy,
        )  # fmt: skip
        text, full = (
            relay(
                route, spec, progress, results, seconds, problems=problems, usage=usage,
                request_id=rid, full=f, ext_lines=ext_lines, ext_limits=ext_limits,
                autonomy=autonomy,
            )
            for f in (False, True)
        )  # fmt: skip
        out: dict[str, Any] = {
            "request_id": rid or None,
            "autonomy": autonomy,
            "status": status,
            "route": route.as_dict(),
            "specialist": spec,
            "progress": progress,
            "seconds": seconds,
            "results": results,
            "relay": text,
            "relay_full": full,
            "summary": {"results": got, "next": nxt, "for_you": you},
            "relay_instructions": (
                "Show `relay` to the person exactly as written. It is short on purpose: show "
                "`relay_full` (passages, steps, limits) only when the person asks for more. The "
                "passages are in `results` either way, for grounding your answer. Add anything of "
                "your own only after it, under a heading that says it is yours. Do not add facts "
                "from memory."
            ),
        }
        if brief_notes:
            out["brief_notes"] = dict(brief_notes)
        if worst is not None:
            out["error"] = worst.as_dict()
        if problems:
            out["problems"] = [p.as_dict() for p in problems]
        if self.hub is not None and rid:
            research = results.get("research", {})
            run_id = research.get("run_id")
            self.hub.ledger.finish(
                rid,
                status,
                seconds=seconds,
                problems=[p.as_dict() for p in problems] or None,
                result={"relay": text, "relay_full": full, "status": status},
                run_id=str(run_id) if run_id else None,
                attached_to=research.get("attached_to"),
            )
        return out


def _passages(found: dict[str, Any]) -> list[str]:
    lines = []
    for i, r in enumerate(found.get("results", []), 1):
        text = " ".join(str(r.get("text", "")).split())
        excerpt = text[:EXCERPT] + ("…" if len(text) > EXCERPT else "")
        source = ", ".join(
            str(x) for x in (r.get("title"), r.get("year"), r.get("doi") and f"doi:{r['doi']}") if x
        )
        lines.append(f"{i}. “{excerpt}” ({source}) `{r.get('evidence_id')}`")
    return lines


def _hits(found: dict[str, Any]) -> list[str]:
    lines = []
    for i, r in enumerate(found.get("results", []), 1):
        source = ", ".join(str(x) for x in (r.get("year"), r.get("doi") and f"doi:{r['doi']}") if x)
        lines.append(f"{i}. {r.get('title') or '(untitled)'}" + (f" ({source})" if source else ""))
    return lines


def relay(
    route: Route,
    spec: dict[str, Any],
    progress: list[str],
    results: dict[str, Any],
    seconds: float,
    *,
    problems: list[Problem] | tuple[Problem, ...] = (),
    usage: str = "",
    request_id: str = "",
    full: bool = False,
    ext_lines: Sequence[tuple[list[str], list[str], list[str], str]] = (),
    ext_limits: str = "",
    autonomy: str = "",
) -> str:
    """The block the chat model shows unchanged. Short (the default): three lists from
    `_short`. Full: problems first, then what ran, what came back, and its limits; last, the
    request id and usage."""
    if not full:
        return replies.short(
            *_short(
                route, spec, results, problems, request_id,
                ext_lines=ext_lines, ext_limits=ext_limits, autonomy=autonomy,
            )
        )  # fmt: skip
    blocks: dict[str, str] = {
        "passages": "",
        "passages_none": "",
        "excluded": "",
        "search": "",
        "research": "",
        "coder": "",
    }
    found = results.get("evidence")
    if found is not None:
        if found.get("results"):
            blocks["passages"] = "\n".join(_passages(found))
        else:
            blocks["passages_none"] = "none matched."
        if found.get("excluded"):
            n = len(found["excluded"])
            blocks["excluded"] = f"({n} passage(s) excluded by the safety or retraction check.)"
    hits = results.get("search")
    if hits is not None:
        blocks["search"] = "\n".join(_hits(hits) or ["None matched."])
    fresh_block = _fresh_block(results.get("fresh"))
    run = results.get("research")
    out: list[str] = []
    if run is not None and run.get("attached_to"):
        out += [
            f"**Attached** to request `{run['attached_to']}`"
            + (f" (run `{run['run_id']}`)" if run.get("run_id") else "")
            + f", asked by session {run.get('session') or 'unknown'}: "
            f"\u201c{str(run.get('question', ''))[:160]}\u201d "
            f"({run['how']}, {run['similarity']}). "
            "No second run was started; its answer comes through `hub__result` with this "
            "request's id.",
        ]
    elif run is not None:
        out += [
            f"**Research run** `{run.get('run_id')}`: {run.get('status')}. It runs on the gateway "
            f"(model: {run.get('model_backend', 'configured provider')}); continue with "
            "`pipeline__research_continue` until it is done, then show its answer with its Limits.",
        ]
    blocks["research"] = "".join(out)
    if results.get("coder"):
        coder = [f"**Coder:** {results['coder']['note']}"]
        if results["coder"].get("overlap"):
            coder += ["", f"**Overlap:** {results['coder']['overlap']}"]
        blocks["coder"] = "\n".join(coder)
    limits = {
        "quick": "These are retrieved passages, not a checked answer: nothing here judged whether "
        "they answer the question. Start the request with `research:` for a checked answer.",
        "research": "The answer arrives when the run is done; until then nothing is concluded.",
        "code": "Any passages are context only; the change itself is not made or tested yet.",
        "fetch": "The papers are indexed, not read: nothing is answered from them yet.",
    }[route.mode]
    if any(p.status == "needs_decision" for p in problems):
        limits = "Nothing was concluded: the request stopped to ask how to handle a failure."
    elif results.get("profiles") is not None:
        limits = PROFILES_LIMIT
    guidance = str(spec.get("guidance") or "") if route.mode != "research" else ""
    footer = _footer(request_id, usage)
    passages = ""
    if blocks["passages"]:
        passages = replies.join(f"**Passages**\n{blocks['passages']}", blocks["excluded"])
    elif blocks["passages_none"]:
        passages = replies.join(f"**Passages:** {blocks['passages_none']}", blocks["excluded"])
    return replies.join(
        "\n\n".join(p.plain() for p in problems),
        f"**What the research tool did** ({seconds} s)\n" + "\n".join(f"- {p}" for p in progress),
        *(
            f"**{one['name']} guidance:** {one['guidance']}"
            for one in (spec.get("specialists") or [])
            if guidance and one.get("guidance")
        ),
        _profiles_block(results.get("profiles")),
        passages,
        blocks["search"]
        and "**Scholarly search results** (titles only: no verbatim passages, not citable)\n"
        + blocks["search"],
        fresh_block,
        *(row[3] for row in ext_lines if row[3]),
        blocks["research"],
        blocks["coder"],
        f"**Limits:** {limits}",
        footer and f"_{footer}_",
    )


def _fresh_block(fresh: dict[str, Any] | None) -> str:
    """The fresh scholarly and web search, as leads: titles and links, nothing read."""
    if fresh is None:
        return ""
    lines = ["**Fresh search outside the library** (unread leads: not citable)"]
    for i, r in enumerate(fresh.get("papers", []), 1):
        lines.append(f"{i}. {r.get('title') or '(untitled)'}: {r.get('url', '')}".rstrip(": "))
    for r in fresh.get("web", []):
        lines.append(f"- {r.get('title') or r['url']}: {r['url']}")
    if len(lines) == 1:
        lines.append("Nothing came back.")
    if fresh.get("not_run"):
        lines.append("Not searched: " + "; ".join(fresh["not_run"]) + ".")
    return "\n".join(lines)


def _profiles_block(rows: list[dict[str, str]] | None) -> str:
    if not rows:
        return ""
    return (
        "**Specialist profiles** (name one in a question to recruit it, up to three)\n"
        + "\n".join(f"- `{r['name']}` ({r['source']}): {r['for']}" for r in rows)
    )


def apply_relevance_floor(found: dict[str, Any]) -> dict[str, Any]:
    """`found` without the passages the reranker scored under RELEVANCE_FLOOR, noting how many."""
    if not found.get("reranker"):
        return found
    rows = found.get("results", [])

    def scored_low(row: dict[str, Any]) -> bool:
        score = row.get("score")
        return isinstance(score, (int, float)) and score < RELEVANCE_FLOOR

    dropped = [r for r in rows if scored_low(r)]
    if not dropped:
        return found
    kept = [r for r in rows if not scored_low(r)]
    best = max(float(r["score"]) for r in dropped)
    return {
        **found,
        "results": kept,
        "n": len(kept),
        "below_floor": {"n": len(dropped), "floor": RELEVANCE_FLOOR, "best_score": best},
    }


def _short(
    route: Route,
    spec: dict[str, Any],
    results: dict[str, Any],
    problems: list[Problem] | tuple[Problem, ...],
    request_id: str,
    *,
    ext_lines: Sequence[tuple[list[str], list[str], list[str], str]] = (),
    ext_limits: str = "",
    autonomy: str = "",
) -> tuple[list[str], list[str], list[str]]:
    """The short reply's three lists, one sentence per item: what came back, what the assistant
    does next, and what the person needs to do (often nothing)."""
    got: list[str] = [p.short() for p in problems]
    nxt: list[str] = []
    you: list[str] = [p.ask_person() for p in problems]
    stopped = any(p.status == "needs_decision" for p in problems)
    if results.get("profiles") is not None:
        rows = results["profiles"]
        got.append(
            f"{len(rows)} specialist profiles: " + ", ".join(r["name"] for r in rows) + "."
            if rows
            else "No specialist profiles are installed."
        )
        you.append("Name a profile in your question to recruit it (up to three answer at once).")
    for name in spec.get("unknown") or []:
        got.append(f"No profile is named {name}; ask for the list of profiles.")
    found = results.get("evidence")
    if found is not None:
        rows = found.get("results", [])
        if rows:
            papers = {r.get("doi") or r.get("title") for r in rows}
            top = rows[0]
            where = ", ".join(str(x) for x in (top.get("title"), top.get("year")) if x)
            got.append(
                f"Found {len(rows)} passage(s) from {len(papers)} paper(s)"
                + (f"; the closest is from {where}." if where else ".")
            )
            nxt.append("The assistant answers from these passages and cites their evidence ids.")
        elif found.get("below_floor"):
            low = found["below_floor"]
            got.append(
                f"No passage was relevant enough to show: the {low['n']} closest scored at most "
                f"{low['best_score']:.2f}, under the relevance floor of {low['floor']:g}."
            )
            nxt.append("The assistant says no passage applies before answering any other way.")
        else:
            got.append("No passage in the library matched.")
            nxt.append("The assistant says no passage applies before answering any other way.")
        if rows and found.get("below_floor"):
            got.append(
                f"{found['below_floor']['n']} weaker passage(s) under the relevance floor were "
                "left out."
            )
        if found.get("excluded"):
            got.append(
                f"{len(found['excluded'])} passage(s) were left out by the safety or "
                "retraction check."
            )
    hits = results.get("search")
    if hits is not None:
        got.append(
            f"Found {len(hits.get('results', []))} paper title(s) in the scholarly indexes, "
            "which can't be cited."
        )
        nxt.append("The assistant says no passage applies before answering any other way.")
    fetched = results.get("fetch")
    if fetched is not None:
        papers = fetched.get("papers", [])
        indexed = [p for p in papers if p["index"].get("status") in INDEXED]
        missed = [p for p in papers if p["index"].get("status") not in INDEXED]
        if indexed:
            titles = "; ".join(str(p.get("title") or "(untitled)") for p in indexed)
            got.append(f"Indexed {len(indexed)} paper(s) for retrieval: {titles}.")
            nxt.append("The assistant asks its question again to answer from the indexed passages.")
        if missed:
            names = "; ".join(
                f"{p.get('title') or p.get('doi') or '(no id)'}: "
                f"{p['index'].get('status') or 'no result'}"
                for p in missed
            )
            got.append(f"Not indexed: {names}.")
        if not papers:
            got.append("No paper was named or matched, so nothing was indexed.")
    fresh = results.get("fresh")
    if fresh is not None:
        n_papers, n_web = len(fresh.get("papers", [])), len(fresh.get("web", []))
        if n_papers or n_web:
            got.append(
                f"A fresh search outside the library found {n_papers} paper(s) from the science "
                f"engines and {n_web} web result(s): unread leads, not evidence."
            )
            level = (brief.coverage(found) or {}).get("level") if found is not None else "none"
            if level in ("partial", "weak", "none"):
                nxt.append(
                    "The assistant fetches and indexes the closest fresh leads, then retrieves "
                    "again."
                    if autonomy == "delegate"
                    else "The assistant points to the closest fresh leads, with their links, and "
                    "suggests `research:` to fetch and read them."
                )
        else:
            got.append("A fresh search outside the library found nothing either.")
        if fresh.get("not_run"):
            got.append("Not searched: " + "; ".join(fresh["not_run"]) + ".")
    run = results.get("research")
    if run is not None and run.get("attached_to"):
        got.append(
            f"Joined request `{run['attached_to']}`, already running on the same question, "
            "instead of starting a second run."
        )
    elif run is not None:
        got.append(f"Started research run `{run.get('run_id')}` ({run.get('status')}).")
    if run is not None:
        nxt.append(
            "The run carries on; the assistant fetches its answer with `hub__result`"
            + (f" (`{request_id}`)." if request_id else ".")
        )
    coder = results.get("coder")
    if coder:
        got.append("Gathered context for the change; the gateway doesn't edit code itself.")
        if coder.get("overlap"):
            got.append(first_sentence(coder["overlap"]))
        nxt.append(
            "The assistant reads the files, then makes the change locally or as a pull request."
        )
    if spec.get("name") and route.mode != "research":
        names = [one["name"] for one in spec.get("specialists") or []] or [spec["name"]]
        got.append(f"Specialist{'s' if len(names) > 1 else ''}: {', '.join(names)}.")
    for found_lines, next_lines, for_you_lines, _relay in ext_lines:
        got.extend(found_lines)
        nxt.extend(next_lines)
        you.extend(for_you_lines)
    got.append(
        "Nothing was concluded."
        if stopped
        else ext_limits
        if ext_limits and results.get("evidence") is None and route.mode == "quick"
        else "Nothing was retrieved."
        if results.get("profiles") is not None
        else {
            "quick": "These are passages, not a checked answer.",
            "research": "Nothing is concluded until the run finishes.",
            "code": "The change isn't made or tested yet.",
            "fetch": "The papers are indexed, not read: nothing is answered from them yet.",
        }[route.mode]
    )
    if stopped:
        nxt = ["Waiting for your choice."]
    return got, nxt, you


PROFILES_LIMIT = "This is the list of installed profiles, not an answer: nothing was retrieved."


def _specialist_line(spec: dict[str, Any]) -> str:
    names = [s["name"] for s in spec.get("specialists") or []]
    if len(names) > 1:
        return f"Specialists: {', '.join(names)} ({spec.get('reason') or spec['why']})"
    return f"Specialist: {spec['name']} ({spec['why']})" if spec["name"] else "Specialist: none"


def _footer(request_id: str, usage: str) -> str:
    """The request id and usage, as one line."""
    return " · ".join(x for x in (request_id and f"Request `{request_id}`", usage) if x)


def parse_tool_text(result: Any) -> dict[str, Any]:
    """The JSON object in an upstream tool's first text block (the pipeline returns JSON text)."""
    content = getattr(result, "content", None) or []
    if getattr(result, "isError", False):
        raise ValueError(content[0].text if content else "the upstream tool failed")
    text = content[0].text if content else "{}"
    data = json.loads(text)
    return json.loads(data["result"]) if isinstance(data.get("result"), str) else data
