"""`ask`: one entry point that routes a request, runs only what it needs, and returns a relay.

The chat model's job is to call `ask` and show `relay` to the person unchanged. The gateway does
the work: a passage retrieval for a question, the research pipeline for an evidence question, and
retrieval plus the coder's tools for a task or code review (routing.py decides which). The relay
says what ran and why, what came back, and its limits, so the chat model has nothing to fill in
from memory, and any chat model can carry the work on.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_mcp.hub.contract import (
    CHOICES,
    COMPONENTS,
    NO_FALLBACK,
    RESTART_HINT,
    Problem,
    first_sentence,
    needs_decision,
)
from research_mcp.hub.health import Hub
from research_mcp.routing import (
    Route,
    Rules,
    _looks_like_code,
    classify,
    load_rules,
    profile_override,
)
from research_pipeline import agent_profiles, replies
from research_pipeline.papers import PapersError

QUICK_PASSAGES = 5
CODE_PASSAGES = 3
EXCERPT = 320
# Under a client's 60 s MCP timeout, with room for the fallback after it.
RETRIEVE_SECONDS = 40.0
# Code work on the same repository from another session within this window is mentioned.
OVERLAP_SECONDS = 2 * 3600
RETRIEVAL_ERRORS = (PapersError, ValueError, KeyError, OSError, RuntimeError)

Retrieve = Callable[[str, int, int], Awaitable[dict[str, Any]]]
# Retrieval with one part of the library's ranking off ("reranker" or "embeddings").
PlainRetrieve = Callable[[str, int, int, str], Awaitable[dict[str, Any]]]
Search = Callable[[str, int], Awaitable[dict[str, Any]]]
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
        plain_retrieve: PlainRetrieve | None = None,
        can_switch: Callable[[str], bool] = lambda _component: True,
    ):
        self.retrieve = retrieve
        self.call_tool = call_tool
        self.has_tool = has_tool
        self.rules_paths = rules_paths
        self.profiles_dir = profiles_dir
        self.extensions_dir = extensions_dir
        self.hub = hub
        self.search = search
        self.plain_retrieve = plain_retrieve
        self.can_switch = can_switch
        # Requests still working after their reply's deadline: held so they finish.
        self._tasks: set[asyncio.Task[dict[str, Any]]] = set()

    def rules(self) -> Rules:
        # Read on every call, so an edit to config/routing.toml applies without a restart.
        return load_rules(*self.rules_paths)

    def specialist(self, question: str, rules: Rules) -> dict[str, Any]:
        own = profile_override(question, rules)
        route = agent_profiles.route(question, self.profiles_dir, "", self.extensions_dir)
        name = own[0] if own else route.get("specialist")
        why = f"your word '{own[1]}'" if own else route.get("reason", "")
        profile = next((p for p in route.get("profiles", []) if p.get("name") == name), {})
        guidance = route.get("guidance", "") if name == route.get("specialist") else ""
        if own and not profile:
            briefs = agent_profiles.load(self.extensions_dir) if self.extensions_dir else {}
            brief = briefs.get(own[0]) or agent_profiles.load(self.profiles_dir).get(own[0])
            if brief is not None:
                profile = {"name": name, "path": str(brief.path), "repository": str(brief.root)}
                guidance = brief.guidance
        return {
            "name": name or None,
            "why": why,
            "roles": route.get("roles", ["researcher"]),
            "brief": profile.get("path", ""),
            "repository": profile.get("repository", ""),
            "guidance": guidance,
        }

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
                    recommendation=f"restart the gateway: `{RESTART_HINT}`",
                    retry_after=retry,
                )
            )
        choice = turn.decisions.get(component)
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
                    f"`{RESTART_HINT}`",
                    retry_after=self._retry("library"),
                )
            ) from exc

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
        started = time.monotonic()
        rules = self.rules()
        route = classify(question, rules, mode, repo)
        spec = self.specialist(route.question, rules)
        sid = session.strip() or caller.get("session", "") or "unknown"
        turn = Turn(sid, {})
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
        progress.append(
            f"Specialist: {spec['name']} ({spec['why']})" if spec["name"] else "Specialist: none"
        )
        results: dict[str, Any] = {}
        turn.rid = rid
        work = self._work(
            turn,
            route,
            spec=spec,
            rules=rules,
            repo=repo,
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
        return self._pending(rid, route, spec=spec, progress=progress, started=started)

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
        text = replies.short(
            [problem.short() if problem else f"Still working after {seconds} s."],
            ["The request keeps running on the gateway."],
            [f"Ask for the answer when you want it (request `{rid}`)."],
        )
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
        )

    async def _run(
        self,
        turn: Turn,
        route: Route,
        *,
        spec: dict[str, Any],
        rules: Rules,
        repo: str,
        strictness: str,
        touches: list[str],
        progress: list[str],
        results: dict[str, Any],
    ) -> None:
        # A code task gets passages only when it has research content: the library holds papers,
        # not code, so "fix the failing test" would get unrelated passages and cost seconds.
        research_words = [p for p in rules.research if p in route.question.lower()]
        context = route.mode == "quick" or bool(spec["name"] or research_words)
        if route.mode == "code" and not context:
            progress.append("Retrieval: skipped (no research content in this code task)")
        if route.mode in ("quick", "code") and context:
            limit = QUICK_PASSAGES if route.mode == "quick" else CODE_PASSAGES
            t = time.monotonic()
            kind, found = await self._evidence(turn, route.question, limit)
            results[kind] = found
            if kind == "evidence":
                progress.append(
                    f"Retrieved {found.get('n', 0)} passage(s) ({found.get('retrieval', '?')}) in "
                    f"{time.monotonic() - t:.1f} s"
                )
            else:
                progress.append(
                    f"Searched the scholarly indexes instead: {len(found.get('results', []))} "
                    f"hit(s) in {time.monotonic() - t:.1f} s"
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
    ) -> dict[str, Any]:
        worst = next((p for p in problems if p.status != "degraded"), None)
        status = worst.status if worst else ("degraded" if problems else "done")
        if status == "done" and "research" in results:
            status = "working"
        usage = self.hub.meters.line() if self.hub is not None else ""
        text, full = (
            relay(
                route, spec, progress, results, seconds, problems=problems, usage=usage,
                request_id=rid, full=f,
            )
            for f in (False, True)
        )  # fmt: skip
        out: dict[str, Any] = {
            "request_id": rid or None,
            "status": status,
            "route": route.as_dict(),
            "specialist": spec,
            "progress": progress,
            "seconds": seconds,
            "results": results,
            "relay": text,
            "relay_full": full,
            "relay_instructions": (
                "Show `relay` to the person exactly as written. It is short on purpose: show "
                "`relay_full` (passages, steps, limits) only when the person asks for more. The "
                "passages are in `results` either way, for grounding your answer. Add anything of "
                "your own only after it, under a heading that says it is yours. Do not add facts "
                "from memory."
            ),
        }
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
) -> str:
    """The block the chat model shows unchanged. Short (the default): three lists from
    `_short`. Full: problems first, then what ran, what came back, and its limits; last, the
    request id and usage."""
    if not full:
        return replies.short(*_short(route, spec, results, problems, request_id))
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
    }[route.mode]
    if any(p.status == "needs_decision" for p in problems):
        limits = "Nothing was concluded: the request stopped to ask how to handle a failure."
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
        guidance and f"**{spec.get('name') or ''} guidance:** {guidance}",
        passages,
        blocks["search"]
        and "**Scholarly search results** (titles only: no verbatim passages, not citable)\n"
        + blocks["search"],
        blocks["research"],
        blocks["coder"],
        f"**Limits:** {limits}",
        footer and f"_{footer}_",
    )


def _short(
    route: Route,
    spec: dict[str, Any],
    results: dict[str, Any],
    problems: list[Problem] | tuple[Problem, ...],
    request_id: str,
) -> tuple[list[str], list[str], list[str]]:
    """The short reply's three lists, one sentence per item: what came back, what the assistant
    does next, and what the person needs to do (often nothing)."""
    got: list[str] = [p.short() for p in problems]
    nxt: list[str] = []
    you: list[str] = [p.ask_person() for p in problems]
    stopped = any(p.status == "needs_decision" for p in problems)
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
        else:
            got.append("No passage in the library matched.")
            nxt.append("The assistant says no passage applies before answering any other way.")
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
        got.append(f"Specialist: {spec['name']}.")
    got.append(
        "Nothing was concluded."
        if stopped
        else {
            "quick": "These are passages, not a checked answer.",
            "research": "Nothing is concluded until the run finishes.",
            "code": "The change isn't made or tested yet.",
        }[route.mode]
    )
    if stopped:
        nxt = ["Waiting for your choice."]
    return got, nxt, you


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
