"""`ask`: one entry point that routes a request, runs only what it needs, and returns a relay.

The chat model's job is to call `ask` and show `relay` to the person unchanged. The gateway does
the work: a passage retrieval for a question, the research pipeline for an evidence question, and
retrieval plus the coder's tools for a task or code review (routing.py decides which). The relay
says what ran and why, what came back, and its limits, so the chat model has nothing to fill in
from memory, and any chat model can carry the work on.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from research_mcp.routing import Route, Rules, classify, load_rules, profile_override
from research_pipeline import agent_profiles

QUICK_PASSAGES = 5
CODE_PASSAGES = 3
EXCERPT = 320

Retrieve = Callable[[str, int, int], Awaitable[dict[str, Any]]]
CallTool = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


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
    ):
        self.retrieve = retrieve
        self.call_tool = call_tool
        self.has_tool = has_tool
        self.rules_paths = rules_paths
        self.profiles_dir = profiles_dir
        self.extensions_dir = extensions_dir

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
            brief = briefs.get(name) or agent_profiles.load(self.profiles_dir).get(name)
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

    async def ask(
        self, question: str, repo: str = "", mode: str = "", strictness: str = ""
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question is required")
        started = time.monotonic()
        rules = self.rules()
        route = classify(question, rules, mode, repo)
        spec = self.specialist(route.question, rules)
        progress = [f"Route: {route.mode} ({route.reason})"]
        progress.append(
            f"Specialist: {spec['name']} ({spec['why']})" if spec["name"] else "Specialist: none"
        )
        results: dict[str, Any] = {}
        # A code task gets passages only when it has research content: the library holds papers,
        # not code, so "fix the failing test" would get unrelated passages and cost seconds.
        research_words = [p for p in rules.research if p in route.question.lower()]
        context = route.mode == "quick" or bool(spec["name"] or research_words)
        if route.mode == "code" and not context:
            progress.append("Retrieval: skipped (no research content in this code task)")
        if route.mode in ("quick", "code") and context:
            limit = QUICK_PASSAGES if route.mode == "quick" else CODE_PASSAGES
            t = time.monotonic()
            found = await self.retrieve(route.question, limit, 1800)
            results["evidence"] = found
            progress.append(
                f"Retrieved {found.get('n', 0)} passage(s) ({found.get('retrieval', '?')}) in "
                f"{time.monotonic() - t:.1f} s"
            )
        if route.mode == "research":
            if not self.has_tool("pipeline__research_start"):
                raise ValueError("the research pipeline upstream is unavailable")
            args: dict[str, Any] = {"question": route.question}
            if strictness:
                args["strictness"] = strictness
            run = await self.call_tool("pipeline__research_start", args)
            if run.get("error"):
                raise ValueError(f"the research run did not start: {run['error']}")
            results["research"] = run
            progress.append(
                f"Started research run {run.get('run_id', '?')} ({run.get('status', '?')})"
            )
        if route.mode == "code":
            results["coder"] = {
                "runs_on": "client",
                "tools": ["repos__read_file", "repos__search_code", "repos__propose_files"],
                "note": (
                    "The gateway does not make code changes itself yet. The assistant makes the "
                    "change: in a local checkout when it has one, otherwise as a pull request "
                    "through repos__propose_files. Read every file before changing it."
                ),
            }
            progress.append("Coder: recruited (the assistant edits; the gateway gave context)")
        seconds = round(time.monotonic() - started, 1)
        return {
            "route": route.as_dict(),
            "specialist": spec,
            "progress": progress,
            "seconds": seconds,
            "results": results,
            "relay": relay(route, spec, progress, results, seconds),
            "relay_instructions": (
                "Show `relay` to the person exactly as written. Add anything of your own only "
                "after it, under a heading that says it is yours. Do not add facts from memory."
            ),
        }


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


def relay(
    route: Route,
    spec: dict[str, Any],
    progress: list[str],
    results: dict[str, Any],
    seconds: float,
) -> str:
    """The block the chat model shows unchanged: what ran, why, what came back, and its limits."""
    out = [f"**What the research tool did** ({seconds} s)", *[f"- {p}" for p in progress]]
    if spec.get("guidance") and route.mode != "research":
        out += ["", f"**{spec['name']} guidance:** {spec['guidance']}"]
    found = results.get("evidence")
    if found is not None:
        out += ["", "**Passages**" if found.get("results") else "**Passages:** none matched."]
        out += _passages(found)
        if found.get("excluded"):
            n = len(found["excluded"])
            out.append(f"({n} passage(s) excluded by the safety or retraction check.)")
    run = results.get("research")
    if run is not None:
        out += [
            "",
            f"**Research run** `{run.get('run_id')}`: {run.get('status')}. It runs on the gateway "
            f"(model: {run.get('model_backend', 'configured provider')}); continue with "
            "`pipeline__research_continue` until it is done, then show its answer with its Limits.",
        ]
    if results.get("coder"):
        out += ["", f"**Coder:** {results['coder']['note']}"]
    limits = {
        "quick": "These are retrieved passages, not a checked answer: nothing here judged whether "
        "they answer the question. Start the request with `research:` for a checked answer.",
        "research": "The answer arrives when the run is done; until then nothing is concluded.",
        "code": "Any passages are context only; the change itself is not made or tested yet.",
    }[route.mode]
    out += ["", f"**Limits:** {limits}"]
    return "\n".join(out)


def parse_tool_text(result: Any) -> dict[str, Any]:
    """The JSON object in an upstream tool's first text block (the pipeline returns JSON text)."""
    content = getattr(result, "content", None) or []
    if getattr(result, "isError", False):
        raise ValueError(content[0].text if content else "the upstream tool failed")
    text = content[0].text if content else "{}"
    data = json.loads(text)
    return json.loads(data["result"]) if isinstance(data.get("result"), str) else data
