"""One error contract for every non-success reply (docs/hub-plan.md, section 5).

Each problem says what broke, what was skipped or used instead, what that means for the work, and
what to do next, in plain words first (so it reaches the person whichever chat model relays it)
and as JSON after (so a client can act on it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mcp.types import CallToolResult, TextContent

# `needs_decision`: the first failure of a component in a session asks how to handle it; the
# answer then holds for the rest of that session (decided 2026-10-28).
STATUSES = ("queued", "degraded", "refused", "failed", "needs_decision")
CHOICES = ("fallback", "stop")
RESTART_HINT = "launchctl kickstart -k gui/$(id -u)/com.polarizetech.research-rag"


@dataclass(frozen=True)
class Component:
    name: str
    fallback: str
    impact: str


# The plan's table (section 4). An upstream has no automatic fallback until phase 3.
COMPONENTS: dict[str, Component] = {
    "reranker": Component(
        "the reranker", "retrieval without reranking", "faster, less precise passages"
    ),
    "embeddings": Component("embeddings (Ollama)", "lexical retrieval", "misses paraphrases"),
    "library": Component(
        "the paper library",
        "scholarly search results only",
        "no verbatim passages; claims are not citable",
    ),
    "provider": Component(
        "the model provider",
        "the next provider on the step's ladder",
        "a weaker model may write; each step names its provider in the run log",
    ),
    "verifier": Component(
        "the verifier",
        "explore strictness; claims marked unverified",
        "no publication-level answers until it is back",
    ),
    "upstream": Component(
        "an upstream process", "none yet (restart with backoff is phase 3)", "a pause"
    ),
    "github": Component(
        "GitHub access",
        "the code route edits the local checkout only",
        "no pull request until it is back",
    ),
    "corpus": Component(
        "the research corpus",
        "work continues; evidence propagation is left pending",
        "cross-study updates come later",
    ),
    "gateway": Component(
        "the gateway",
        "none: the bridge reports it",
        "files can still be read; no grounded claims",
    ),
    "request": Component("the request", "none", "nothing ran"),
    "queue": Component(
        "a busy resource", "none: it waits its turn", "the answer comes later, through hub__result"
    ),
}
NO_FALLBACK = ("upstream", "gateway", "request", "queue")


@dataclass
class Problem:
    status: str
    component: str
    what_failed: str
    impact: str = ""
    fallback_used: str | None = None
    options: list[dict[str, str]] = field(default_factory=list)
    recommendation: str = ""
    retry_after: int | None = None
    decided: str = ""  # how the fallback was chosen: "your choice for this session", "detected"

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        if not self.impact:
            self.impact = COMPONENTS.get(self.component, COMPONENTS["request"]).impact

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "component": self.component,
            "what_failed": self.what_failed,
            "impact": self.impact,
            "fallback_used": self.fallback_used,
            "options": self.options,
            "recommendation": self.recommendation,
            "retry_after": self.retry_after,
        }

    def plain(self) -> str:
        """Markdown lines in plain words, for the top of the relay block."""
        name = COMPONENTS.get(self.component, Component(self.component, "", "")).name
        retry = f" Retry in about {self.retry_after} s." if self.retry_after else ""
        if self.status == "needs_decision":
            opts = "; ".join(f"`{o['choice']}`: {o['effect']}" for o in self.options)
            return (
                f"**Your choice is needed: {name} failed.** {self.what_failed}\n"
                f"- Options: {opts}.\n"
                f"- Recommended: {self.recommendation}\n"
                f"- Tell the assistant which. It passes it to `ask` as "
                f'`decisions={{"{self.component}": "<choice>"}}`, and the gateway applies that '
                "choice for the rest of this session."
            )
        if self.status == "degraded":
            how = f" ({self.decided})" if self.decided else ""
            return (
                f"**Degraded: {name}.** {self.what_failed} Used instead: {self.fallback_used}"
                f"{how}. What it means: {self.impact}."
            )
        label = {"failed": "Failed", "refused": "Refused", "queued": "Queued"}[self.status]
        nxt = f" Next: {self.recommendation}" if self.recommendation else ""
        return f"**{label}: {name}.** {self.what_failed} What it means: {self.impact}.{nxt}{retry}"


def decision_options(component: str, retry_after: int | None) -> list[dict[str, str]]:
    spec = COMPONENTS[component]
    wait = f" and retry in about {retry_after} s" if retry_after else " and retry later"
    return [
        {"choice": "fallback", "effect": f"continue with {spec.fallback} ({spec.impact})"},
        {"choice": "stop", "effect": f"stop this request{wait}"},
    ]


def needs_decision(component: str, what_failed: str, retry_after: int | None) -> Problem:
    return Problem(
        "needs_decision",
        component,
        what_failed,
        options=decision_options(component, retry_after),
        recommendation="`fallback`: the work keeps moving, and the reply labels what it lost.",
        retry_after=retry_after,
    )


def error_result(problem: Problem, detail: str = "") -> CallToolResult:
    """A tool error carrying the contract: plain words first, then the JSON."""
    text = (detail + "\n\n" if detail else "") + problem.plain()
    text += "\n\n```json\n" + json.dumps({"error": problem.as_dict()}, indent=2) + "\n```"
    return CallToolResult(isError=True, content=[TextContent(type="text", text=text)])
