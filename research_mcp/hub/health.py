"""Per-component health and the hub object `ask` and the hub__ tools share (docs/hub-plan.md, 4).

Each component has a check and a circuit breaker. Checks are cheap and cached: Ollama's model list
for embeddings and the verifier, the upstream connections, GitHub's auth status, the corpus
checkout, and the provider meters. Breakers learn from real calls: `ask` records each failure and
success of retrieval, and a component that keeps failing is skipped for a cool-down.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from research_mcp.hub.ledger import Ledger
from research_pipeline.config import Settings
from research_pipeline.usage import Breakers, Fetch, Meters, http_json

COMPONENT_NAMES = (
    "reranker", "embeddings", "library", "provider", "verifier", "upstream", "github", "corpus",
)  # fmt: skip
OLLAMA_TTL = 60.0
GITHUB_TTL = 300.0

CallTool = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


def _has_model(models: set[str], name: str) -> bool:
    base = name.split(":", maxsplit=1)[0]
    return (
        name in models
        or f"{name}:latest" in models
        or any(m.split(":")[0] == base for m in models if ":" not in name)
    )


class Hub:
    def __init__(
        self,
        ledger: Ledger,
        config: dict[str, Any],
        settings: Settings,
        *,
        upstreams: Callable[[], tuple[set[str], dict[str, str]]] = lambda: (set(), {}),
        call_tool: CallTool | None = None,
        has_tool: Callable[[str], bool] = lambda _name: False,
        corpus: Callable[[], Path | None] = lambda: None,
        meters: Meters | None = None,
        breakers: Breakers | None = None,
        fetch: Fetch = http_json,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.ledger = ledger
        self.config = config
        self.settings = settings
        self.upstreams = upstreams
        self.call_tool = call_tool
        self.has_tool = has_tool
        self.corpus = corpus
        self.fetch = fetch
        self.clock = clock
        self.meters = meters or Meters(config, settings, fetch=fetch)
        self.breakers = breakers or Breakers(config, clock=clock)
        health = config.get("health", {})
        self.embed_model = str(health.get("embed_model", "bge-m3"))
        # What retrieval looked like when it last worked: a reranker name, hybrid or lexical.
        self.seen: dict[str, str] = {}
        self._ollama: tuple[float, set[str] | None] | None = None
        self._github: tuple[float, dict[str, Any]] | None = None
        self._refresh: asyncio.Task[Any] | None = None

    # -- cached probes -------------------------------------------------------------------------
    async def ollama_models(self) -> set[str] | None:
        """Installed Ollama models, or None when Ollama does not answer."""
        if self._ollama and self.clock() - self._ollama[0] < OLLAMA_TTL:
            return self._ollama[1]
        try:
            tags = await asyncio.to_thread(self.fetch, f"{self.settings.ollama_url}/api/tags", {})
            models: set[str] | None = {str(m.get("name", "")) for m in tags.get("models", [])}
        except (OSError, ValueError):
            models = None
        self._ollama = (self.clock(), models)
        return models

    async def github(self) -> dict[str, Any]:
        if self._github and self.clock() - self._github[0] < GITHUB_TTL:
            return self._github[1]
        if self.call_tool is None or not self.has_tool("repos__auth_status"):
            state = {"state": "down", "detail": "the repos upstream is not connected"}
        else:
            try:
                got = await self.call_tool("repos__auth_status", {})
                ok = bool(got.get("authenticated"))
                state = {"state": "up" if ok else "down", "detail": str(got.get("detail", ""))}
            except Exception as exc:  # noqa: BLE001 - any failure means GitHub is not usable now
                state = {"state": "down", "detail": f"{type(exc).__name__}: {exc}"[:300]}
        self._github = (self.clock(), state)
        return state

    def provider_ladder(self, step: str = "writing") -> list[str]:
        return [str(n) for n in self.config.get("ladders", {}).get(step, [])]

    async def first_usable(self, step: str = "writing") -> tuple[str | None, list[str]]:
        """The first rung the pipeline would use, and why the ones above it were passed."""
        passed: list[str] = []
        names = self.provider_ladder(step)
        for i, name in enumerate(names):
            breaker = self.breakers[f"provider:{name}"]
            if not breaker.allow():
                passed.append(f"{name}: circuit open ({breaker.last_error})")
                continue
            ok, why = await asyncio.to_thread(self.meters.usable, name)
            if ok or (i == len(names) - 1 and "down" not in why):
                return name, passed
            passed.append(why)
        return None, passed

    def refresh_usage_soon(self) -> None:
        """Read the meters in the background when the cache is stale; never waits on them."""
        if self._refresh is not None and not self._refresh.done():
            return
        stale = [n for n in self.meters.providers if self.meters.cached(n) is None]
        if stale or self._refresh is None:
            self._refresh = asyncio.get_running_loop().create_task(
                asyncio.to_thread(self.meters.all)
            )

    # -- per-component checks ----------------------------------------------------------------
    def _with_breaker(self, name: str, state: str, detail: str) -> dict[str, Any]:
        breaker = self.breakers[name]
        if breaker.state == "open":
            state = "down"
        return {"state": state, "detail": detail, "breaker": breaker.as_dict()}

    async def check(self, component: str) -> dict[str, Any]:
        connected, errors = self.upstreams()
        if component == "reranker":
            seen = self.seen.get("reranker", "")
            return self._with_breaker(
                "reranker", "up" if seen else "unknown", seen or "no reranked retrieval seen yet"
            )
        if component in ("embeddings", "verifier"):
            models = await self.ollama_models()
            want = self.embed_model if component == "embeddings" else self.settings.verifier_model
            if not want:
                return self._with_breaker(component, "down", "no verifier is configured")
            if models is None:
                return self._with_breaker(component, "down", "Ollama does not answer")
            ok = all(_has_model(models, m.strip()) for m in want.split(",") if m.strip())
            return self._with_breaker(
                component, "up" if ok else "down", want if ok else f"{want} is not installed"
            )
        if component == "library":
            name = self.settings.papers_upstream
            if name in connected:
                return self._with_breaker("library", "up", f"upstream '{name}' connected")
            return self._with_breaker(
                "library", "down", errors.get(name, f"upstream '{name}' is not connected")
            )
        if component == "provider":
            used, passed = await self.first_usable()
            ladder = self.provider_ladder()
            state = "up" if used == (ladder[0] if ladder else None) else "degraded"
            if used is None:
                state = "down"
            return {"state": state, "detail": used or "no usable provider", "passed": passed}
        if component == "upstream":
            if errors:
                return {"state": "down", "detail": errors}
            return {"state": "up", "detail": sorted(connected)}
        if component == "github":
            return await self.github()
        if component == "corpus":
            root = self.corpus()
            if root is None:
                return {"state": "down", "detail": "no research corpus checkout found"}
            if "research" not in connected:
                return {"state": "down", "detail": "the research upstream is not connected"}
            return {"state": "up", "detail": str(root)}
        raise ValueError(f"unknown component {component!r}")

    async def report(self) -> dict[str, Any]:
        checks = await asyncio.gather(*(self.check(c) for c in COMPONENT_NAMES))
        return dict(zip(COMPONENT_NAMES, checks, strict=True))
