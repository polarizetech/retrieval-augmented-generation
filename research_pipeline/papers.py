"""Client for the paper library (paper-fetch), over MCP.

paper-fetch owns everything about finding and indexing papers: discovery across providers,
discipline profiles (a field's indexed terms), collections, search memory, open-access rules,
full-text storage, provenance, and the passage index that retrieval reads. This module does not
reimplement any of it: it spawns the same stdio server the gateway federates (the `papers`
upstream in config/mcp-gateway.json) and calls its tools. What stays in this repository is what
happens after a passage is found: reading it, judging it, and checking what is claimed from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .upstream import Upstream, UpstreamError


class PapersError(UpstreamError):
    """The paper library refused or failed a call; `code` is its own (not_found, unavailable)."""


@dataclass
class Passage:
    """One retrieved passage: `text` is `indexed_text[start:end]` of its paper, exactly."""

    id: str  # "<work>#p<ord>": stable across rebuilds of paper-fetch's index
    work: str
    ord: int
    start: int
    end: int
    text: str
    sha: str = ""
    score: float = 0.0  # the library's rank score (its cross-encoder's, when it has one)
    bm25_rank: int | None = None
    dense_rank: int | None = None
    paper: dict[str, Any] = field(default_factory=dict)

    @property
    def fused(self) -> float:
        return self.score

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Passage:
        return cls(
            id=d["id"],
            work=d["work"],
            ord=d["ord"],
            start=d["start"],
            end=d["end"],
            text=d["text"],
            sha=d.get("sha", ""),
            score=float(d.get("rerank_score", d.get("score", 0.0)) or 0.0),
            bm25_rank=d.get("bm25_rank"),
            dense_rank=d.get("dense_rank"),
            paper=dict(d.get("paper") or {}),
        )


class PaperLibrary(Upstream):
    error = PapersError

    def __init__(self, gateway_config: Path, upstream: str = "papers"):
        super().__init__(gateway_config, upstream)

    async def search(
        self,
        query: str,
        limit: int = 10,
        include_closed: bool = True,
        profile: str = "",
        collection: str = "",
    ) -> dict[str, Any]:
        """Merged hits plus per-provider status. A provider that did not answer is NOT zero hits.
        With a discipline `profile`, the library also searches the field's indexed terms."""
        return (
            await self._call(
                "search",
                query=query,
                limit=limit,
                include_closed=include_closed,
                profile=profile,
                collection=collection,
            )
        )["data"]

    async def fetch(self, identifier: str, collection: str = "") -> dict[str, Any]:
        return (await self._call("fetch", identifier=identifier, collection=collection))["data"]

    async def retrieve(
        self,
        queries: list[str],
        *,
        identifiers: list[str] | None = None,
        collection: str = "",
        limit: int = 20,
        per_paper: int = 0,
    ) -> dict[str, Any]:
        """Passages that best answer each query: `{"results": [{"query", "passages"}], ...}`."""
        return (
            await self._call(
                "retrieve",
                queries=queries,
                identifiers=identifiers or [],
                collection=collection,
                limit=limit,
                per_paper=per_paper,
            )
        )["data"]

    async def passages(self, identifier: str, start: int = 0, limit: int = 4) -> list[Passage]:
        """A held paper's passages in order from position `start`."""
        data = (await self._call("passages", identifier=identifier, start=start, limit=limit))[
            "data"
        ]
        return [Passage.from_json(p) for p in data["passages"]]

    async def index(self, identifiers: list[str]) -> dict[str, Any]:
        """Put held full texts into the library's passage index (no-op for indexed ones)."""
        return (await self._call("index", identifiers=identifiers))["data"]

    async def relevance(self, targets: list[str], texts: list[str]) -> list[float]:
        return (await self._call("relevance", targets=targets, texts=texts))["data"]["scores"]

    async def profile(self, slug: str) -> dict[str, Any] | None:
        """A discipline profile's guidance for writing queries, or None if the library has none."""
        try:
            return (await self._call("profiles", slug=slug))["data"]
        except PapersError as exc:
            if exc.code == "tool_error" and "unknown profile" in str(exc):
                return None
            raise

    async def status(self) -> dict[str, Any]:
        return (await self._call("status"))["data"]

    async def held(self, limit: int = 100000) -> list[dict[str, Any]]:
        return (await self._call("library", limit=limit))["data"]["works"]

    async def full_text(self, identifier: str) -> str:
        parts: list[str] = []
        offset = 0
        while True:
            data = (
                await self._call("text", identifier=identifier, offset=offset, max_chars=100000)
            )["data"]
            parts.append(data["text"])
            offset = data["end"]
            if offset >= data["total_chars"] or not data["text"]:
                return "".join(parts)

    async def provenance(self, identifier: str) -> dict[str, Any]:
        return (await self._call("provenance", identifier=identifier))["data"]

    async def citations(
        self, identifier: str, direction: str = "citations", limit: int = 25
    ) -> dict[str, Any]:
        return (
            await self._call("citations", identifier=identifier, direction=direction, limit=limit)
        )["data"]
