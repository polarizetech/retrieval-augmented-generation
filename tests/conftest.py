"""Shared fixtures. Every test runs offline: no Ollama, no network.

The paper library is real: paper-fetch's `Library` over an in-memory store with no providers and
a fake embedder, reached through `LibraryAdapter`, which speaks the same contract as the MCP client
(`research_pipeline.papers.PaperLibrary`). So retrieval, chunking, offsets and passage ids in these
tests are paper-fetch's own.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from paper_fetch import Library, MemoryStore, NotFound, OpenAlex
from paper_fetch.passages import PassageIndex

from research_pipeline.papers import PapersError, Passage
from research_pipeline.schema import Evidence

# A synthetic paper. The topic is deliberately ordinary; what matters is the structure: a result
# paragraph, a null-result paragraph, and a reference list that must not be indexed.
PAPER = (
    "Resting systolic blood pressure was measured in 48 adults before and after twelve weeks of "
    "supervised aerobic training. "
    * 8
    + "\n\n"
    + "Aerobic training lowered resting systolic blood pressure by 5 mmHg relative to control "
    "(p = 0.02). "
    * 10
    + "\n\n"
    + "Training did not change resting heart rate in the same participants (p = 0.41). " * 10
    + "\n\nReferences\n\n1. Someone A. An earlier trial of exercise and blood pressure. 2016."
)

Embed = Callable[[list[str]], list[list[float]]]


def fake_embed(texts: list[str]) -> list[list[float]]:
    """A two-dimensional bag of words: enough to make dense ranking deterministic."""
    return [
        [float(t.lower().count("lowered")) + 0.01, float(t.lower().count("heart")) + 0.01]
        for t in texts
    ]


class OfflineHttp:
    calls = 0

    def get(self, *_: Any, **__: Any) -> Any:
        raise ConnectionError("tests are offline")

    def post_json(self, *_: Any, **__: Any) -> Any:
        raise ConnectionError("tests are offline")


def papers_library(tmp_path: Path) -> Library:
    """A paper-fetch library that holds nothing yet, never touches the network, and embeds with
    `fake_embed`."""
    lib = Library(
        MemoryStore(),
        OpenAlex(http=OfflineHttp(), api_key="", email=""),  # type: ignore[arg-type]
        providers=[],
        search_providers=[],
    )
    lib.passages = PassageIndex(tmp_path / "passages.sqlite", "fake-embed", fake_embed)
    lib.reranker = None
    return lib


def hold(
    lib: Library,
    work: str,
    text: str,
    *,
    doi: str,
    title: str,
    year: int = 2020,
    authors: tuple[str, ...] = ("Ada Lovelace", "Alan Turing"),
    retracted: bool = False,
    route: str = "fixture:txt",
    full_text: bool = True,
) -> None:
    """Put a paper into the library as a fetch would have: with readable full text, or (with
    `full_text=False`) as a paper no open copy was found for, not to be asked about again soon."""
    base = f"papers/works/{work}/"
    if full_text:
        lib.store.put(base + "fulltext.txt", text.encode())
    record = {
        "id": f"https://openalex.org/{work}",
        "doi": f"https://doi.org/{doi}",
        "title": title,
        "publication_year": year,
        "authorships": [{"author": {"display_name": a}} for a in authors],
        "is_retracted": retracted,
        "ids": {},
    }
    lib.store.put(base + "work.json", json.dumps(record).encode())
    provenance = {
        "work": work,
        "full_text": full_text,
        "route": route if full_text else None,
        "format": "txt" if full_text else None,
        "license": "cc-by" if full_text else None,
        "retrieved": "2026-01-01T00:00:00Z",
        "ids": {"doi": doi},
        "sha256": {},
        "retry_after": None if full_text else "2999-01-01T00:00:00Z",
    }
    lib.store.put(base + "provenance.json", json.dumps(provenance).encode())
    lib.rebuild_index()


def hold_w1(lib: Library, text: str = PAPER) -> None:
    hold(lib, "W1", text, doi="10.1000/w1", title="Training and blood pressure")


class LibraryAdapter:
    """`PaperLibrary`'s contract over an in-process paper-fetch `Library`."""

    def __init__(self, lib: Library) -> None:
        self.lib = lib
        self.calls: list[str] = []

    async def __aenter__(self) -> LibraryAdapter:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def _run(self, name: str, fn: Callable[[], Any]) -> Any:
        self.calls.append(name)
        try:
            return fn()
        except NotFound as exc:
            raise PapersError(f"{name}: {exc}", "not_found") from exc
        except ConnectionError as exc:
            raise PapersError(f"{name}: {exc}", "unavailable") from exc
        except (ValueError, RuntimeError) as exc:
            raise PapersError(f"{name}: {exc}", "tool_error") from exc

    async def search(
        self,
        query: str,
        limit: int = 10,
        include_closed: bool = True,
        profile: str = "",
        collection: str = "",
    ) -> dict[str, Any]:
        return self._run(
            "search",
            lambda: self.lib.search(
                query,
                limit=limit,
                oa_only=not include_closed,
                profile=profile or None,
                collection=collection or None,
                web_fallback=False,
            ),
        )

    async def fetch(self, identifier: str, collection: str = "") -> dict[str, Any]:
        return self._run("fetch", lambda: self.lib.fetch(identifier, collection=collection or None))

    async def retrieve(
        self,
        queries: list[str],
        *,
        identifiers: list[str] | None = None,
        collection: str = "",
        limit: int = 20,
        per_paper: int = 0,
    ) -> dict[str, Any]:
        return self._run(
            "retrieve",
            lambda: self.lib.retrieve(
                queries,
                identifiers=identifiers or None,
                collection=collection or None,
                limit=limit,
                per_paper=per_paper or None,
            ),
        )

    async def passages(self, identifier: str, start: int = 0, limit: int = 4) -> list[Passage]:
        data = self._run("passages", lambda: self.lib.paper_passages(identifier, start, limit))
        return [Passage.from_json(p) for p in data["passages"]]

    async def index(self, identifiers: list[str]) -> dict[str, Any]:
        return self._run(
            "index",
            lambda: {**self.lib.index_works(identifiers), "index": self.lib.passages.stats()},
        )

    async def relevance(self, targets: list[str], texts: list[str]) -> list[float]:
        return self._run("relevance", lambda: self.lib.relevance(targets, texts))["scores"]

    async def profile(self, slug: str) -> dict[str, Any] | None:
        known = self.lib.profiles()
        return known[slug].guidance() if slug in known else None

    async def status(self) -> dict[str, Any]:
        return self._run("status", self.lib.status)

    async def held(self, limit: int = 100000) -> list[dict[str, Any]]:
        return list(self.lib.index().values())[:limit]


@pytest.fixture
def papers(tmp_path: Path) -> Iterator[Library]:
    """A paper-fetch library holding W1 (PAPER), indexed."""
    lib = papers_library(tmp_path)
    hold_w1(lib)
    lib.index_works()
    yield lib
    lib.passages.close()


def make_evidence(eid: str, work: str, direction: str = "affirms", **overrides: Any) -> Evidence:
    fields: dict[str, Any] = {
        "id": eid,
        "subquestion": "S1",
        "work": work,
        "passage_id": "W1#p0",
        "start": 0,
        "end": 10,
        "direction": direction,
        "finding": "f",
        "quote": "q",
        "study_type": "primary_human",
        "population": "adults",
        "secondhand": False,
        "role": "E",
    }
    return Evidence(**(fields | overrides))
