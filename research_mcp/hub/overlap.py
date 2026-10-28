"""Duplicate and near-duplicate research questions (docs/hub-plan.md, section 3).

A research question that matches one already on the go attaches to it instead of starting a
second run: the same normalised words, or a bge-m3 embedding similarity at or above `attach`.
Between `warn` and `attach` the new run starts, with a warning naming the other request. Without
embeddings (Ollama down) only exact matches attach, and the reply does not claim more.
"""

from __future__ import annotations

import asyncio
import json
import math
import urllib.request
from collections.abc import Callable
from typing import Any

from research_mcp.hub.ledger import fingerprint

Post = Callable[[str, dict[str, Any]], Any]
EMBED_TIMEOUT = 10.0


def http_post(url: str, body: dict[str, Any]) -> Any:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as response:
        return json.load(response)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class Overlap:
    def __init__(self, config: dict[str, Any], ollama_url: str, post: Post = http_post):
        cfg = config.get("duplicates", {})
        self.attach = float(cfg.get("attach", 0.92))
        self.warn = float(cfg.get("warn", 0.80))
        self.model = str(config.get("health", {}).get("embed_model", "bge-m3"))
        self.url = ollama_url.rstrip("/") + "/api/embed"
        self.post = post
        self._vectors: dict[str, list[float]] = {}  # by normalised-question fingerprint

    async def _embed(self, texts: list[str]) -> list[list[float]] | None:
        missing = [t for t in texts if fingerprint(t) not in self._vectors]
        if missing:
            try:
                got = await asyncio.to_thread(
                    self.post, self.url, {"model": self.model, "input": missing}
                )
                for text, vector in zip(missing, got["embeddings"], strict=True):
                    self._vectors[fingerprint(text)] = [float(x) for x in vector]
            except (OSError, ValueError, KeyError, TypeError):
                return None
        return [self._vectors[fingerprint(t)] for t in texts]

    async def closest(
        self, question: str, others: list[dict[str, Any]]
    ) -> tuple[dict[str, Any] | None, float, str]:
        """The most similar request on the go, its similarity, and how it was measured."""
        if not others:
            return None, 0.0, ""
        mine = fingerprint(question)
        for row in others:
            if row.get("fingerprint") == mine:
                return row, 1.0, "the same words"
        vectors = await self._embed([question, *[str(r.get("question", "")) for r in others]])
        if vectors is None:
            return None, 0.0, "embeddings unavailable: only exact matches were checked"
        scores = [cosine(vectors[0], v) for v in vectors[1:]]
        best = max(range(len(others)), key=lambda i: scores[i])
        return others[best], round(scores[best], 3), "bge-m3 similarity"
