"""Rerankers: hybrid retrieval finds passages on the topic; reranking finds the ones that answer.

Retrieval belongs to the paper library, which can also apply a cross-encoder of its own
(PAPER_FETCH_RERANK_MODEL); its order arrives as each passage's score. This module adds only what
needs this application's model:

  llm   the local text model scores passages in small batches
  none  keep the library's order

`auto` means llm when the pipeline's model is local, none when it is the MCP client's: a client
would spend a turn per four passages.
"""

from __future__ import annotations

from typing import Protocol

from . import prompts as _prompts
from .config import Settings
from .llm import Ollama
from .papers import Passage
from .prompts import DEFAULT, Prompts


class Reranker(Protocol):
    name: str

    def score(self, question: str, passages: list[Passage]) -> list[float]: ...


class NoReranker:
    name = "none"

    def score(self, question: str, passages: list[Passage]) -> list[float]:
        return [p.fused for p in passages]


class LLMReranker:
    def __init__(
        self, llm: Ollama, batch: int = 4, max_chars: int = 900, prompts: Prompts = DEFAULT
    ):
        self.llm, self.batch, self.max_chars = llm, batch, max_chars
        self.prompts = prompts
        self.name = f"llm:{llm.s.text_model}"

    def score(self, question: str, passages: list[Passage]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(passages), self.batch):
            group = passages[i : i + self.batch]
            body = "\n\n".join(
                f"Passage {n}:\n{_prompts.wrap(p.text[: self.max_chars])}"
                for n, p in enumerate(group, 1)
            )
            got = self.llm.chat_json(
                "rerank",
                self.prompts.rerank_system,
                f"Research question: {question}\n\n{body}\n\n"
                f"Score the {len(group)} passages in order.",
                self.prompts.rerank_schema(len(group)),
            )
            scores = [float(s) for s in got.get("scores", [])][: len(group)]
            scores += [0.0] * (len(group) - len(scores))
            # Break ties by retrieval order so equal scores keep a stable, explainable ranking.
            out.extend(s + p.fused for s, p in zip(scores, group, strict=True))
        return out


def build(settings: Settings, llm: Ollama | None, prompts: Prompts = DEFAULT) -> Reranker:
    """The configured reranker. `llm` is None when the pipeline's model is the MCP client's."""
    choice = settings.reranker
    if choice not in ("auto", "llm", "none"):
        raise ValueError(
            f"PIPELINE_RERANKER={choice!r}: use auto, llm or none. The cross-encoder moved to the "
            "paper library: set PAPER_FETCH_RERANK_MODEL there."
        )
    if choice == "llm" and llm is None:
        raise ValueError("PIPELINE_RERANKER=llm needs the local model (PIPELINE_MCP_LLM=ollama)")
    if choice == "none" or llm is None:
        return NoReranker()
    return LLMReranker(llm, prompts=prompts)
