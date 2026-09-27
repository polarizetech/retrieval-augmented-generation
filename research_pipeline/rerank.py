"""Rerankers. Hybrid retrieval finds passages on the topic; reranking finds the ones that answer.

Three interchangeable implementations:
  onnx  a cross-encoder run with onnxruntime (the optional [rerank] extra), no torch
  llm   the text model scores passages in small batches; no download, slower, weaker
  none  keep the fused retrieval order (for ablation in the eval suite)
`auto` uses onnx when a model is configured and loadable, else llm.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from . import prompts as _prompts
from .config import Settings
from .index import Passage
from .llm import Ollama
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


class OnnxReranker:
    """Cross-encoder from the Hugging Face hub in ONNX form, e.g. a quantised bge-reranker."""

    def __init__(
        self, repo: str, onnx_file: str = "onnx/model_quantized.onnx", max_length: int = 512
    ):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.name = f"onnx:{repo}/{onnx_file}"
        self.session = ort.InferenceSession(
            hf_hub_download(repo, onnx_file), providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.inputs = {i.name for i in self.session.get_inputs()}

    def score(self, question: str, passages: list[Passage], batch: int = 8) -> list[float]:
        out: list[float] = []
        for i in range(0, len(passages), batch):
            enc = self.tokenizer.encode_batch([(question, p.text) for p in passages[i : i + batch]])
            width = max(len(e.ids) for e in enc)
            feed = {
                "input_ids": np.array(
                    [e.ids + [1] * (width - len(e.ids)) for e in enc], dtype=np.int64
                ),
                "attention_mask": np.array(
                    [e.attention_mask + [0] * (width - len(e.ids)) for e in enc], dtype=np.int64
                ),
                "token_type_ids": np.array(
                    [e.type_ids + [0] * (width - len(e.ids)) for e in enc], dtype=np.int64
                ),
            }
            logits = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            out.extend(float(x) for x in np.asarray(logits).reshape(len(enc), -1)[:, 0])
        return out


def build(settings: Settings, llm: Ollama | None, prompts: Prompts = DEFAULT) -> Reranker:
    """The configured reranker. Without a local model, a failed ONNX load keeps retrieval order."""
    choice = settings.reranker
    if choice == "none":
        return NoReranker()
    if choice in ("onnx", "auto") and settings.reranker_model:
        try:
            repo, _, file = settings.reranker_model.partition("::")
            return OnnxReranker(repo, file) if file else OnnxReranker(repo)
        except Exception as exc:
            if choice == "onnx":
                raise RuntimeError(
                    f"PIPELINE_RERANKER=onnx but the model did not load: {exc}"
                ) from exc
    elif choice == "onnx":
        raise RuntimeError("PIPELINE_RERANKER=onnx needs PIPELINE_RERANKER_MODEL")
    return LLMReranker(llm, prompts=prompts) if llm is not None else NoReranker()
