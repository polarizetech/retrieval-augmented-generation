"""Entry points and pluggable parts: the CLI, rerankers, eval scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks.research.eval_pipeline import score
from research_pipeline import __main__ as cli
from research_pipeline import rerank
from research_pipeline.config import Settings
from research_pipeline.papers import Passage
from tests.conftest import LibraryAdapter, papers_library


class NamedModel:
    s = Settings(text_model="text")


def passages(*scores: float) -> list[Passage]:
    return [Passage(f"W#p{i}", "W", i, 0, 1, f"text {i}", score=s) for i, s in enumerate(scores)]


class TestRerankers:
    def test_none_keeps_the_fused_retrieval_order(self) -> None:
        assert rerank.NoReranker().score("q", passages(0.3, 0.1)) == [0.3, 0.1]

    def test_llm_scores_in_batches_and_breaks_ties_by_retrieval(self) -> None:
        class Scorer(NamedModel):
            def chat_json(
                self, task: str, system: str, user: str, schema: dict[str, Any], **_: Any
            ) -> dict[str, Any]:
                return {"scores": [1.0] * user.count("Passage ")}

        ranker = rerank.LLMReranker(Scorer(), batch=2)  # type: ignore[arg-type]
        assert ranker.score("q", passages(0.3, 0.1, 0.2)) == [1.3, 1.1, 1.2]
        assert ranker.name == "llm:text"

    def test_build_honours_the_setting(self) -> None:
        llm: Any = NamedModel()
        assert rerank.build(Settings(reranker="none"), llm).name == "none"
        assert isinstance(rerank.build(Settings(reranker="llm"), llm), rerank.LLMReranker)
        assert isinstance(rerank.build(Settings(reranker="auto"), llm), rerank.LLMReranker)
        # A client model never reranks: a turn per four passages. It keeps the library's order.
        assert rerank.build(Settings(reranker="auto"), None).name == "none"
        with pytest.raises(ValueError, match="needs the local model"):
            rerank.build(Settings(reranker="llm"), None)

    def test_the_cross_encoder_setting_points_to_the_library(self) -> None:
        with pytest.raises(ValueError, match="PAPER_FETCH_RERANK_MODEL"):
            rerank.build(Settings(reranker="onnx"), NamedModel())  # type: ignore[arg-type]


class TestCli:
    def run(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *argv: str) -> int:
        monkeypatch.setenv("PIPELINE_DATA_DIR", str(tmp_path))
        library = papers_library(tmp_path)
        monkeypatch.setattr(cli, "PaperLibrary", lambda *_: LibraryAdapter(library))
        monkeypatch.setattr(sys, "argv", ["research-pipeline", *argv])
        return cli.main()

    def test_status_reports_the_library_index(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert self.run(monkeypatch, tmp_path, "status") == 0
        out = json.loads(capsys.readouterr().out)
        assert out["passage_index"]["papers"] == 0
        assert out["passage_index"]["embedding_model"] == "fake-embed"

    def test_index_asks_the_library(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert self.run(monkeypatch, tmp_path, "index") == 0
        assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["papers"] == 0

    def test_domains_lists_what_is_installed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli.domain_registry, "discover", lambda: [])
        assert self.run(monkeypatch, tmp_path, "domains") == 0
        assert "generic policy" in capsys.readouterr().out


class TestEvalScoring:
    log: dict[str, Any] = {  # noqa: RUF012 - read-only fixture
        "seconds": 1.0,
        "claims": [
            {"text": "Moderate drinking is associated with lower risk.", "verdict": "SUPPORTED"}
        ],
        "evidence": [
            {"work": "A", "role": "E", "direction": "affirms"},
            {"work": "B", "role": "E", "direction": "denies"},
        ],
        "papers": {"A": {"doi": "10.1/A"}, "B": {"doi": "10.1/b"}},
        "candidates": [{"work": "C", "ids": {"doi": "10.1/c"}}],
    }

    def test_recall_counts_only_papers_that_became_evidence(self) -> None:
        case = {"id": "x", "kind": "k", "expect_dois": ["10.1/a", "10.1/c"]}
        # 10.1/c was discovered but never became evidence, so it does not count.
        assert score(case, self.log)["source_recall"] == 0.5

    def test_conflict_needs_different_papers_on_each_side(self) -> None:
        case = {"id": "x", "kind": "k", "expect_conflict": True}
        assert score(case, self.log)["conflict_shown"] is True

    def test_forbidden_phrasing_fails_the_case(self) -> None:
        case = {"id": "x", "kind": "k", "must_not_claim": ["associated"]}
        row = score(case, self.log)
        assert row["forbidden"] == 1
        assert row["pass"] is False

    def test_an_unanswerable_question_must_abstain(self) -> None:
        row = score({"id": "x", "kind": "k", "answerable": False}, self.log)
        assert row["abstained"] is False
        assert row["pass"] is False


def test_a_rerank_batch_with_no_answer_keeps_retrieval_order() -> None:
    from research_pipeline.llm import ModelOutputError

    class Stalls(NamedModel):
        asked = 0

        def chat_json(self, *_: Any, **__: Any) -> dict[str, Any]:
            self.asked += 1
            if self.asked == 1:
                raise ModelOutputError("rerank: cut off at the 2048-token cap")
            return {"scores": [5.0]}

    ranker = rerank.LLMReranker(Stalls(), batch=2)  # type: ignore[arg-type]
    # The first batch (two passages) failed: retrieval scores only. The second was scored.
    assert ranker.score("q", passages(0.3, 0.1, 0.2)) == [0.3, 0.1, 5.2]
    assert ranker.failed == 1
