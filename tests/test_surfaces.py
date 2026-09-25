"""Entry points and pluggable parts: the CLI, the pipeline MCP server, rerankers, eval scoring."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks.research.eval_pipeline import score
from research_pipeline import __main__ as cli
from research_pipeline import mcp_server, rerank
from research_pipeline.config import Settings
from research_pipeline.index import Passage


class NamedModel:
    s = Settings(text_model="text")


def passages(*scores: float) -> list[Passage]:
    return [Passage(i, "W", i, 0, 1, f"text {i}", fused=s) for i, s in enumerate(scores)]


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
        assert isinstance(
            rerank.build(Settings(reranker="llm", reranker_model=""), llm), rerank.LLMReranker
        )
        with pytest.raises(RuntimeError, match="needs PIPELINE_RERANKER_MODEL"):
            rerank.build(Settings(reranker="onnx", reranker_model=""), llm)

    def test_a_forced_onnx_reranker_that_cannot_load_is_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unavailable(*_: Any) -> None:
            raise ImportError("onnxruntime is not installed")

        monkeypatch.setattr(rerank, "OnnxReranker", unavailable)
        llm: Any = NamedModel()
        settings = Settings(reranker="onnx", reranker_model="repo::model.onnx")
        with pytest.raises(RuntimeError, match="did not load"):
            rerank.build(settings, llm)
        auto = rerank.build(Settings(reranker="auto", reranker_model="repo::model.onnx"), llm)
        assert isinstance(auto, rerank.LLMReranker)


class TestCli:
    def run(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *argv: str) -> int:
        monkeypatch.setenv("PIPELINE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("PIPELINE_EMBEDDING_MODEL", "fake-embed")
        monkeypatch.setattr(sys, "argv", ["research-pipeline", *argv])
        return cli.main()

    def test_status_reports_the_index(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert self.run(monkeypatch, tmp_path, "status") == 0
        assert "'papers': 0" in capsys.readouterr().out

    def test_domains_lists_what_is_installed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli.domain_registry, "discover", lambda: [])
        assert self.run(monkeypatch, tmp_path, "domains") == 0
        assert "generic policy" in capsys.readouterr().out


class TestPipelineServer:
    def test_start_status_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Finished:
            def __init__(self, settings: Settings, progress: Any, offline: bool) -> None:
                self.progress = progress

            async def run(self, question: str) -> dict[str, Any]:
                self.progress("plan", "planning")
                return {
                    "run_dir": "runs/x",
                    "answer": f"# {question}",
                    "seconds": 1.0,
                    "searches": [],
                    "evidence": [],
                    "claims": [{"verdict": "SUPPORTED"}, {"verdict": "NOT_SUPPORTED"}],
                }

        monkeypatch.setattr(mcp_server, "Pipeline", Finished)

        async def scenario() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            started = json.loads(await mcp_server.research_start("Does X affect Y?"))
            busy = json.loads(await mcp_server.research_start("another"))
            assert "already in progress" in busy["error"]
            assert mcp_server._active is not None
            await mcp_server._active
            status = json.loads(await mcp_server.research_status(started["run_id"]))
            result = json.loads(await mcp_server.research_result(started["run_id"]))
            return started, status, result

        started, status, result = asyncio.run(scenario())
        assert status["done"] is True
        assert result["answer"] == "# Does X affect Y?"
        assert result["stats"]["claims_kept"] == 1
        assert (
            "unknown run_id" in json.loads(asyncio.run(mcp_server.research_status("nope")))["error"]
        )
        assert started["run_id"] in mcp_server.RUNS

    def test_a_failed_run_is_reported_not_lost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Broken:
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

            async def run(self, question: str) -> dict[str, Any]:
                raise RuntimeError("ollama unreachable")

        monkeypatch.setattr(mcp_server, "Pipeline", Broken)

        async def scenario() -> dict[str, Any]:
            started = json.loads(await mcp_server.research_start("q"))
            assert mcp_server._active is not None
            await mcp_server._active
            return json.loads(await mcp_server.research_result(started["run_id"]))

        assert "ollama unreachable" in asyncio.run(scenario())["error"]


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
