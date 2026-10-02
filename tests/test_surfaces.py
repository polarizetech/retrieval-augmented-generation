"""Entry points: the CLI, and eval scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks.research.eval_pipeline import score
from research_pipeline import __main__ as cli
from tests.conftest import LibraryAdapter, papers_library


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
