"""A whole offline run with a scripted model standing in for Ollama.

The stub answers each stage the way a small model plausibly would, including two mistakes the
pipeline must catch: a claim with a number its source does not contain, and a citation to an
evidence id that was never retrieved. The test then checks what reached the answer and the log.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

from research_pipeline import verify
from research_pipeline.config import Settings
from research_pipeline.index import PassageIndex
from research_pipeline.pipeline import Pipeline
from tests.conftest import PAPER, fake_embed

QUESTION = "Does aerobic training lower resting blood pressure in adults?"
EFFECT = "Aerobic training lowered resting systolic blood pressure by 5 mmHg relative to control"
NULL = "Training did not change resting heart rate in the same participants (p = 0.41)."


class ScriptedModel:
    """Implements the parts of research_pipeline.llm.Ollama the pipeline calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def resolve(self, model: str) -> tuple[str, str]:
        return model, "sha256:test"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return fake_embed(texts)

    def chat_text(self, task: str, model: str, user: str, **_: Any) -> str:
        raise AssertionError("no classifier verifier is configured in this test")

    def chat_json(
        self, task: str, system: str, user: str, schema: dict[str, Any], **_: Any
    ) -> dict[str, Any]:
        self.calls.append({"task": task})
        return getattr(self, task)(user)

    def plan(self, _: str) -> dict[str, Any]:
        return {
            "mode": "multi_paper_synthesis",
            "core_question": QUESTION,
            "subquestions": [
                {
                    "question": "Does aerobic training lower systolic blood pressure?",
                    "queries": ["aerobic training lowered systolic blood pressure"],
                }
            ],
            "falsification_queries": ["training did not change heart rate"],
        }

    def extract(self, user: str) -> dict[str, Any]:
        base = {"population": "adults", "secondhand": False}
        if "lowered" in user:
            return base | {
                "relevant": True,
                "direction": "affirms",
                "finding": "Training lowered systolic pressure by 5 mmHg.",
                "quote": EFFECT + " (p = 0.02).",
            }
        if "did not change" in user:
            return base | {
                "relevant": True,
                "direction": "denies",
                "finding": "Training did not change heart rate.",
                "quote": NULL,
            }
        return base | {"relevant": False, "direction": "neutral", "finding": "", "quote": ""}

    def classify_paper(self, _: str) -> dict[str, Any]:
        return {"study_type": "primary_human", "population": "adults"}

    def synthesise(self, user: str) -> dict[str, Any]:
        ids = re.findall(r"^\[(S\d+-E\d+)\]", user, flags=re.M)
        if not ids:
            return {"claims": [], "insufficient": True}
        return {
            "insufficient": False,
            "claims": [
                {
                    "text": "Aerobic training lowered resting systolic blood pressure by 5 mmHg.",
                    "evidence_ids": ids[:1],
                },
                {
                    "text": "Aerobic training lowered resting systolic blood pressure by 9 mmHg.",
                    "evidence_ids": ids[:1],
                },
                {"text": "Training cured hypertension.", "evidence_ids": ["S1-E99"]},
            ],
        }

    def critique(self, _: str) -> dict[str, Any]:
        return {"gaps": []}

    def verify(self, _: str) -> dict[str, Any]:
        return {"closest_sentence": "", "claim_adds": "", "verdict": "SUPPORTED"}

    def summarise(self, user: str) -> dict[str, Any]:
        ids = re.findall(r"^\[(S\d+-C\d+)\]", user, flags=re.M)
        return {
            "sentences": [
                {
                    "text": "Aerobic training lowered systolic pressure by 5 mmHg.",
                    "claim_ids": ids[:1],
                }
            ]
        }


@pytest.fixture
def run_log(tmp_path: Path) -> dict[str, Any]:
    settings = Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        embedding_model="fake-embed",
        reranker="none",
        verifier_model="",
        text_model="stub-model",
        max_rounds=2,
    )
    PassageIndex(settings.index_path, settings.embedding_model).add(
        {
            "work": "W1",
            "doi": "10.1000/w1",
            "title": "Training and blood pressure",
            "year": 2020,
            "authors": ["Ada Lovelace"],
            "route": "fixture",
        },
        PAPER,
        fake_embed,
    )
    pipeline = Pipeline(settings, offline=True)
    pipeline.llm = ScriptedModel()  # type: ignore[assignment]
    return asyncio.run(pipeline.run(QUESTION))


def claims_by_text(log: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["text"]: c for c in log["claims"]}


def test_a_correct_claim_is_kept_with_its_quote(run_log: dict[str, Any]) -> None:
    kept = claims_by_text(run_log)[
        "Aerobic training lowered resting systolic blood pressure by 5 mmHg."
    ]
    assert kept["verdict"] == "SUPPORTED"
    assert "## Bottom line" in run_log["answer"]
    assert EFFECT in run_log["answer"]


def test_a_wrong_number_removes_the_claim_even_when_the_verifier_accepts_it(
    run_log: dict[str, Any],
) -> None:
    wrong = claims_by_text(run_log)[
        "Aerobic training lowered resting systolic blood pressure by 9 mmHg."
    ]
    assert wrong["verdict"] == verify.VERDICT_NUMBERS
    assert "9 mmHg" not in run_log["answer"].split("## Limits")[0]
    assert "stated a number their cited passage does not contain" in run_log["answer"]


def test_an_unretrieved_evidence_id_never_becomes_a_claim(run_log: dict[str, Any]) -> None:
    assert "Training cured hypertension." not in claims_by_text(run_log)
    assert any("S1-E99" in note for note in run_log["notes"])


def test_the_null_result_is_reported_as_opposing_evidence(run_log: dict[str, Any]) -> None:
    directions = {e["direction"] for e in run_log["evidence"]}
    assert directions == {"affirms", "denies"}


def test_self_verification_is_disclosed(run_log: dict[str, Any]) -> None:
    assert run_log["models"]["verifiers"] == [{"name": "stub-model", "digest": "sha256:test"}]
    assert "shares the writer's blind spots" in run_log["answer"]


def test_the_run_log_holds_passages_models_and_the_answer(run_log: dict[str, Any]) -> None:
    saved = json.loads((Path(run_log["run_dir"]) / "run.json").read_text())
    assert saved["passages"]
    assert all(e["quote"] in saved["passages"][e["id"]] for e in saved["evidence"])
    assert saved["domain"]["name"].endswith("generic")
    assert (Path(run_log["run_dir"]) / "answer.md").read_text() == run_log["answer"]


def test_offline_runs_say_so_and_skip_retraction_checks(run_log: dict[str, Any]) -> None:
    assert run_log["integrity"] == {}
    assert "Offline run" in run_log["answer"]
    assert "editorial status not checked" in run_log["answer"]


class FakeLibrary:
    """The paper library's client interface, with one provider that never answers."""

    fetched: list[str]

    def __init__(self, *_: Any) -> None:
        self.fetched = []

    async def __aenter__(self) -> FakeLibrary:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def search(self, query: str, limit: int = 10) -> dict[str, Any]:
        hits = [
            {
                "title": "Training and blood pressure",
                "year": 2020,
                "ids": {"doi": "10.1000/w1"},
                "authors": ["Ada Lovelace"],
                "providers": ["a"],
            },
            {
                "title": "A closed-access trial",
                "year": 2021,
                "ids": {"doi": "10.1000/closed"},
                "providers": ["a"],
            },
        ]
        return {"hits": hits, "providers": {"a": {"status": "ok"}, "b": {"status": "unavailable"}}}

    async def fetch(self, identifier: str) -> dict[str, Any]:
        self.fetched.append(identifier)
        if identifier == "10.1000/closed":
            return {"work": "W2", "full_text": False}
        return {
            "work": "W1",
            "doi": "10.1000/w1",
            "title": "Training and blood pressure",
            "year": 2020,
            "authors": ["Ada Lovelace"],
            "full_text": True,
            "route": "fake",
        }

    async def full_text(self, identifier: str) -> str:
        return PAPER.replace("Resting", "Resting\u200b\u200b\u200b\u200b", 1)


class CriticalModel(ScriptedModel):
    """Asks for one more search in the critique round, then is satisfied."""

    def __init__(self) -> None:
        super().__init__()
        self.rounds = 0

    def critique(self, _: str) -> dict[str, Any]:
        self.rounds += 1
        if self.rounds > 1:
            return {"gaps": []}
        return {
            "gaps": [
                {
                    "subquestion": "S1",
                    "issue": "one paper",
                    "query": "exercise blood pressure replication",
                }
            ]
        }


@pytest.fixture
def online_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from research_pipeline import pipeline as module

    monkeypatch.setattr(module, "PaperLibrary", FakeLibrary)
    monkeypatch.setattr(
        module.integrity, "check_all", lambda dois: {d: {"status": "retracted"} for d in dois}
    )
    settings = Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        embedding_model="fake-embed",
        reranker="none",
        verifier_model="",
        text_model="stub-model",
        max_rounds=2,
        max_fetch=5,
    )
    pipeline = Pipeline(settings, offline=False)
    pipeline.llm = CriticalModel()  # type: ignore[assignment]
    return asyncio.run(pipeline.run(QUESTION))


def test_silent_providers_are_recorded_not_read_as_empty(online_log: dict[str, Any]) -> None:
    assert all(s["did_not_answer"] == ["b"] for s in online_log["searches"])
    assert "did not answer at least once: b" in online_log["answer"]


def test_critique_produces_new_searches(online_log: dict[str, Any]) -> None:
    queries = [s["query"] for s in online_log["searches"]]
    assert "exercise blood pressure replication" in queries
    assert "training did not change heart rate" in queries  # the falsification pass always runs


def test_unreadable_papers_are_listed_as_limits(online_log: dict[str, Any]) -> None:
    outcomes = {c["key"]: c["outcome"] for c in online_log["candidates"]}
    assert outcomes == {"10.1000/w1": "indexed", "10.1000/closed": "not_obtainable"}
    assert "A closed-access trial" in online_log["answer"]


def test_hidden_characters_are_noted_before_they_are_stripped(online_log: dict[str, Any]) -> None:
    assert any("4 hidden characters" in note for note in online_log["notes"])


def test_a_retracted_source_supports_nothing(online_log: dict[str, Any]) -> None:
    verdicts = {c["verdict"] for c in online_log["claims"]}
    assert "SUPPORTED" not in verdicts
    assert "RETRACTED_SOURCE" in verdicts
    assert "Crossref/Retraction Watch lists their source as retracted" in online_log["answer"]


class ComponentLibrary(FakeLibrary):
    """Lists a figure of an article as its own work, before the article itself."""

    async def search(self, query: str, limit: int = 10) -> dict[str, Any]:
        return {
            "hits": [
                {
                    "title": "Simulating the influence of precision on uncertainty.",
                    "ids": {"doi": "10.1371/journal.pcbi.1010490.g004", "openalex": "W-fig"},
                    "work": "W-fig",
                },
                {
                    "title": "In the Body's Eye",
                    "ids": {"doi": "10.1371/journal.pcbi.1010490"},
                    "year": 2022,
                },
            ],
            "providers": {"a": {"status": "ok"}},
        }


def test_a_figure_doi_is_folded_into_its_article(tmp_path: Path) -> None:
    from research_pipeline.pipeline import State

    settings = Settings(
        data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", embedding_model="fake-embed"
    )
    pipeline = Pipeline(settings, offline=False)
    st = State(QUESTION)
    asyncio.run(pipeline.discover(st, ComponentLibrary(), ["precision"]))  # type: ignore[arg-type]
    assert list(st.candidates) == ["10.1371/journal.pcbi.1010490"]
    cand = st.candidates["10.1371/journal.pcbi.1010490"]
    assert cand.title == "In the Body's Eye"
    assert cand.work is None
    assert cand.ids == {"doi": "10.1371/journal.pcbi.1010490"}
    assert any("component" in note for note in st.notes)
