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
from research_pipeline.pipeline import Pipeline, State
from tests.conftest import PAPER, LibraryAdapter, hold_w1, papers_library

QUESTION = "Does aerobic training lower resting blood pressure in adults?"
EFFECT = "Aerobic training lowered resting systolic blood pressure by 5 mmHg relative to control"
NULL = "Training did not change resting heart rate in the same participants (p = 0.41)."


class ScriptedModel:
    """Implements the parts of research_pipeline.llm.Ollama the pipeline calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def resolve(self, model: str) -> tuple[str, str]:
        return model, "sha256:test"

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


def settings_for(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        verifier_model="",
        text_model="stub-model",
        max_rounds=2,
        # No gateway config: no companion or SearXNG instance is reached from a test.
        **{"gateway_config": tmp_path / "no-gateway.json", **overrides},
    )


@pytest.fixture
def run_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from research_pipeline import pipeline as module

    lib = papers_library(tmp_path)
    hold_w1(lib)
    lib.index_works()
    monkeypatch.setattr(module, "PaperLibrary", lambda *_: LibraryAdapter(lib))
    pipeline = Pipeline(settings_for(tmp_path), offline=True)
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


def test_publication_mode_refuses_self_verification(tmp_path: Path) -> None:
    pipeline = Pipeline(settings_for(tmp_path), offline=True, strictness="publication")
    pipeline.llm = ScriptedModel()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="publication strictness requires"):
        pipeline._resolve_models(State(QUESTION))


def test_publication_mode_accepts_a_different_verifier(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    settings.verifier_model = "independent-model"
    pipeline = Pipeline(
        settings,
        offline=True,
        strictness="publication",
    )
    pipeline.llm = ScriptedModel()  # type: ignore[assignment]
    assert pipeline._resolve_models(State(QUESTION)) == ["independent-model"]
    assert pipeline.models["independent_verification"] is True


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


class SpecialistModel(ScriptedModel):
    """Writes a view for each specialist; the cardiology one also overreaches once."""

    def specialist_view(self, user: str) -> dict[str, Any]:
        ids = re.findall(r"^\[(S\d+-C\d+)\]", user, flags=re.M)[:1]
        rows = [{"text": "Training lowered systolic pressure by 5 mmHg.", "claim_ids": ids}]
        if sum(c["task"] == "specialist_view" for c in self.calls) == 1:
            rows.append({"text": "Training cures heart failure.", "claim_ids": ids})
        return {"sentences": rows}

    def verify(self, _: str) -> dict[str, Any]:
        bad = "cures heart failure" in _
        return {
            "closest_sentence": "",
            "claim_adds": "a cure" if bad else "",
            "verdict": "NOT_SUPPORTED" if bad else "SUPPORTED",
        }


def test_each_specialist_writes_a_checked_paragraph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research_pipeline import pipeline as module

    lib = papers_library(tmp_path)
    hold_w1(lib)
    lib.index_works()
    monkeypatch.setattr(module, "PaperLibrary", lambda *_: LibraryAdapter(lib))
    route = {
        "specialist": "cardiology",
        "specialists": ["cardiology", "exercise-physiology"],
        "guidances": {"cardiology": "Name the pressure.", "exercise-physiology": "Name the dose."},
        "guidance": "[cardiology] Name the pressure.\n\n[exercise-physiology] Name the dose.",
    }
    pipeline = Pipeline(settings_for(tmp_path), offline=True, agent_route=route)
    pipeline.llm = SpecialistModel()  # type: ignore[assignment]
    log = asyncio.run(pipeline.run(QUESTION))
    views = log["specialist_views"]
    assert list(views) == ["cardiology", "exercise-physiology"]
    assert [r["kept"] for r in views["cardiology"]] == [True, False]
    answer = log["answer"]
    section = answer.split("## Specialist views")[1].split("\n## ")[0]
    assert "**cardiology.** Training lowered systolic pressure by 5 mmHg. [1]" in section
    assert "1 sentence(s) dropped" in section
    assert "cures heart failure" not in answer.split("## Limits")[0]
    assert "**exercise-physiology.**" in section


def test_a_run_with_no_specialist_has_no_views_section(run_log: dict[str, Any]) -> None:
    assert run_log["specialist_views"] == {}
    assert "## Specialist views" not in run_log["answer"]


class FakeLibrary(LibraryAdapter):
    """A real library whose search is scripted: one provider never answers."""

    def __init__(self, lib: Any) -> None:
        super().__init__(lib)
        self.fetched: list[str] = []

    async def search(
        self,
        query: str,
        limit: int = 10,
        include_closed: bool = True,
        profile: str = "",
        collection: str = "",
    ) -> dict[str, Any]:
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

    async def fetch(self, identifier: str, collection: str = "") -> dict[str, Any]:
        self.fetched.append(identifier)
        if identifier == "10.1000/closed":
            return {"work": "W2", "full_text": False}
        # The copy carries four hidden characters; the library strips and counts them.
        hold_w1(self.lib, PAPER.replace("Resting", "Resting\u200b\u200b\u200b\u200b", 1))
        return self.lib.fetch(identifier, collection=collection or None)


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

    monkeypatch.delenv("SEARXNG_URL", raising=False)
    lib = papers_library(tmp_path)
    monkeypatch.setattr(module, "PaperLibrary", lambda *_: FakeLibrary(lib))
    monkeypatch.setattr(
        module.integrity, "check_all", lambda dois: {d: {"status": "retracted"} for d in dois}
    )
    pipeline = Pipeline(settings_for(tmp_path, max_fetch=5), offline=False)
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

    async def search(
        self,
        query: str,
        limit: int = 10,
        include_closed: bool = True,
        profile: str = "",
        collection: str = "",
    ) -> dict[str, Any]:
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

    pipeline = Pipeline(settings_for(tmp_path), offline=False)
    st = State(QUESTION)
    library = ComponentLibrary(papers_library(tmp_path))
    asyncio.run(pipeline.discover(st, library, ["precision"]))  # type: ignore[arg-type]
    assert list(st.candidates) == ["10.1371/journal.pcbi.1010490"]
    cand = st.candidates["10.1371/journal.pcbi.1010490"]
    assert cand.title == "In the Body's Eye"
    assert cand.work is None
    assert cand.ids == {"doi": "10.1371/journal.pcbi.1010490"}
    assert any("component" in note for note in st.notes)


class PlanRecorder(CriticalModel):
    """Keeps the planner's input, to see what the field's profile added to it."""

    plan_input = ""

    def chat_json(
        self, task: str, system: str, user: str, schema: dict[str, Any], **kw: Any
    ) -> dict[str, Any]:
        if task == "plan":
            PlanRecorder.plan_input = system + "\n" + user
        return super().chat_json(task, system, user, schema, **kw)


class RecordingLibrary(FakeLibrary):
    """Records what the pipeline tells the library: the field's profile and the collection."""

    seen: list[tuple[str, str, str]] = []  # noqa: RUF012 - reset per test

    async def search(
        self,
        query: str,
        limit: int = 10,
        include_closed: bool = True,
        profile: str = "",
        collection: str = "",
    ) -> dict[str, Any]:
        self.seen.append(("search", profile, collection))
        return await super().search(query, limit, include_closed, profile, collection)

    async def fetch(self, identifier: str, collection: str = "") -> dict[str, Any]:
        self.seen.append(("fetch", "", collection))
        return await super().fetch(identifier, collection)

    async def retrieve(self, queries: list[str], **kw: Any) -> dict[str, Any]:
        self.seen.append(("retrieve", "", kw.get("collection", "")))
        return await super().retrieve(queries, **kw)


def test_the_field_and_the_collection_reach_the_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research_pipeline import domains
    from research_pipeline import pipeline as module

    lib = papers_library(tmp_path)
    lib.create_collection("review")
    RecordingLibrary.seen = []
    monkeypatch.setattr(module, "PaperLibrary", lambda *_: RecordingLibrary(lib))
    monkeypatch.setattr(module.integrity, "check_all", lambda dois: {})
    policy = domains.derive(
        domains.GENERIC, slug="cardiology", label="Cardiology", profile="cardiovascular"
    )
    domain = domains.Domain("rag-domain-cardiology", "tests", "0.1.0", ">=0.1", "test", policy)
    pipeline = Pipeline(
        settings_for(tmp_path, max_fetch=5), offline=False, domain=domain, collection="review"
    )
    pipeline.llm = PlanRecorder()  # type: ignore[assignment]
    log = asyncio.run(pipeline.run(QUESTION))

    # The planner saw the library's profile: the field's indexed terms and its search advice.
    assert "Terms this field is indexed under:" in PlanRecorder.plan_input
    assert "heart rate variability" in PlanRecorder.plan_input
    assert "MeSH term" in PlanRecorder.plan_input
    assert log["domain"]["profile"]["slug"] == "cardiovascular"
    # Every search named the profile and the collection; fetches and retrieval used the collection.
    kinds = {kind for kind, _, _ in RecordingLibrary.seen}
    assert kinds == {"search", "fetch", "retrieve"}
    assert all(c == "review" for _, _, c in RecordingLibrary.seen)
    assert all(p == "cardiovascular" for k, p, _ in RecordingLibrary.seen if k == "search")
    assert log["collection"] == "review"
    # The fetched paper was listed in the collection, and its passages were read from it.
    assert [m["work"] for m in lib.collection("review")["members"]] == ["W1"]
    assert log["evidence"]


def test_a_profile_the_library_lacks_is_noted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research_pipeline import domains
    from research_pipeline import pipeline as module

    lib = papers_library(tmp_path)
    hold_w1(lib)
    lib.index_works()
    monkeypatch.setattr(module, "PaperLibrary", lambda *_: LibraryAdapter(lib))
    policy = domains.derive(domains.GENERIC, slug="nowhere", label="Nowhere", profile="nowhere")
    domain = domains.Domain("rag-domain-nowhere", "tests", "0.1.0", ">=0.1", "test", policy)
    pipeline = Pipeline(settings_for(tmp_path), offline=True, domain=domain)
    pipeline.llm = ScriptedModel()  # type: ignore[assignment]
    log = asyncio.run(pipeline.run(QUESTION))
    assert any("no 'nowhere' profile" in note for note in log["notes"])
    assert log["domain"]["profile"] is None


class Unreadable(ScriptedModel):
    """Gives no usable answer for the null-result passage, and none when classifying the paper."""

    def chat_json(
        self, task: str, system: str, user: str, schema: dict[str, Any], **kw: Any
    ) -> dict[str, Any]:
        from research_pipeline.llm import ModelOutputError

        if task == "classify_paper" or (task == "extract" and "did not change" in user):
            raise ModelOutputError(f"{task}: cut off at the 2048-token cap")
        return super().chat_json(task, system, user, schema, **kw)


def test_one_call_with_no_answer_costs_that_item_not_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research_pipeline import pipeline as module

    lib = papers_library(tmp_path)
    hold_w1(lib)
    lib.index_works()
    monkeypatch.setattr(module, "PaperLibrary", lambda *_: LibraryAdapter(lib))
    pipeline = Pipeline(settings_for(tmp_path), offline=True)
    pipeline.llm = Unreadable()  # type: ignore[assignment]
    log = asyncio.run(pipeline.run(QUESTION))
    # The passage the model could not read was dropped and counted; the others became evidence.
    assert {e["direction"] for e in log["evidence"]} == {"affirms"}
    assert any("no_answer_from_model" in d["flags"] for d in log["dropped_passages"])
    assert "the model gave no usable answer when reading them" in log["answer"]
    # The paper could not be classified: graded unclear this run, and not cached as such.
    assert {e["study_type"] for e in log["evidence"]} == {"unclear"}
    assert any("no study-design answer" in n for n in log["notes"])
    assert pipeline.notes.get("W1") is None


def test_open_access_candidates_are_fetched_before_closed_ones(tmp_path: Path) -> None:
    """A closed paper with the more relevant title must not take the only fetch."""
    from research_pipeline.pipeline import State
    from research_pipeline.schema import Candidate, Plan, SubQuestion

    class Ordered(FakeLibrary):
        async def relevance(self, targets: list[str], texts: list[str]) -> list[float]:
            return [0.9 if "closed" in t.lower() else 0.2 for t in texts]

    library = Ordered(papers_library(tmp_path))
    pipeline = Pipeline(settings_for(tmp_path, max_fetch=1), offline=False)
    st = State(QUESTION)
    st.plan = Plan(QUESTION, "m", QUESTION, [SubQuestion("S1", QUESTION, "evidence", [])])
    st.candidates = {
        "10.1000/closed": Candidate(
            "10.1000/closed",
            "A closed-access trial",
            2021,
            {"doi": "10.1000/closed"},
            [],
            False,
            [],
        ),
        "10.1000/w1": Candidate(
            "10.1000/w1", "Training and blood pressure", 2020, {"doi": "10.1000/w1"}, [], True, []
        ),
    }
    asyncio.run(pipeline.acquire(st, library))  # type: ignore[arg-type]
    assert library.fetched == ["10.1000/w1"]
    assert st.candidates["10.1000/w1"].outcome == "indexed"
    assert st.candidates["10.1000/closed"].outcome == "over_budget"


class WebLibrary(FakeLibrary):
    """Reads paper identifiers out of web results the way paper-fetch's `leads` does."""

    async def leads(self, found: list[str]) -> dict[str, Any]:
        out = []
        for item in found:
            if "10.1000/web1" in item:
                out.append({"from": item[:300], "ids": {"doi": "10.1000/web1"}, "work": None})
            elif "10.1000/w1" in item:
                out.append({"from": item[:300], "ids": {"doi": "10.1000/w1"}, "work": None})
        return {"leads": out, "no_identifier": []}


def test_the_web_search_adds_linked_papers_as_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research_pipeline import pipeline as module
    from research_pipeline.pipeline import State

    asked: list[str] = []

    async def search(
        base: str, query: str, limit: int = 8, *, categories: str = "general", **_: Any
    ) -> list[dict[str, str]]:
        asked.append(f"{categories}: {query}")
        return [
            {"title": "A web-only paper", "url": "https://doi.org/10.1000/web1", "snippet": ""},
            {"title": "Training (repost)", "url": "https://x.org/10.1000/w1", "snippet": ""},
            {"title": "A blog post", "url": "https://blog.example/post", "snippet": "no id"},
        ]

    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")
    monkeypatch.setattr(module.web, "search", search)
    pipeline = Pipeline(settings_for(tmp_path, web_queries=2), offline=False)
    st = State(QUESTION)
    library = WebLibrary(papers_library(tmp_path))
    asyncio.run(pipeline.discover(st, library, ["blood pressure"]))  # type: ignore[arg-type]
    asyncio.run(
        pipeline.web_discover(st, library, [QUESTION, "blood pressure", "third"])  # type: ignore[arg-type]
    )
    # Bounded by web_queries; each query is asked of the science engines and the general web.
    assert sorted(asked) == sorted(
        f"{c}: {q}" for c in ("science", "general") for q in (QUESTION, "blood pressure")
    )
    assert st.candidates["10.1000/web1"].providers == ["web"]
    assert st.candidates["10.1000/web1"].title == "A web-only paper"
    assert "web" in st.candidates["10.1000/w1"].providers  # already found, now also by the web
    assert [w["new_candidates"] for w in st.web] == [1, 0]
    assert st.web[0]["results"][2]["url"] == "https://blog.example/post"


def test_a_run_without_searxng_says_the_web_was_not_searched(
    online_log: dict[str, Any],
) -> None:
    assert online_log["web"] == [{"query": "", "error": "not configured"}]
    assert "The web was not searched" in online_log["answer"]


class ContraryModel(ScriptedModel):
    """A panel model that reads the effect passage the other way and writes its own claim."""

    def extract(self, user: str) -> dict[str, Any]:
        got = super().extract(user)
        if got["direction"] == "affirms":
            got = got | {"direction": "denies", "finding": "No reliable change in pressure."}
        return got

    def synthesise(self, user: str) -> dict[str, Any]:
        ids = re.findall(r"^\[(S\d+-E\d+)\]", user, flags=re.M)
        if not ids:
            return {"claims": [], "insufficient": True}
        text = "Aerobic training did not reliably change resting systolic pressure."
        return {"insufficient": False, "claims": [{"text": text, "evidence_ids": ids[:1]}]}


@pytest.fixture
def panel_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from research_pipeline import pipeline as module
    from research_pipeline.panel import Member

    lib = papers_library(tmp_path)
    hold_w1(lib)
    lib.index_works()
    monkeypatch.setattr(module, "PaperLibrary", lambda *_: LibraryAdapter(lib))
    pipeline = Pipeline(settings_for(tmp_path), offline=True)
    pipeline.llm = ScriptedModel()  # type: ignore[assignment]
    pipeline.panel = [
        Member("same", "Same model", "m1", "u", llm=ScriptedModel()),  # type: ignore[arg-type]
        Member(
            "contrary",
            "Contrary model",
            "m2",
            "u",
            retention="keeps requests 30 days.",
            llm=ContraryModel(),  # type: ignore[arg-type]
        ),
        Member("absent", "Absent model", "m3", "u", skipped="no API key in x"),
    ]
    return asyncio.run(pipeline.run(QUESTION))


def test_panel_models_read_the_same_passages(panel_log: dict[str, Any]) -> None:
    totals = panel_log["panel"]["comparison"]["totals"]
    assert set(totals) == {"run", "same", "contrary"}
    # Every model read exactly the passages the run's model read.
    assert totals["same"]["passages_read"] == totals["run"]["passages_read"]
    assert totals["contrary"]["passages_read"] == totals["run"]["passages_read"]


def test_the_comparison_shows_agreement_and_opposite_readings(panel_log: dict[str, Any]) -> None:
    comparison = panel_log["panel"]["comparison"]
    run_claims = comparison["claims"]["run"]
    assert run_claims
    assert all(row["also_reached_by"] == ["same"] for row in run_claims)
    assert comparison["claims"]["contrary"][0]["also_reached_by"] == []
    opposite = comparison["opposite_readings"]
    assert len(opposite) == 1
    assert opposite[0]["readings"]["run"] == "affirms"
    assert opposite[0]["readings"]["contrary"] == "denies"
    answer = panel_log["answer"]
    assert "## Model comparison" in answer
    assert "Passages read in opposite directions" in answer
    assert "(also reached by Same model)" in answer


def test_panel_limits_name_retention_and_absent_members(panel_log: dict[str, Any]) -> None:
    answer = panel_log["answer"]
    assert "Contrary model read this run's passages; keeps requests 30 days." in answer
    assert "Absent model did not take part: no API key in x" in answer
    members = {m["name"]: m for m in panel_log["panel"]["members"]}
    assert members["absent"]["skipped"] == "no API key in x"
    assert members["contrary"]["usage"]["calls"] > 0


def test_a_failing_panel_member_never_ends_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research_pipeline import pipeline as module
    from research_pipeline.panel import Member

    class Broken(ScriptedModel):
        def extract(self, user: str) -> dict[str, Any]:
            raise RuntimeError("HTTP 401")

    lib = papers_library(tmp_path)
    hold_w1(lib)
    lib.index_works()
    monkeypatch.setattr(module, "PaperLibrary", lambda *_: LibraryAdapter(lib))
    pipeline = Pipeline(settings_for(tmp_path), offline=True)
    pipeline.llm = ScriptedModel()  # type: ignore[assignment]
    pipeline.panel = [Member("broken", "Broken model", "m", "u", llm=Broken())]  # type: ignore[arg-type]
    log = asyncio.run(pipeline.run(QUESTION))
    assert log["claims"]
    assert "Broken model did not take part: RuntimeError: HTTP 401" in log["answer"]


def test_no_panel_means_no_comparison(run_log: dict[str, Any]) -> None:
    assert run_log["panel"] is None
    assert "Model comparison" not in run_log["answer"]


def test_panel_members_come_from_the_hub_config(tmp_path: Path) -> None:
    from research_pipeline.panel import members

    key = tmp_path / "k.key"
    key.write_text("sk-test\n")
    config = {
        "providers": {
            "a": {
                "kind": "openai",
                "url": "https://api.example/v1",
                "model": "m",
                "key_file": str(key),
                "label": "A",
            },
            "b": {
                "kind": "openai",
                "url": "https://api.example/v1",
                "model": "m",
                "key_file": str(tmp_path / "missing.key"),
            },
            "c": {"kind": "ollama", "model": "x"},
        }
    }
    got = {m.name: m for m in members(settings_for(tmp_path, panel="a, b,c,d"), config)}
    assert got["a"].llm is not None
    assert got["a"].llm.name == "m"
    assert got["b"].skipped.startswith("no API key")
    assert "OpenAI-compatible" in got["c"].skipped
    assert got["d"].skipped == "no such provider in the hub config"
