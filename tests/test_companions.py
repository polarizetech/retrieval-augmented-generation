"""Companion tools: requested by the plan, run by code, printed from the tool's own records."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from research_pipeline import companions
from research_pipeline.pipeline import Pipeline
from research_pipeline.prompts import Prompts
from research_pipeline.upstream import UpstreamError
from tests.conftest import LibraryAdapter, hold_w1, papers_library
from tests.test_pipeline import QUESTION, ScriptedModel, settings_for

DATA_CARD = {
    "ref": "physionet/bp-exercise@1.0.0",
    "title": "Blood pressure before and after aerobic training",
    "license": "ODC-By-1.0",
    "doi": "doi:10.13026/abc",
    "creators": ["A One", "B Two", "C Three", "D Four"],
    "total_bytes": 2_500_000,
    "files": {"n_files": 3, "extensions": {".csv": 2, ".hea": 1}, "data_files": True},
    "data_files": True,
    "kind": "database",
    "n_subjects": 48,
    "modalities": ["blood pressure"],
    "provider": "physionet",
    "score": 0.8,
    "score_reason": "matched 4 of 5 topic terms",
}
UNVERIFIED = {
    "ref": "osf/xyz@2024",
    "title": "Exercise physiology study",
    "license": None,
    "doi": None,
    "creators": [],
    "files": {"n_files": 0, "extensions": None},
    "data_files": None,
    "provider": "osf",
    "score": 0.9,
}


class FakeDatasets:
    """dataset-fetch's `recommend` contract, as returned for each topic."""

    def __init__(self, *, down: tuple[str, ...] = ()) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.down = down

    async def __aenter__(self) -> FakeDatasets:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def call(self, tool: str, **kw: Any) -> Any:
        self.calls.append((tool, kw))
        assert tool == "recommend"
        statuses = {
            "physionet": {"status": "ok", "why": None},
            "osf": {"status": "ok", "why": None},
            "gfz": {"status": "skipped", "why": "outside its scope"},
        }
        for name in self.down:
            statuses[name] = {"status": "unavailable", "why": "no answer in 20 s"}
        lower = {**DATA_CARD, "score": 0.4}  # the same record, found by the second query
        return {
            "results": [UNVERIFIED, DATA_CARD] if "first" in kw["topic"] else [lower],
            "documents_only": [{"ref": "zenodo/1@2022", "title": "A PDF", "why": "documents"}],
            "more_results": 2,
            "providers": statuses,
        }


async def rank(texts: list[str]) -> list[float]:
    raise AssertionError("dataset-fetch ranks; the companion never asks the library to")


def test_one_recommend_call_per_query_merged_and_listed_from_its_cards() -> None:
    up = FakeDatasets(down=("zenodo",))
    result = asyncio.run(companions.find_datasets(up, ["first topic", "second topic"], rank))  # type: ignore[arg-type]
    assert [(t, kw["topic"]) for t, kw in up.calls] == [
        ("recommend", "first topic"),
        ("recommend", "second topic"),
    ]
    # Verified data first, although the unverified card scored higher; the duplicate card
    # keeps its better score.
    assert [i["ref"] for i in result["items"]] == ["physionet/bp-exercise@1.0.0", "osf/xyz@2024"]
    assert result["items"][0]["score"] == 0.8
    assert result["asked"] == ["osf", "physionet", "zenodo"]
    assert result["documents_only"] == 1
    assert result["more"] == 4
    assert result["limits"] == [
        "Dataset catalogues that did not answer: zenodo. Their silence is not an absence of "
        "datasets."
    ]
    lines = "\n".join(companions.render_datasets(result))
    assert "**Blood pressure before and after aerobic training** — physionet (database)" in lines
    assert "ODC-By-1.0; 3 file(s) (.csv ×2, .hea ×1), 2.5 MB; 48 subjects" in lines
    assert "A One, B Two, C Three et al.; doi:10.13026/abc." in lines  # no doubled "doi:"
    assert "*Why listed: matched 4 of 5 topic terms.*" in lines
    assert "no licence stated, which is not permission to reuse" in lines
    assert "files not listed by the catalogue" in lines
    assert "Whether it holds data files could not be verified" in lines
    assert "Not listed: 4 further record(s) ranked below these; 1 record(s) hold only" in lines
    assert "Nothing was downloaded" in lines


def test_a_catalogue_that_answers_one_query_is_not_reported_silent() -> None:
    class Flaky(FakeDatasets):
        async def call(self, tool: str, **kw: Any) -> Any:
            self.down = ("osf",) if "first" in kw["topic"] else ()
            return await super().call(tool, **kw)

    result = asyncio.run(companions.find_datasets(Flaky(), ["first", "second"], rank))  # type: ignore[arg-type]
    assert result["limits"] == []


def test_nothing_found_is_said_plainly() -> None:
    class Empty(FakeDatasets):
        async def call(self, tool: str, **kw: Any) -> Any:
            return {"results": [], "documents_only": [], "more_results": 0, "providers": {}}

    result = asyncio.run(companions.find_datasets(Empty(), ["x"], rank))  # type: ignore[arg-type]
    assert result["items"] == []
    assert "- No dataset record with data files was found." in companions.render_datasets(result)


def test_only_configured_companions_are_offered(tmp_path: Path) -> None:
    config = tmp_path / "gateway.json"
    config.write_text(json.dumps({"upstreams": {}}))
    assert companions.configured(config) == {}
    assert companions.configured(tmp_path / "missing.json") == {}
    config.write_text(json.dumps({"companions": {"datasets": {"command": "x", "enabled": False}}}))
    assert companions.configured(config) == {}
    config.write_text(json.dumps({"companions": {"datasets": {"command": "x"}, "other": {}}}))
    assert list(companions.configured(config)) == ["datasets"]


def test_the_plan_offers_tools_only_when_some_are_configured() -> None:
    plain = Prompts()
    assert "tools" not in plain.plan_schema(4, 2)["properties"]
    assert "Other tools" not in plain.plan_system
    offered = Prompts(companions={"datasets": companions.DATASETS})
    tools = offered.plan_schema(4, 2)["properties"]["tools"]
    assert tools["items"]["properties"]["tool"]["enum"] == ["datasets"]
    assert "tools" in offered.plan_schema(4, 2)["required"]
    assert "most questions need none" in offered.plan_system
    assert offered.version != plain.version


class AsksForDatasets(ScriptedModel):
    def plan(self, _: str) -> dict[str, Any]:
        return super().plan(_) | {
            "tools": [
                {"tool": "datasets", "queries": ["blood pressure training"], "why": "measured"},
                {"tool": "datasets", "queries": ["asked twice"], "why": "dup"},
                {"tool": "telescope", "queries": ["not a tool"], "why": "invented"},
            ]
        }


def run_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upstream: Any, *, offline: bool = False
) -> dict[str, Any]:
    from research_pipeline import pipeline as module

    lib = papers_library(tmp_path)
    hold_w1(lib)
    lib.index_works()
    monkeypatch.setattr(module, "PaperLibrary", lambda *_: LibraryAdapter(lib))
    monkeypatch.setattr(module.integrity, "check_all", lambda dois: {})
    monkeypatch.setattr(companions, "configured", lambda _: {"datasets": companions.DATASETS})
    monkeypatch.setattr(companions, "connect", lambda *_: upstream)
    pipeline = Pipeline(settings_for(tmp_path, max_fetch=0), offline=offline)
    pipeline.llm = AsksForDatasets()  # type: ignore[assignment]
    return asyncio.run(pipeline.run(QUESTION))


def test_a_requested_companion_adds_its_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    up = FakeDatasets()
    log = run_with(tmp_path, monkeypatch, up)
    # Asked once, with the first request's queries; the invented tool was dropped.
    assert log["companions"]["requested"] == [
        {"tool": "datasets", "queries": ["blood pressure training"], "why": "measured"}
    ]
    answer = log["answer"]
    section = answer.split("## Datasets that could test this")[1].split("## Limits")[0]
    assert "Blood pressure before and after aerobic training" in section
    assert answer.index("## Datasets that could test this") < answer.index("## Limits")
    assert log["companions"]["results"]["datasets"]["documents_only"] == 1
    # A companion is a pointer for the reader: it never becomes evidence for a claim.
    assert all(e["work"] == "W1" for e in log["evidence"])


def test_a_companion_that_fails_is_reported_and_the_run_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken(FakeDatasets):
        async def call(self, tool: str, **kw: Any) -> Any:
            raise UpstreamError("providers: the server did not start", "unavailable")

    log = run_with(tmp_path, monkeypatch, Broken())
    assert "The datasets tool was asked and did not answer" in log["answer"]
    assert "That is not an absence of results." in log["answer"]
    assert log["claims"]


def test_an_offline_run_asks_no_companion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    up = FakeDatasets()
    log = run_with(tmp_path, monkeypatch, up, offline=True)
    assert up.calls == []
    assert log["companions"] == {"available": [], "requested": [], "results": {}}
    assert "## Datasets" not in log["answer"]


def test_one_refused_query_keeps_the_others_results() -> None:
    class Picky(FakeDatasets):
        async def call(self, tool: str, **kw: Any) -> Any:
            if kw["topic"] == "the and of":
                raise UpstreamError("the topic has no content words to search for", "tool_error")
            return await super().call(tool, **kw)

    result = asyncio.run(companions.find_datasets(Picky(), ["first topic", "the and of"], rank))  # type: ignore[arg-type]
    assert result["items"]
    assert result["limits"][0].startswith(
        "dataset-fetch did not answer for: “the and of” (tool_error"
    )
    with pytest.raises(UpstreamError, match="answered none of the queries"):
        asyncio.run(companions.find_datasets(Picky(), ["the and of"], rank))  # type: ignore[arg-type]
