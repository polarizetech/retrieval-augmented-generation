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

CARDS = {
    "zenodo/1@2024": {
        "provider": "zenodo",
        "title": "Blood pressure recordings before and after aerobic training",
        "doi": "10.5281/zenodo.1",
        "license": "cc-by-4.0",
        "creators": ["A One", "B Two", "C Three", "D Four"],
        "total_bytes": 2_500_000,
        "files": {"n_files": 3, "extensions": {".csv": 2, ".pdf": 1}},
    },
    "zenodo/2@2023": {
        "provider": "zenodo",
        "title": "Aerobic training lowered blood pressure: a report",
        "doi": None,
        "license": None,
        "creators": [],
        "total_bytes": 170_263,
        "files": {"n_files": 1, "extensions": {".pdf": 1}},
    },
}


class FakeDatasets:
    """dataset-fetch's contract: providers, search (per provider), describe."""

    def __init__(self, *, down: tuple[str, ...] = ()) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.down = down

    async def __aenter__(self) -> FakeDatasets:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def call(self, tool: str, **kw: Any) -> Any:
        self.calls.append((tool, kw))
        if tool == "providers":
            return {
                "providers": [
                    {"name": "zenodo", "search_filters": ["q", "size"]},
                    {"name": "eegdash", "search_filters": ["modality", "query"]},
                    {"name": "gfz", "search_filters": []},  # no free-text search: never asked
                ]
            }
        if tool == "search":
            if kw["provider"] in self.down:
                raise UpstreamError("search: timed out", "unavailable")
            refs = list(CARDS) if kw["provider"] == "zenodo" else []
            return {"n": len(refs), "results": [{"ref": r} for r in refs]}
        assert tool == "describe"
        return {"ref": kw["ref"], **CARDS[kw["ref"]]}


async def rank(texts: list[str]) -> list[float]:
    return [0.9 if "report" in t else 0.5 for t in texts]


def test_datasets_are_found_across_catalogues_and_listed_from_their_records() -> None:
    up = FakeDatasets(down=("eegdash",))
    result = asyncio.run(companions.find_datasets(up, ["blood pressure training"], rank))  # type: ignore[arg-type]
    searched = [(kw["provider"], kw["filters"]) for tool, kw in up.calls if tool == "search"]
    # Each catalogue is asked with its own name for the free-text filter; gfz has none.
    assert searched == [
        ("zenodo", {"q": "(blood pressure training) AND resource_type.type:dataset"}),
        ("eegdash", {"query": "blood pressure training"}),
    ]
    # The record that holds only a PDF is counted, not listed, although it scored higher.
    assert [i["ref"] for i in result["items"]] == ["zenodo/1@2024"]
    assert result["documents_only"] == 1
    assert result["limits"] == [
        "Dataset catalogues that did not answer at least once: eegdash. "
        "Their silence is not an absence of datasets."
    ]
    lines = "\n".join(companions.render_datasets(result))
    assert "Blood pressure recordings before and after aerobic training" in lines
    assert "cc-by-4.0; 3 file(s) (.csv ×2, .pdf ×1), 2.5 MB; A One, B Two, C Three et al." in lines
    assert "doi:10.5281/zenodo.1. Ref `zenodo/1@2024`." in lines
    assert "1 further record(s) hold only documents" in lines
    assert "Nothing was downloaded" in lines


def test_nothing_found_is_said_plainly() -> None:
    class Empty(FakeDatasets):
        async def call(self, tool: str, **kw: Any) -> Any:
            if tool == "search":
                return {"n": 0, "results": []}
            return await super().call(tool, **kw)

    result = asyncio.run(companions.find_datasets(Empty(), ["x"], rank))  # type: ignore[arg-type]
    assert result["items"] == []
    assert "- No dataset record with data files was found by these searches." in (
        companions.render_datasets(result)
    )


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
    assert "Blood pressure recordings before and after aerobic training" in section
    assert answer.index("## Datasets that could test this") < answer.index("## Limits")
    assert log["companions"]["results"]["datasets"]["found"] == 2
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


def test_a_record_with_no_licence_or_no_file_list_says_so() -> None:
    result = {
        "queries": ["q"],
        "catalogues": ["zenodo"],
        "items": [
            {
                "ref": "zenodo/3@2021",
                "provider": "zenodo",
                "title": "A study dataset",
                "doi": None,
                "license": "unknown",
                "creators": [],
                "total_bytes": None,
                "n_files": 0,
                "extensions": {},
                "data_files": None,
            }
        ],
    }
    line = companions.render_datasets(result)[-1]
    assert "no licence stated, which is not permission to reuse" in line
    assert "files not listed by the catalogue (access may be restricted)" in line
