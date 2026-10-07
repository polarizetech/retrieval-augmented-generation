"""The patents companion: patent-fetch's command line, asked by code, printed from its records."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from research_pipeline import companions
from research_pipeline.pipeline import Pipeline
from research_pipeline.upstream import Command, UpstreamError
from tests.conftest import LibraryAdapter, hold_w1, papers_library
from tests.test_pipeline import QUESTION, ScriptedModel, settings_for


def hit(pid: str, title: str, year: int, **extra: Any) -> dict[str, Any]:
    return {
        "id": pid,
        "kind": "B2",
        "title": title,
        "abstract": "An abstract the companion never prints.",
        "year": year,
        "applicants": [],
        "inventors": ["ONE A", "TWO B"],
        "classifications": ["A61B5/021", "A61B5/022", "A61B5/00", "G16H20/30"],
        "status": None,
        "url": f"https://patents.example/{pid}",
        "found_by": ["europepmc"],
    } | extra


CUFF = hit("US7000001", "Blood pressure cuff for use during aerobic exercise", 2006)
BIKE = hit(
    "EP2000002", "Exercise bicycle frame", 2011, applicants=["Acme Cycles"], status="Expired"
)
KEYLESS = [
    {"provider": "europepmc", "status": "ok", "hits": 2},
    {"provider": "epo-ops", "status": "skipped", "hits": 0, "detail": "needs EPO_OPS_KEY"},
]


class FakePatents:
    """`patent-fetch --json search <query> -n N`, as the command answers it."""

    def __init__(self, providers: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.providers = KEYLESS if providers is None else providers

    async def __aenter__(self) -> FakePatents:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def run(self, *arguments: str, timeout: float = 120.0) -> Any:
        self.calls.append(arguments)
        assert arguments[0] == "search"
        if "refused" in arguments[1]:
            raise UpstreamError("patents: exit 2: error: bad query")
        found = [BIKE, CUFF] if "first" in arguments[1] else [CUFF]
        return {"query": arguments[1], "hits": found, "providers": self.providers, "broken": []}


async def by_title(texts: list[str]) -> list[float]:
    """The library's relevance, stood in for: closer when the title names blood pressure."""
    return [0.9 if "pressure" in t.lower() else 0.2 for t in texts]


def find(tool: FakePatents, queries: list[str]) -> dict[str, Any]:
    return asyncio.run(companions.find_patents(tool, queries, by_title))  # type: ignore[arg-type]


def test_hits_are_merged_by_number_and_ordered_by_closeness_to_the_question() -> None:
    tool = FakePatents()
    result = find(tool, ["first query", "second query"])
    assert tool.calls == [
        ("search", "first query", "-n", "8"),
        ("search", "second query", "-n", "8"),
    ]
    assert [h["id"] for h in result["items"]] == ["US7000001", "EP2000002"]  # closest title first
    assert result["items"][0]["asked_as"] == "first query"  # found by both; listed once
    assert "abstract" not in result["items"][0]
    text = "\n".join(companions.render_patents(result))
    assert "**Blood pressure cuff for use during aerobic exercise** — `US7000001`; 2006" in text
    assert "ONE A, TWO B" in text
    assert "classes A61B5/021, A61B5/022, A61B5/00." in text  # three classes, not all four
    assert "Acme Cycles" in text
    assert "status as the service reports it: Expired" in text
    assert "not a legal opinion" in text
    assert "finding none is not evidence that none exists" in text


def test_the_limits_say_what_was_and_was_not_searched() -> None:
    limits = " ".join(find(FakePatents(), ["first"])["limits"])
    assert "life-science patents only, and none after 2012" in limits
    assert "not asked (no account configured): epo-ops" in limits

    keyed = [{"provider": "epo-ops", "status": "ok"}, {"provider": "lens", "status": "unavailable"}]
    limits = " ".join(find(FakePatents(keyed), ["first"])["limits"])
    assert "2012" not in limits
    assert "did not answer: lens. Their silence is not an absence of patents." in limits

    nobody = [{"provider": "epo-ops", "status": "skipped"}]
    result = find(FakePatents(nobody), ["second"])
    assert "says nothing either way" in " ".join(result["limits"])


def test_no_record_is_said_plainly_and_one_refused_query_keeps_the_other() -> None:
    class Empty(FakePatents):
        async def run(self, *arguments: str, timeout: float = 120.0) -> Any:
            return {"hits": [], "providers": [{"provider": "epo-ops", "status": "ok"}]}

    assert "- No patent record was returned." in companions.render_patents(find(Empty(), ["q"]))
    result = find(FakePatents(), ["first", "refused"])
    assert result["items"]
    assert result["limits"][0].startswith("patent-fetch did not answer for: “refused”")
    with pytest.raises(UpstreamError, match="answered none of the queries"):
        find(FakePatents(), ["refused"])


# ---------------------------------------------------------------------------------- the command


def command(tmp_path: Path, program: str, **spec: Any) -> Command:
    config = tmp_path / "gateway.json"
    entry = {"command": sys.executable, "args": ["-c", program, "--json"], **spec}
    config.write_text(json.dumps({"companions": {"patents": entry}}))
    assert list(companions.configured(config)) == ["patents"]
    tool = companions.connect(config, "patents")
    assert isinstance(tool, Command)
    return tool


def test_a_command_is_run_with_an_argument_list_and_read_as_json(tmp_path: Path) -> None:
    echo = "import json, sys; print(json.dumps({'argv': sys.argv[1:]}))"
    tool = command(tmp_path, echo)
    # Whatever a model wrote as a query is one argument: no shell ever reads it.
    nasty = 'x"; rm -rf ~; echo $(whoami) `id` | cat'
    got = asyncio.run(tool.run("search", nasty, "-n", "3"))
    assert got == {"argv": ["--json", "search", nasty, "-n", "3"]}


@pytest.mark.parametrize(
    ("program", "match", "code"),
    [
        ("import sys; sys.stderr.write('boom\\nerror: bad query\\n'); sys.exit(2)",
         "exit 2: error: bad query", "tool_error"),
        ("print('not json')", "output is not JSON", "tool_error"),
        ("import time; time.sleep(30)", "no answer in 0 s", "unavailable"),
    ],
)  # fmt: skip
def test_a_command_that_fails_says_how(tmp_path: Path, program: str, match: str, code: str) -> None:
    tool = command(tmp_path, program)
    with pytest.raises(UpstreamError, match=match) as failed:
        asyncio.run(tool.run("search", "q", timeout=0.3 if "sleep" in program else 30))
    assert failed.value.code == code


def test_a_command_that_is_not_installed_is_unavailable(tmp_path: Path) -> None:
    config = tmp_path / "gateway.json"
    config.write_text(json.dumps({"companions": {"patents": {"command": "no-such-patent-tool"}}}))
    with pytest.raises(UpstreamError, match="cannot start") as failed:
        asyncio.run(companions.connect(config, "patents").run("search", "q"))  # type: ignore[union-attr]
    assert failed.value.code == "unavailable"


# ---------------------------------------------------------------------------------- in a run


class AsksForPatents(ScriptedModel):
    def plan(self, _: str) -> dict[str, Any]:
        return super().plan(_) | {
            "tools": [{"tool": "patents", "queries": ["first cuff query"], "why": "a device"}]
        }


def run_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: Any) -> dict[str, Any]:
    from research_pipeline import pipeline as module

    lib = papers_library(tmp_path)
    hold_w1(lib)
    lib.index_works()
    monkeypatch.setattr(module, "PaperLibrary", lambda *_: LibraryAdapter(lib))
    monkeypatch.setattr(module.integrity, "check_all", lambda dois: {})
    monkeypatch.setattr(companions, "configured", lambda _: {"patents": companions.PATENTS})
    monkeypatch.setattr(companions, "connect", lambda *_: tool)
    pipeline = Pipeline(settings_for(tmp_path, max_fetch=0), offline=False)
    pipeline.llm = AsksForPatents()  # type: ignore[assignment]
    return asyncio.run(pipeline.run(QUESTION))


def test_requested_patents_add_their_section_and_never_become_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = run_with(tmp_path, monkeypatch, FakePatents())
    answer = log["answer"]
    section = answer.split("## Existing patents on this")[1].split("## Limits")[0]
    assert "Blood pressure cuff for use during aerobic exercise" in section
    assert "none after 2012" in answer.split("## Limits")[1]
    # The real relevance ranked the titles: the library's, against the question.
    assert all("relevance" in h for h in log["companions"]["results"]["patents"]["items"])
    assert all(e["work"] == "W1" for e in log["evidence"])


def test_a_missing_patent_tool_is_reported_and_the_run_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Missing(FakePatents):
        async def run(self, *arguments: str, timeout: float = 120.0) -> Any:
            raise UpstreamError("patents: cannot start 'patent-fetch'", "unavailable")

    log = run_with(tmp_path, monkeypatch, Missing())
    assert "The patents tool was asked and did not answer" in log["answer"]
    assert log["claims"]


def test_the_planner_is_told_when_patents_serve_a_question() -> None:
    from research_pipeline.prompts import Prompts

    offered = Prompts(companions={"patents": companions.PATENTS, "datasets": companions.DATASETS})
    assert "`patents`: request it when the question is about a device" in offered.plan_system
    tools = offered.plan_schema(4, 2)["properties"]["tools"]
    assert tools["items"]["properties"]["tool"]["enum"] == ["datasets", "patents"]
