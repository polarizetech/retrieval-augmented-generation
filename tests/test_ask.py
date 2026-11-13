import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from research_mcp.ask import Asker, parse_tool_text
from research_mcp.brief import compact, coverage

ROOT = Path(__file__).resolve().parents[1]


def brief(root: Path, name: str, words: str, guidance: str) -> None:
    (root / "agents").mkdir(parents=True, exist_ok=True)
    (root / "agents" / f"{name}.md").write_text(
        f"---\nname: {name}\ndescription: Use for {words}.\nextends: researcher\n---\n\n"
        f"Positive routing examples:\n\n1. {words}?\n\n## Pipeline guidance\n\n{guidance}\n"
    )


def asker(tmp_path: Path, calls: list[tuple[str, Any]], local: str = "", **hooks: Any) -> Asker:
    public, private = tmp_path / "public", tmp_path / "private"
    brief(public, "researcher", "literature", "")
    brief(private, "noise-engineering", "noise colour pink brown", "Name the domain.")
    (tmp_path / "routing.toml").write_text(local)

    async def retrieve(query: str, limit: int, max_chars: int) -> dict[str, Any]:
        calls.append(("retrieve", (query, limit)))
        return {
            "retrieval": "hybrid",
            "n": 1,
            "excluded": [],
            "results": [
                {
                    "evidence_id": "W1#p0.abcdef12",
                    "text": "Pink noise has equal energy per octave.",
                    "title": "Colours of noise",
                    "year": 2017,
                    "doi": "10.1/x",
                }
            ],
        }

    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append((name, arguments))
        return {"run_id": "r1", "status": "working", "model_backend": "endpoint"}

    return Asker(
        retrieve=retrieve,
        call_tool=call_tool,
        has_tool=lambda n: n == "pipeline__research_start",
        rules_paths=(ROOT / "config" / "routing.default.toml", tmp_path / "routing.toml"),
        profiles_dir=public,
        extensions_dir=private,
        **hooks,
    )


def test_a_question_runs_one_retrieval_and_nothing_else(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    out = asyncio.run(asker(tmp_path, calls).ask("What is pink noise?"))
    assert [c[0] for c in calls] == ["retrieve"]
    assert out["route"]["mode"] == "quick"
    assert out["specialist"]["name"] == "noise-engineering"
    assert set(out["results"]) == {"evidence", "fresh"}
    # With no scholarly or web search connected, the fresh search says so instead of failing.
    assert out["results"]["fresh"]["not_run"] == [
        "scholarly search: not connected",
        "web search: no SearXNG instance is configured",
    ]
    relay = out["relay_full"]
    for part in ("Route: quick", "`W1#p0.abcdef12`", "Name the domain.", "**Limits:**"):
        assert part in relay


def test_an_evidence_question_starts_the_pipeline_without_retrieving_twice(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    out = asyncio.run(
        asker(tmp_path, calls).ask(
            "What does the evidence say about pink noise?", strictness="explore"
        )
    )
    assert calls == [
        (
            "pipeline__research_start",
            {"question": "What does the evidence say about pink noise?", "strictness": "explore"},
        )
    ]
    assert "Research run** `r1`" in out["relay_full"]


def test_a_code_task_recruits_the_coder_with_context(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    out = asyncio.run(
        asker(tmp_path, calls).ask("Fix the bug in the pink noise generator", repo="polarizetech/x")
    )
    assert [c[0] for c in calls] == ["retrieve"]
    assert calls[0][1][1] == 3  # fewer passages: context, not an answer
    assert out["results"]["coder"]["runs_on"] == "client"
    assert "**Coder:**" in out["relay_full"]


def test_a_plain_code_task_skips_retrieval(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    out = asyncio.run(asker(tmp_path, calls).ask("Fix the failing test in tests/test_rag.py"))
    assert calls == []  # the library holds papers, not code: nothing to retrieve
    assert "Retrieval: skipped" in out["relay_full"]
    assert out["results"]["coder"]["runs_on"] == "client"


def test_my_own_words_pick_the_specialist(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    a = asker(tmp_path, calls, '[profiles]\nnoise-engineering = ["hiss"]\n')
    out = asyncio.run(a.ask("Where does tape hiss come from?"))
    assert out["specialist"]["name"] == "noise-engineering"
    assert out["specialist"]["why"] == "your word 'hiss'"
    assert out["specialist"]["guidance"] == "Name the domain."


def test_upstream_text_is_parsed_and_errors_raise() -> None:
    class Text:
        def __init__(self, text: str) -> None:
            self.text = text

    class Result:
        def __init__(self, text: str, error: bool = False) -> None:
            self.content, self.isError = [Text(text)], error

    assert parse_tool_text(Result('{"run_id": "r1"}')) == {"run_id": "r1"}
    with pytest.raises(ValueError, match="boom"):
        parse_tool_text(Result("boom", error=True))


def _floor_asker(tmp_path: Path, scores: list[float], reranker: str | None = "onnx:bge") -> Asker:
    async def retrieve(query: str, limit: int, max_chars: int) -> dict[str, Any]:
        rows = [
            {"evidence_id": f"W{i}#p0.abcdef12", "text": "t", "title": f"Paper {i}", "score": s}
            for i, s in enumerate(scores)
        ]
        return {"retrieval": "hybrid", "reranker": reranker, "n": len(rows), "excluded": []} | {
            "results": rows
        }

    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {}

    (tmp_path / "routing.toml").write_text("")
    return Asker(
        retrieve=retrieve,
        call_tool=call_tool,
        has_tool=lambda n: False,
        rules_paths=(ROOT / "config" / "routing.default.toml", tmp_path / "routing.toml"),
        profiles_dir=tmp_path / "none",
        extensions_dir=tmp_path / "none",
    )


def test_passages_under_the_relevance_floor_are_not_shown(tmp_path: Path) -> None:
    out = asyncio.run(_floor_asker(tmp_path, [-4.2, -4.3, -5.5]).ask("use the reg tool to audit"))
    found = out["results"]["evidence"]
    assert found["results"] == []
    assert found["below_floor"] == {"n": 3, "floor": -3.0, "best_score": -4.2}
    assert "No passage was relevant enough to show" in out["relay"]
    assert "Found" not in out["relay"]


def test_near_misses_from_a_whole_question_are_still_shown(tmp_path: Path) -> None:
    # Reranked against a long question, a relevant passage can score just under 0.
    out = asyncio.run(
        _floor_asker(tmp_path, [-0.03, -0.95, -2.9]).ask(
            "How do two tones mix nonlinearly in the cochlea to create a difference tone?"
        )
    )
    found = out["results"]["evidence"]
    assert [r["score"] for r in found["results"]] == [-0.03, -0.95, -2.9]
    assert "below_floor" not in found
    assert "No passage was relevant enough to show" not in out["relay"]


def test_the_floor_keeps_relevant_passages_and_says_what_it_left_out(tmp_path: Path) -> None:
    out = asyncio.run(
        _floor_asker(tmp_path, [5.1, 0.4, -4.2]).ask("What is vibrational resonance?")
    )
    found = out["results"]["evidence"]
    assert [r["score"] for r in found["results"]] == [5.1, 0.4]
    assert found["n"] == 2
    assert "1 weaker passage(s) under the relevance floor were left out." in out["relay"]


def test_without_a_reranker_scores_are_ranks_and_no_floor_applies(tmp_path: Path) -> None:
    out = asyncio.run(_floor_asker(tmp_path, [-0.1, -0.2], reranker=None).ask("What is noise?"))
    assert len(out["results"]["evidence"]["results"]) == 2
    assert "below_floor" not in out["results"]["evidence"]


# The compact brief the gateway sends instead of the full reply (research_mcp/brief.py).


def found(*scores: float, reranked: bool = True) -> dict[str, Any]:
    return {
        "query": "q",
        "reranker": "onnx:bge" if reranked else None,
        "results": [
            {"evidence_id": f"W{i}#p0", "text": "t", "score": s} for i, s in enumerate(scores)
        ],
    }


def test_coverage_levels_follow_the_best_reranker_score() -> None:
    assert coverage(found(4.6, 2.0))["level"] == "strong"  # type: ignore[index]
    assert coverage(found(0.4, -2.0))["level"] == "partial"  # type: ignore[index]
    weak = coverage(found(-1.3, -2.7))
    assert weak is not None
    assert weak["level"] == "weak"
    assert weak["best_score"] == -1.3
    assert "Say so before anything else" in weak["note"]
    assert coverage(
        {"results": [], "below_floor": {"n": 5, "floor": -3.0, "best_score": -4.2}}
    ) == {
        "level": "none",
        "note": "No passage in the library matched. Say so before answering any other way.",
        "left_out": "5 weaker passage(s) scored under the relevance floor (-3; best -4.20).",
    }
    assert coverage(found(1.0, reranked=False))["level"] == "unscored"  # type: ignore[index]
    assert coverage(None) is None


def test_a_quick_reply_becomes_one_brief_with_each_passage_once(tmp_path: Path) -> None:
    full = asyncio.run(asker(tmp_path, []).ask("What is pink noise?"))
    full["links"] = {"record": "http://x/records/c1"}
    out = compact(full)
    assert set(out) == {"request_id", "status", "mode", "brief", "answer_format", "links"}
    brief = out["brief"]
    assert brief["passages"] == [
        {
            "id": "W1#p0.abcdef12",
            "source": "Colours of noise, 2017, doi:10.1/x",
            "text": "Pink noise has equal energy per octave.",
        }
    ]
    assert brief["coverage"]["level"] == "unscored"  # the fake retrieval has no reranker
    assert brief["specialist"] == {"name": "noise-engineering", "guidance": "Name the domain."}
    assert "passages, not a checked answer" in brief["limits"]
    assert out["links"] == {"record": "http://x/records/c1"}
    # The passage text is sent once, not three times as in the full reply.
    assert json.dumps(out).count("equal energy per octave") == 1
    assert json.dumps(full).count("equal energy per octave") >= 2
    assert "Bottom line" in out["answer_format"]
    assert "evidence id" in out["answer_format"]


def test_a_research_reply_keeps_its_run_and_next_step(tmp_path: Path) -> None:
    full = asyncio.run(asker(tmp_path, []).ask("research: does pink noise improve sleep?"))
    out = compact(full)
    assert out["mode"] == "research"
    assert out["brief"]["tool_results"]["research"]["run_id"] == "r1"
    assert any("hub__result" in n for n in out["brief"]["next"])
    assert "passages" not in out["brief"]


def test_anything_that_is_not_a_full_ask_reply_passes_through() -> None:
    error = {"error": {"status": "refused"}}
    assert compact(error) is error


def test_several_specialists_each_get_guidance_and_a_place_in_the_brief(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    a = asker(tmp_path, calls)
    brief(tmp_path / "private", "signal-processing", "fourier spectrum noise", "Give the rate.")
    out = asyncio.run(a.ask("Ask signal-processing: what is the spectrum of pink noise?"))
    names = [s["name"] for s in out["specialist"]["specialists"]]
    assert names == ["signal-processing", "noise-engineering"]
    assert "Specialists: signal-processing, noise-engineering" in out["relay_full"]
    assert "**signal-processing guidance:** Give the rate." in out["relay_full"]
    assert "**noise-engineering guidance:** Name the domain." in out["relay_full"]
    small = compact(out)["brief"]
    assert small["specialists"] == [
        {"name": "signal-processing", "guidance": "Give the rate."},
        {"name": "noise-engineering", "guidance": "Name the domain."},
    ]
    assert "Specialist views" in compact(out)["answer_format"]


def test_asking_for_the_profiles_lists_them_without_retrieving(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    out = asyncio.run(asker(tmp_path, calls).ask("Which profiles are available?"))
    assert calls == []
    rows = out["results"]["profiles"]
    assert [r["name"] for r in rows] == ["noise-engineering"]
    assert "`noise-engineering` (private)" in out["relay_full"]
    assert "1 specialist profiles: noise-engineering." in out["relay"]
    assert "These are passages" not in out["relay"]
    assert "Nothing was retrieved." in out["relay"]
    assert "list of installed profiles" in compact(out)["brief"]["limits"]


def test_an_unknown_profile_in_my_words_is_reported(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    a = asker(tmp_path, calls, '[profiles]\nghost-profile = ["tape hiss"]\n')
    out = asyncio.run(a.ask("Where does tape hiss come from?"))
    assert out["specialist"]["unknown"] == ["ghost-profile"]
    assert "No profile is named ghost-profile" in out["relay"]


async def _search(query: str, limit: int) -> dict[str, Any]:
    return {
        "results": [
            {"title": "Octave-band noise", "year": 2021, "doi": "10.2/y", "is_oa": True},
        ],
        "not_answered": ["core"],
    }


async def _web(query: str, limit: int) -> list[dict[str, str]]:
    return [{"title": "Pink noise explained", "url": "https://example.org/pink", "snippet": "…"}]


def test_every_quick_question_also_searches_the_indexes_and_the_web(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    out = asyncio.run(asker(tmp_path, calls, search=_search, web=_web).ask("What is pink noise?"))
    fresh = out["results"]["fresh"]
    assert [p["doi"] for p in fresh["papers"]] == ["10.2/y"]
    assert [w["url"] for w in fresh["web"]] == ["https://example.org/pink"]
    assert fresh["not_run"] == ["scholarly provider core: did not answer"]
    assert any(
        p.startswith("Fresh search outside the library: 1 paper(s)") for p in out["progress"]
    )
    relay = out["relay_full"]
    assert "**Fresh search outside the library** (unread leads: not citable)" in relay
    assert "https://doi.org/10.2/y" in relay
    assert "https://example.org/pink" in relay
    assert any(
        "fresh search outside the library found 1 paper" in g for g in out["summary"]["results"]
    )
    sent = compact(out)["brief"]
    assert sent["fresh_search"]["papers"][0]["link"] == "https://doi.org/10.2/y"
    assert sent["fresh_search"]["web"][0]["url"] == "https://example.org/pink"
    assert "fresh" not in sent.get("tool_results", {})


def test_fresh_leads_are_offered_when_the_library_falls_short(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    a = asker(tmp_path, calls, search=_search, web=_web)

    async def nothing(query: str, limit: int, max_chars: int) -> dict[str, Any]:
        return {"retrieval": "hybrid", "n": 0, "excluded": [], "results": []}

    a.retrieve = nothing
    out = asyncio.run(a.ask("What is pink noise?"))
    assert any("closest fresh leads" in n for n in out["summary"]["next"])
    assert "Fresh leads" in compact(out)["answer_format"]


def test_a_failed_web_search_never_stops_the_answer(tmp_path: Path) -> None:
    async def broken(query: str, limit: int) -> list[dict[str, str]]:
        raise RuntimeError("SearXNG is down")

    calls: list[tuple[str, Any]] = []
    out = asyncio.run(asker(tmp_path, calls, search=_search, web=broken).ask("What is pink noise?"))
    assert out["status"] == "done"
    assert "web search: RuntimeError" in out["results"]["fresh"]["not_run"]
    assert out["results"]["evidence"]["results"]
