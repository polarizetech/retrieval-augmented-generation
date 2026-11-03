import asyncio
from pathlib import Path
from typing import Any

import pytest

from research_mcp.ask import Asker, parse_tool_text

ROOT = Path(__file__).resolve().parents[1]


def brief(root: Path, name: str, words: str, guidance: str) -> None:
    (root / "agents").mkdir(parents=True, exist_ok=True)
    (root / "agents" / f"{name}.md").write_text(
        f"---\nname: {name}\ndescription: Use for {words}.\nextends: researcher\n---\n\n"
        f"Positive routing examples:\n\n1. {words}?\n\n## Pipeline guidance\n\n{guidance}\n"
    )


def asker(tmp_path: Path, calls: list[tuple[str, Any]], local: str = "") -> Asker:
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
    )


def test_a_question_runs_one_retrieval_and_nothing_else(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    out = asyncio.run(asker(tmp_path, calls).ask("What is pink noise?"))
    assert [c[0] for c in calls] == ["retrieve"]
    assert out["route"]["mode"] == "quick"
    assert out["specialist"]["name"] == "noise-engineering"
    assert set(out["results"]) == {"evidence"}
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
    assert found["below_floor"] == {"n": 3, "floor": 0.0, "best_score": -4.2}
    assert "No passage was relevant enough to show" in out["relay"]
    assert "Found" not in out["relay"]


def test_the_floor_keeps_relevant_passages_and_says_what_it_left_out(tmp_path: Path) -> None:
    out = asyncio.run(
        _floor_asker(tmp_path, [5.1, 0.4, -1.2]).ask("What is vibrational resonance?")
    )
    found = out["results"]["evidence"]
    assert [r["score"] for r in found["results"]] == [5.1, 0.4]
    assert found["n"] == 2
    assert "1 weaker passage(s) under the relevance floor were left out." in out["relay"]


def test_without_a_reranker_scores_are_ranks_and_no_floor_applies(tmp_path: Path) -> None:
    out = asyncio.run(_floor_asker(tmp_path, [-0.1, -0.2], reranker=None).ask("What is noise?"))
    assert len(out["results"]["evidence"]["results"]) == 2
    assert "below_floor" not in out["results"]["evidence"]
