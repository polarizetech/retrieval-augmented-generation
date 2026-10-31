"""Reply layout: the short lists and the full detail, built in code."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_mcp.ask import relay
from research_mcp.routing import Route
from research_pipeline import replies


def test_the_short_reply_leaves_out_an_empty_list() -> None:
    text = replies.short(["Found 2 passages.", " "], ["Answer from them."], [])
    assert text == "**Results**\n- Found 2 passages.\n\n**Next**\n- Answer from them."
    assert "**For you**\n- Pick one." in replies.short(["x"], ["y"], ["Pick one."])


def test_no_template_file_is_read() -> None:
    assert not (Path(replies.__file__).parents[1] / "config" / "replies.default.md").exists()
    assert not hasattr(replies, "render")


def _items(text: str, heading: str) -> list[str]:
    """The `- ` lines under a bold heading in a short reply."""
    part = text.split(f"**{heading}**\n", 1)
    if len(part) == 1:
        return []
    return [ln[2:] for ln in part[1].split("\n\n", 1)[0].splitlines() if ln.startswith("- ")]


def test_the_short_relay_is_three_lists_of_one_sentence_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = {
        "results": [
            {"text": "Pink noise. " * 80, "title": "Noise", "year": 2020, "evidence_id": "W1#p0"},
            {"text": "More.", "title": "Noise", "year": 2020, "evidence_id": "W1#p1"},
        ],
        "excluded": [{"id": "W2#p0"}],
    }
    args = (
        Route("quick", "r", "q"),
        {"name": "acoustics"},
        ["Retrieved 2"],
        {"evidence": found},
        1.0,
    )
    short = relay(*args, request_id="r1")
    full = relay(*args, request_id="r1", full=True)
    assert short.startswith("**Results**\n- Found 2 passage(s) from 1 paper(s)")
    assert _items(short, "Next") == [
        "The assistant answers from these passages and cites their evidence ids."
    ]
    assert "**For you**" not in short  # nothing for the person to do
    for item in _items(short, "Results") + _items(short, "Next"):
        assert ". " not in item, item
        assert len(item) < 140, item
    for detail in ("Pink noise", "`W1#p0`", "Request `r1`"):
        assert detail not in short
    for detail in ("“Pink noise.", "**Limits:**", "Request `r1`"):
        assert detail in full


def test_a_decision_lands_in_for_you() -> None:
    from research_mcp.hub.contract import Problem, decision_options

    p = Problem(
        "needs_decision", "embeddings", "The index did not answer. It may be restarting.",
        options=decision_options("embeddings", 30), recommendation="Use the fallback. It is fine.",
    )  # fmt: skip
    text = relay(Route("quick", "r", "q"), {}, [], {}, 1.0, problems=[p])
    you = _items(text, "For you")
    assert len(you) == 1
    assert you[0].startswith("Choose `fallback` or `stop` for")
    assert "(recommended: Use the fallback)" in you[0]
    assert _items(text, "Next") == ["Waiting for your choice."]
    assert _items(text, "Results")[0].startswith("Stopped: ")
    assert _items(text, "Results")[-1] == "Nothing was concluded."


def test_a_finished_answer_is_short_until_asked() -> None:
    from research_pipeline.mcp_server import _result

    log = {
        "question": "Q?",
        "bottom_line": "**One.** Two is (here). Three. Four.",
        "claims": [
            {"verdict": "SUPPORTED", "supported_by": ["W1#p0"]},
            {"verdict": "REJECTED", "supported_by": ["W2#p0"]},
        ],
        "evidence": [{"id": "W1#p0", "work": "W1"}, {"id": "W2#p0", "work": "W2"}],
        "answer": "# Q?\n\n## Bottom line\n\nlong", "run_dir": "/r", "seconds": 1,
        "searches": [],
    }  # fmt: skip
    out = _result(log)
    assert out["answer"].startswith("# Q?")
    assert _items(out["relay"], "Results") == [
        "One.", "Two is (here).", "Three.", "The bottom line goes on in the full answer.",
        "1 of 2 claim(s) passed verification, from 1 source(s).",
        "Findings, sources and limits are in the full answer.",
    ]  # fmt: skip
    assert "only when the person asks" in out["relay_instructions"]
