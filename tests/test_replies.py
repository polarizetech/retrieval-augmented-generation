"""Reply templates: the shorthands, the operator's own file, and the default layout."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from research_mcp.ask import relay
from research_mcp.routing import Route
from research_pipeline import replies


def test_fields_and_sections_fill_and_empty_ones_vanish() -> None:
    template = "A {{x}}\n{{#y}}\n\n**Y:** {{y}}\n{{/y}}\n{{^z}}\n\nno z{{/z}}\n\n\n\nend"
    assert replies.fill(template, {"x": "1", "y": "2", "z": ""}) == "A 1\n\n**Y:** 2\n\nno z\n\nend"
    assert replies.fill(template, {"x": "1", "y": " ", "z": "3"}) == "A 1\n\nend"


def test_an_unknown_name_stays_visible_so_a_typo_shows() -> None:
    assert replies.fill("{{pasages}}", {"passages": "p"}) == "{{pasages}}"


def test_comments_are_dropped_and_templates_split_by_marker() -> None:
    text = "intro\n<!-- template: a -->\n<!-- a note -->\nA {{x}}\n<!-- template: b -->\nB\n"
    assert replies.parse(text) == {"a": "A {{x}}", "b": "B"}


def test_my_own_file_replaces_a_template_and_keeps_the_rest(tmp_path: Path) -> None:
    default = tmp_path / "default.md"
    default.write_text("<!-- template: ask -->\nold\n<!-- template: answer -->\nkept\n")
    local = tmp_path / "mine.md"
    local.write_text("<!-- template: ask -->\nnew {{x}}\n")
    assert replies.load(default, local) == {"ask": "new {{x}}", "answer": "kept"}


def test_a_template_lays_out_the_relay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mine = tmp_path / "replies.md"
    mine.write_text(
        "<!-- template: ask -->\n## Sources first\n{{passages}}\n\n> {{limits}}\n{{#footer}}"
        "\n\n_{{footer}}_{{/footer}}\n"
    )
    monkeypatch.setenv("REPLY_TEMPLATES", str(mine))
    found = {"results": [{"text": "Pink noise.", "title": "T", "evidence_id": "W1#p0"}]}
    text = relay(Route("quick", "r", "q"), {}, ["ran"], {"evidence": found}, 1.0, request_id="r1")
    assert text.startswith("## Sources first\n1. “Pink noise.” (T) `W1#p0`\n\n> These are")
    assert text.endswith("_Request `r1`_")
    assert "What the research tool did" not in text


def test_the_default_templates_name_only_blocks_the_code_writes() -> None:
    templates = replies.load(replies.DEFAULT_TEMPLATES, None)
    assert set(templates) == {
        "ask",
        "ask_full",
        "pending",
        "pending_full",
        "answer",
        "answer_short",
    }
    short = {"results", "next", "for_you"}
    ask = {
        "problems",
        "route",
        "question",
        "seconds",
        "steps",
        "specialist",
        "guidance",
        "passages",
        "passages_none",
        "excluded",
        "search",
        "research",
        "coder",
        "limits",
        "request",
        "usage",
        "footer",
    } | short
    pending = {"problems", "working", "seconds", "steps", "request", "usage", "footer"} | short
    known = {
        "ask": ask, "ask_full": ask, "pending": pending, "pending_full": pending,
        "answer": {
            "question", "bottom_line", "thin", "risk", "findings", "conflicts", "tools", "limits",
            "sources", "footer",
        },
        "answer_short": {"question"} | short,
    }  # fmt: skip
    for name, template in templates.items():
        used = set(re.findall(r"\{\{[#^/]?\s*([\w-]+)\s*\}\}", template))
        assert used <= known[name], (name, used - known[name])


def _items(text: str, heading: str) -> list[str]:
    """The `- ` lines under a bold heading in a short reply."""
    part = text.split(f"**{heading}**\n", 1)
    if len(part) == 1:
        return []
    return [ln[2:] for ln in part[1].split("\n\n", 1)[0].splitlines() if ln.startswith("- ")]


def test_the_short_relay_is_three_lists_of_one_sentence_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REPLY_TEMPLATES", str(replies.DEFAULT_TEMPLATES))
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
