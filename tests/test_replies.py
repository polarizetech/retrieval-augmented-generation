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
    assert set(templates) == {"ask", "pending", "answer"}
    known = {
        "ask": {
            "problems", "route", "question", "seconds", "steps", "specialist", "guidance",
            "passages", "passages_none", "excluded", "search", "research", "coder", "limits",
            "request", "usage", "footer",
        },
        "pending": {"problems", "working", "seconds", "steps", "request", "usage", "footer"},
        "answer": {
            "question", "bottom_line", "risk", "findings", "conflicts", "tools", "limits",
            "sources", "footer",
        },
    }  # fmt: skip
    for name, template in templates.items():
        used = set(re.findall(r"\{\{[#^/]?\s*([\w-]+)\s*\}\}", template))
        assert used <= known[name], (name, used - known[name])
