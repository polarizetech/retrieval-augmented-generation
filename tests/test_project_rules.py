"""The project rules step: a project's binding rules go with every `ask` reply.

The projects here are folders made in a temporary directory: a polarize tool with an AGENTS.md
in the scaffold's shape, and polarize-thesis holding the shared integration profile.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from research_mcp import brief, project_rules
from research_mcp.ask import Asker
from research_mcp.routing import load_rules

ROOT = Path(__file__).resolve().parents[1]
RULES = load_rules(ROOT / "config" / "routing.default.toml")
CONFIG = RULES.project_rules

AGENTS = """<!-- kit_ap:start -->
# Agent protocols

- The rules in this block and the documents in `.agents/protocols/` are binding.
<!-- kit_ap:end -->

# polarize-demo

**Kind:** tool (`TOOL.toml`). A demo.

- **Depend on tags.** Pin each tool at a released tag, never a branch.
- **No absolute home-directory paths.**
- **Visual work uses the design system.** Anything visual uses polarize-ui.
  - **Design first, with the designer.** A page is designed before it is built.

**The output is for reference only (owner, 2026-11-13).** Nothing this tool produces is played to a
listener. The owner: "The mixer is for reference only, yes."

Text with a **term**, in running prose, is not a rule.
"""

PROFILE = """# Integrator

## Rules that hold in all five

- **Departures, never averages** (owner, 2026-11-08). No mean, median, RMS or smoothing.
- **Depend on tags.** Pin each tool at a released tag in `pyproject.toml`.

## Known gaps

- **holo's wheel is old.** Not a rule.

## Lessons already paid for

- **P6:** a lesson, not a rule.

### Added in the 2026-11-13 pass

- **Only polarize-particle makes sound** (owner, 2026-11-13; all five tools). Hard requirement.
"""


@pytest.fixture
def projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "sites"
    (root / "polarize-demo").mkdir(parents=True)
    (root / "polarize-demo" / "AGENTS.md").write_text(AGENTS)
    (root / "polarize-demo" / "CLAUDE.md").write_text("@AGENTS.md\n")
    profile = root / "polarize-thesis" / ".claude" / "agents" / "polarize-integrator.md"
    profile.parent.mkdir(parents=True)
    profile.write_text(PROFILE)
    (root / "polarize-thesis" / "AGENTS.md").write_text("- **Predict first.** Before any run.\n")
    (root / "other-project").mkdir()
    (root / "other-project" / "AGENTS.md").write_text("- **A rule.** Not read.\n")
    monkeypatch.setenv("PROJECT_RULES_DIRS", str(root))
    monkeypatch.setenv("POLARIZE_TOOLS_DIR", str(root))
    return root


def config(**extra: Any) -> dict[str, Any]:
    shared = [{**CONFIG["shared"][0], "applies_to": ["polarize-demo", "polarize-thesis"]}]
    return {**CONFIG, "shared": shared, **extra}


def by_id(held: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in held["rules"]}


def test_rules_are_the_bold_led_items() -> None:
    rules = project_rules.parse(AGENTS, "AGENTS.md")
    assert [r["title"] for r in rules] == [
        "Depend on tags",
        "No absolute home-directory paths",
        "Visual work uses the design system",
        "The output is for reference only",
    ]
    visual = rules[2]
    assert "Design first, with the designer" in visual["text"]  # a sub-bullet stays with its rule
    owner = rules[3]
    assert owner["owner"] == "2026-11-13"
    assert owner["source"] == "AGENTS.md:16"
    assert "The owner:" in owner["text"]  # a paragraph rule runs to its blank line


def test_a_profile_is_read_only_in_its_rule_sections() -> None:
    rules = project_rules.parse(PROFILE, "p.md", ["rules", "added in"])
    assert [r["id"] for r in rules] == [
        "departures-never-averages",
        "depend-on-tags",
        "only-polarize-particle-makes-sound",
    ]
    assert rules[2]["applies"] == "all five tools"


def test_a_project_gets_its_own_and_the_shared_rules(projects: Path) -> None:
    held = project_rules.load("polarize-demo", "", config())
    assert held is not None
    rules = by_id(held)
    assert "agent-protocols-are-binding" in rules
    assert "departures-never-averages" in rules
    assert "holo-s-wheel-is-old" not in rules
    assert "p6" not in rules
    tags = rules["depend-on-tags"]  # in both files: the project's own wording, the copy noted
    assert tags["source"] == "AGENTS.md:11"
    assert tags["also_in"] == ["polarize-integrator.md:6"]
    assert held["owner_rules"] == 3
    assert [s["file"] for s in held["sources"]] == [
        "polarize-demo/AGENTS.md",
        "polarize-demo/CLAUDE.md",
        "polarize-thesis/.claude/agents/polarize-integrator.md",
    ]
    assert "Follow every one" in held["binding"]


def test_only_configured_projects_carry_rules(projects: Path) -> None:
    assert project_rules.load("other-project", "", config()) is None
    assert project_rules.load("", "", config()) is None
    missing = project_rules.load("polarize-nowhere", "", config())
    assert missing is not None
    assert "no checkout" in missing["error"]


def test_the_rules_nearest_the_request_come_in_full(projects: Path) -> None:
    held = project_rules.load("polarize-demo", "Smooth the series with a running average", config())
    assert held is not None
    first = held["rules"][0]
    assert first["id"] == "departures-never-averages"
    assert "text" in first
    rest = [r for r in held["rules"] if "line" in r]
    assert rest
    assert all(len(r["line"]) <= project_rules.LINE_CHARS for r in rest)


def test_an_exception_needs_explicit_words_and_a_named_rule(projects: Path) -> None:
    asked = "Make an exception to the departures, never averages rule and smooth the series."
    held = project_rules.load("polarize-demo", asked, config())
    assert held is not None
    assert held["exceptions"]["rules"] == ["departures-never-averages"]
    assert by_id(held)["departures-never-averages"]["set_aside"]
    assert "set_aside" not in by_id(held)["depend-on-tags"]

    named_only = project_rules.load("polarize-demo", "Departures, never averages: why?", config())
    assert named_only is not None
    assert named_only["exceptions"] == {"asked": False, "rules": []}

    vague = project_rules.load("polarize-demo", "Just break the rule this once.", config())
    assert vague is not None
    assert vague["exceptions"]["asked"]
    assert vague["exceptions"]["rules"] == []
    assert "ask the person which rule" in vague["exceptions"]["note"]


def asker(tmp_path: Path) -> Asker:
    async def retrieve(query: str, limit: int, max_chars: int) -> dict[str, Any]:
        return {"retrieval": "hybrid", "n": 0, "excluded": [], "results": []}

    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError(f"no upstream call expected: {name}")

    (tmp_path / "agents").mkdir(exist_ok=True)
    return Asker(
        retrieve=retrieve,
        call_tool=call_tool,
        has_tool=lambda _: False,
        rules_paths=(ROOT / "config" / "routing.default.toml", None),
        profiles_dir=tmp_path / "agents",
        extensions_dir=None,
    )


def test_ask_puts_the_projects_rules_first(projects: Path, tmp_path: Path) -> None:
    ask = asker(tmp_path).ask("How should I pin a dependency?", project="polarize-demo")
    out = asyncio.run(ask)
    held = out["results"]["project_rules"]
    assert held["project"] == "polarize-demo"
    assert any(g.startswith("Rules of polarize-demo:") for g in out["summary"]["results"])
    assert "**Project rules (polarize-demo, binding):**" in out["relay_full"]
    compact = brief.compact(out)
    assert next(iter(compact["brief"])) == "project_rules"
    assert compact["answer_format"].startswith(brief.RULES_FORMAT)
    assert "**Rules checked:**" in compact["answer_format"]


def test_ask_finds_the_project_from_the_callers_folder(projects: Path, tmp_path: Path) -> None:
    out = asyncio.run(
        asker(tmp_path).ask("What is a departure?", caller={"project": "polarize-demo"})
    )
    assert out["results"]["project_rules"]["project"] == "polarize-demo"


def test_ask_without_a_rules_project_is_unchanged(projects: Path, tmp_path: Path) -> None:
    out = asyncio.run(asker(tmp_path).ask("What is a departure?", project="other-project"))
    assert "project_rules" not in out["results"]
    assert not brief.compact(out)["answer_format"].startswith(brief.RULES_FORMAT)


def test_ask_asks_which_rule_when_an_exception_names_none(projects: Path, tmp_path: Path) -> None:
    out = asyncio.run(
        asker(tmp_path).ask("Break the rule just this once.", project="polarize-demo")
    )
    assert any("Name the rule" in y for y in out["summary"]["for_you"])
