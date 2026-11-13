"""The polarize tools step: pinned reads of the tools' repos and holo renders run on a copy.

The tool repositories here are small git repos made in a temporary folder. Their render scripts
write a short WAV with the standard library, standing in for holo's own renders; `uv` is replaced
by a shim that runs the script with this Python, so no environment is built.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from research_mcp import polarize
from research_mcp.ask import Asker
from research_mcp.routing import load_rules

ROOT = Path(__file__).resolve().parents[1]
RULES = load_rules(ROOT / "config" / "routing.default.toml")

WRITER = """
import json, struct, sys, wave
from pathlib import Path

def write(path, hz):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(b"".join(struct.pack("<h", (i * hz) % 3000) for i in range(8000)))
"""

SPIRAL = (
    '"""A spiral.\n\nusage: uv run python render_spiral.py OUT_DIR SUN_AZ_DEG\n"""\n'
    + WRITER
    + """
out, az = Path(sys.argv[1]), float(sys.argv[2])
write(out / "spiral.wav", int(az))
(out / "spiral.json").write_text(json.dumps({"sun_az_deg": az, "seconds": 1.0}))
"""
)
BELL = (
    '"""A bell; writes bell.wav beside itself."""\n'
    + WRITER
    + '\nwrite(Path(__file__).parent / "bell.wav", 440)\n'
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def make_repo(root: Path, name: str, files: dict[str, str], tag: str = "") -> Path:
    repo = root / name
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "first")
    if tag:
        git(repo, "tag", tag)
    return repo


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "tools"
    holo = make_repo(
        root,
        "polarize-holo",
        {
            "README.md": "holo at the tag\n",
            "TOOL.toml": 'name = "polarize-holo"\nversion = "0.1.0"\n',
            "renders/2026-01-01-spiral/render_spiral.py": SPIRAL,
        },
        tag="v0.1.0",
    )
    # Past the tag: a new render and a changed README.
    (holo / "README.md").write_text("holo after the tag\n")
    (holo / "renders/2026-01-02-bell").mkdir(parents=True)
    (holo / "renders/2026-01-02-bell/render_bell.py").write_text(BELL)
    git(holo, "add", "-A")
    git(holo, "commit", "-qm", "bell")
    make_repo(root, "polarize-mixer", {"README.md": "mixer\n", "TOOL.toml": ""}, tag="v0.2.0")
    make_repo(root, "polarize-physics", {"README.md": "physics, untagged\n"})
    shim = tmp_path / "uv"
    skip = 'while [ "$1" != "python" ]; do shift; done\nshift\n'
    shim.write_text(f'#!/bin/sh\n{skip}exec {sys.executable} "$@"\n')
    shim.chmod(0o755)
    monkeypatch.setenv("POLARIZE_TOOLS_DIR", str(root))
    monkeypatch.setenv("POLARIZE_RENDER_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("POLARIZE_UV", str(shim))
    for tool in polarize.TOOLS:
        monkeypatch.delenv(f"POLARIZE_{tool.upper()}_DIR", raising=False)
    return root


def test_each_tool_is_pinned_to_its_newest_tag(tools: Path) -> None:
    holo = polarize.pin("holo")
    assert holo["ref"] == "v0.1.0"
    assert holo["pinned"]
    assert holo["ahead"] == 1
    assert "1 commit(s) past it" in holo["note"]
    physics = polarize.pin("physics")
    assert not physics["pinned"]
    assert "no release tag" in physics["note"]
    clock = polarize.pin("clock")
    assert clock["ref"] == ""
    assert "no local checkout" in clock["note"]


def test_routing_names_the_tools_and_audio() -> None:
    q = "How does polarize-holo use polarize physics? Make an audio file."
    assert polarize.named(q, RULES.polarize) == ["holo", "physics"]
    assert polarize.wants_audio(q, RULES.polarize_audio)
    assert polarize.named("How do mixers work?", RULES.polarize) == []


def test_files_are_read_through_github_at_the_tag(tools: Path) -> None:
    asked: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
        asked.append((name, arguments))
        if name == "repos__read_file":
            return {"text": f"{arguments['path']} at {arguments['ref']}"}
        return [{"path": "apps/orrery/app.js", "url": "https://github.com/x"}]

    got = asyncio.run(
        polarize.read("holo", "how is the orrery drawn?", RULES.polarize, call_tool, lambda _: True)
    )
    reads = [a for n, a in asked if n == "repos__read_file"]
    assert {a["ref"] for a in reads} == {"v0.1.0"}
    assert {a["repository"] for a in reads} == {"polarizetech/polarize-holo"}
    assert got["files"]["README.md"] == {
        "text": "README.md at v0.1.0",
        "truncated": False,
        "via": "GitHub",
    }
    search = next(a for n, a in asked if n == "repos__search_code")
    assert search["query"] == "orrery drawn repo:polarizetech/polarize-holo"
    assert got["code"] == [{"path": "apps/orrery/app.js", "url": "https://github.com/x"}]


def test_without_github_the_checkout_is_read_at_the_tag(tools: Path) -> None:
    got = asyncio.run(polarize.read("holo", "", RULES.polarize, None, lambda _: False))
    assert got["files"]["README.md"]["text"] == "holo at the tag\n"  # not the newer HEAD text
    assert got["files"]["README.md"]["via"] == "local checkout"


def test_renders_say_whether_they_are_in_the_tag(tools: Path) -> None:
    listed = polarize.renders()
    rows = {r["slug"]: r for r in listed["renders"]}
    assert rows["2026-01-01-spiral"]["ref"] == "v0.1.0"
    assert rows["2026-01-01-spiral"]["scripts"] == [
        {"script": "render_spiral.py", "args": ["OUT_DIR", "SUN_AZ_DEG"]}
    ]
    assert rows["2026-01-02-bell"]["ref"] == "HEAD"
    assert not rows["2026-01-02-bell"]["at_tag"]


def test_a_render_asks_for_what_its_usage_line_needs(tools: Path) -> None:
    got = polarize.render("2026-01-01-spiral")
    assert got["status"] == "needs arguments"
    assert got["missing"] == ["SUN_AZ_DEG"]


def test_a_render_runs_holos_script_on_a_copy(tools: Path) -> None:
    holo = tools / "polarize-holo"
    got = polarize.render("2026-01-01-spiral", args={"sun_az_deg": 209.1})
    assert got["status"] == "done", got
    assert got["pinned"]
    assert got["ref"] == "v0.1.0"
    assert got["commit"] == git(holo, "rev-parse", "--short", "v0.1.0")
    wav = got["wavs"][0]
    assert Path(wav["path"]).name == "spiral.wav"
    assert (wav["sample_rate"], wav["bits"], wav["channels"], wav["seconds"]) == (8000, 16, 1, 1.0)
    assert got["settings"]["spiral.json"] == {"sun_az_deg": 209.1, "seconds": 1.0}
    assert got["command"][-2:] == [got["output_dir"], "209.1"]
    assert got["environment"] == "holo's root project"
    assert git(holo, "status", "--short") == ""  # holo's checkout is never written to


def test_a_render_past_the_tag_says_so(tools: Path) -> None:
    got = polarize.render("2026-01-02-bell")
    assert got["status"] == "done", got
    assert not got["pinned"]
    assert "not in holo's newest tag (v0.1.0)" in got["note"]
    assert [Path(w["path"]).name for w in got["wavs"]] == ["bell.wav"]


def test_a_render_refuses_unknown_names(tools: Path) -> None:
    with pytest.raises(ValueError, match="no render"):
        polarize.render("2026-09-09-missing")
    with pytest.raises(ValueError, match="bad slug"):
        polarize.render("../escape")
    with pytest.raises(ValueError, match="no script"):
        polarize.render("2026-01-02-bell", script="render_other.py")
    got = polarize.render("2026-01-01-spiral", args={"SUN_AZ_DEG": "1; rm -rf /"})
    assert got["status"] == "needs arguments"


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


def test_ask_reads_the_named_tools(tools: Path, tmp_path: Path) -> None:
    out = asyncio.run(asker(tmp_path).ask("What does polarize-mixer do?"))
    read = out["results"]["polarize"]["read"]
    assert [r["tool"] for r in read] == ["polarize-mixer"]
    assert read[0]["files"]["README.md"]["text"] == "mixer\n"
    assert any(g.startswith("Read polarize-mixer at v0.2.0") for g in out["summary"]["results"])
    assert "**Polarize tools**" in out["relay_full"]
    assert "render" not in out["results"]["polarize"]


def test_ask_renders_audio_when_asked(tools: Path, tmp_path: Path) -> None:
    out = asyncio.run(asker(tmp_path).ask("Make an audio file of the bell with polarize-holo"))
    audio = out["results"]["polarize"]["render"]
    assert audio["status"] == "done"
    assert any("polarize-holo rendered 2026-01-02-bell" in g for g in out["summary"]["results"])
    assert "bell.wav" in out["relay_full"]


def test_ask_lists_renders_when_none_is_named(tools: Path, tmp_path: Path) -> None:
    out = asyncio.run(asker(tmp_path).ask("Generate audio with polarize-holo"))
    assert "render" not in out["results"]["polarize"]
    assert [r["slug"] for r in out["results"]["polarize"]["renders"]] == [
        "2026-01-01-spiral",
        "2026-01-02-bell",
    ]
    assert any("Name one of holo's renders" in y for y in out["summary"]["for_you"])


def test_a_question_about_polarize_clock_skips_the_sky_prompt(tools: Path, tmp_path: Path) -> None:
    out = asyncio.run(asker(tmp_path).ask("How does polarize-clock compute lunar time?"))
    assert "sky" not in out["results"]
    clock = out["results"]["polarize"]["read"][0]
    assert clock["tool"] == "polarize-clock"
    assert not clock["pinned"]
    assert json.dumps(out)  # the reply stays serialisable
