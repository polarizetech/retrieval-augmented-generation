"""math__physical_mix: pnm behind one tool, against a fake pnm that echoes its arguments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from research_mcp import physical_mix

FAKE = "import json, sys\nprint(json.dumps({'argv': sys.argv[1:]}, indent=2))\n"


def _config(tmp_path: Path) -> Path:
    script = tmp_path / "fake_pnm.py"
    script.write_text(FAKE)
    root = tmp_path / "pnm"
    for sub in ("runs", "devices", "interfaces"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    a = {"kind": "two tones", "recorded": "2026-11-08", "stage": "exploratory"}
    (root / "runs" / "pickle-a.json").write_text(json.dumps(a))
    b = {"kind": "sweep", "invalid": "backlog"}
    (root / "runs" / "pickle-b.json").write_text(json.dumps(b))
    (root / "devices" / "pickle.toml").write_text('slug = "pickle"\n')
    (root / "interfaces" / "scarlett.toml").write_text('slug = "scarlett"\n')
    cfg = tmp_path / "gateway.json"
    spec = {"command": sys.executable, "args": [str(script)], "cwd": str(root)}
    cfg.write_text(json.dumps({"calculators": {"physical_mixing": spec}}))
    return cfg


def test_without_config(tmp_path: Path) -> None:
    assert physical_mix.step(tmp_path / "none.json", {})["status"] == "not configured"


def test_calc_arguments() -> None:
    argv = physical_mix.calc_argv({"tones": "6:-20, 9.708:-20", "model": "cube", "order": 3})
    assert argv == [
        "calc", "--tones", "6:-20,9.708:-20", "--via", "model", "--model", "cube",
        "--order", "3", "--json",
    ]  # fmt: skip
    loop = physical_mix.calc_argv({
        "tones": "1000:-20,1370:-20", "via": "loop", "interface": "scarlett-2i2-3rd-gen",
        "devices": ["pickle"], "repeats": 99,
    })  # fmt: skip
    assert "--interface" in loop
    assert loop[loop.index("--device") + 1] == "pickle"
    assert loop[loop.index("--repeats") + 1] == "64"


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"tones": "1000; rm -rf /"}, "tones"),
        ({"tones": "1000", "expr": "t"}, "not both"),
        ({}, "not both"),
        ({"tones": "1000", "via": "uno"}, "via"),
        ({"tones": "1000", "via": "loop", "devices": ["pickle"]}, "go with interface"),
        ({"tones": "1000", "via": "loop", "interface": "--x"}, "interface"),
        ({"expr": "__import__('os')"}, "expr"),
    ],
)
def test_calc_refuses(arguments: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        physical_mix.calc_argv(arguments)


def test_calibrate_level_is_bounded() -> None:
    with pytest.raises(ValueError, match="level"):
        physical_mix.calibrate_argv({"level": 0})


def test_calc_runs_the_configured_command(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    out = physical_mix.step(cfg, {"tones": "6,9.708", "model": "cube"})
    assert out["status"] == "done"
    assert out["tier"].startswith("DERIVED")
    assert out["answer"]["argv"][:3] == ["calc", "--tones", "6,9.708"]
    assert out["answer"]["argv"][-1] == "--json"
    assert physical_mix.step(cfg, {"tones": "x"})["status"] == "refused"


def test_runs_and_setup_read_the_checkout(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    listed = physical_mix.step(cfg, {"action": "runs"})
    assert [r["name"] for r in listed["runs"]] == ["pickle-a.json", "pickle-b.json"]
    assert listed["runs"][1]["invalid"] == "backlog"
    narrowed = physical_mix.step(cfg, {"action": "runs", "match": "-a"})
    assert narrowed["runs"][0]["kind"] == "two tones"
    one = physical_mix.step(cfg, {"action": "runs", "name": "pickle-a.json"})
    assert one["run"]["recorded"] == "2026-11-08"
    assert physical_mix.step(cfg, {"action": "runs", "name": "../x.json"})["status"] == "refused"
    setup = physical_mix.step(cfg, {"action": "setup"})
    assert "pickle" in setup["devices"]
    assert "scarlett" in setup["interfaces"]
