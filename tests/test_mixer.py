"""The sign mixer step: derived values, odd-order lines, tone parsing and the random run."""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from research_mcp import mixer
from research_mcp.ask import Asker
from tests.test_ask import ROOT, brief


@pytest.mark.parametrize(
    ("m", "n", "r", "expected", "tol"),
    [
        (1, 0, 0.0, 2 / math.pi, 1e-12),  # square wave fundamental
        (1, 0, 1.0, 4 / math.pi**2, 2e-6),  # equal tones: product of two square waves
        (2, -1, 1.0, -4 / (3 * math.pi**2), 2e-6),
        (0, 1, 0.3, 0.09660531855236879, 1e-12),  # polarity-calculator's Julia and Python values
        (2, -1, 0.3, -0.09220714183712568, 1e-12),
        (1, 1, 0.3, 0.0, 0.0),  # even order: none
    ],
)
def test_coefficients_match_the_calculator(
    m: int, n: int, r: float, expected: float, tol: float
) -> None:
    assert abs(mixer.coefficient(m, n, r) - expected) <= tol


def test_strong_tone_captures_the_weak_one() -> None:
    cap = mixer.sign_mix(100, 1, 137, 1e-4, order=3)["capture"]
    assert abs(cap["ratio"] - 0.5) < 1e-6


def test_only_odd_order_lines_appear() -> None:
    out = mixer.sign_mix(1000, 1, 1370, 0.5, order=9, limit=1000)
    freqs = {ln["f_hz"] for ln in out["lines"]}
    assert {1000, 1370, 630, 1740}.issubset(freqs)  # f1, f2, 2f1 − f2, 2f2 − f1
    assert not freqs & {2000, 2740, 2370, 370}  # 2f1, 2f2, f1 + f2, f2 − f1
    assert out["lines"][0]["f_hz"] == 1000


def test_the_keyed_mixer_scales_lines_by_erf() -> None:
    bare = mixer.sign_mix(120, 1, 210, 0.4, order=5)["lines"][0]["amplitude"]
    keyed = mixer.sign_mix(120, 1, 210, 0.4, order=5, keyed=True)["lines"][0]["amplitude"]
    assert abs(keyed / bare - math.erf(1 / math.sqrt(2))) < 1e-9


def test_tones_are_read_from_plain_words() -> None:
    assert mixer.parse_tones("mix 120 Hz and 0.21 kHz, a2=0.4") == {"f1": 120, "f2": 210, "a2": 0.4}
    assert mixer.parse_tones("what does clipping do?") is None


def test_without_config_only_the_prediction_comes_back(tmp_path: Path) -> None:
    out = mixer.sign_mix_step(tmp_path / "none.json", {"f1": 120, "f2": 210, "a2": 0.4})
    assert out["prediction"]["lines"]
    assert out["random_run"]["status"] == "not configured"


def _fake_calculator(tmp_path: Path) -> Path:
    """A gateway config whose sign_mixer prints a run, recording the arguments it was given."""
    script = tmp_path / "fake_mix.py"
    script.write_text(
        "import json, sys\n"
        "print(json.dumps({'passed': True, 'key_source': 'fake', 'lines_pass': True,"
        " 'twin_pass': True, 'run': 'runs/x', 'argv': sys.argv[1:],"
        " 'settle': None if '--no-settle' in sys.argv else {'K_settled': 2.9,"
        " 'twin_K_settled': 2.95, 'lyapunov': 0.0001}}))\n"
    )
    cfg = tmp_path / "gateway.json"
    spec = {"command": sys.executable, "args": [str(script), "--json"]}
    cfg.write_text(json.dumps({"calculators": {"sign_mixer": spec}}))
    return cfg


def test_the_configured_run_has_settle_on_by_default(tmp_path: Path) -> None:
    cfg = _fake_calculator(tmp_path)
    run = mixer.sign_mix_step(cfg, {"f1": 120, "f2": 210, "a2": 0.4})["random_run"]
    assert run["status"] == "done"
    assert run["settle"]["K_settled"] == 2.9
    off = mixer.sign_mix_step(cfg, {"f1": 120, "f2": 210, "settle": False})["random_run"]
    assert off["settle"] is None


def test_ask_adds_the_mixer_step(tmp_path: Path) -> None:
    public, private = tmp_path / "public", tmp_path / "private"
    brief(public, "researcher", "literature", "")
    brief(private, "noise-engineering", "noise colour pink brown", "Name the domain.")
    (tmp_path / "routing.toml").write_text("")
    cfg = _fake_calculator(tmp_path)

    async def retrieve(query: str, limit: int, max_chars: int) -> dict[str, Any]:
        return {"retrieval": "hybrid", "n": 0, "excluded": [], "results": []}

    async def call_tool(name: str, args: dict[str, Any]) -> Any:
        raise AssertionError(name)

    a = Asker(
        retrieve=retrieve,
        call_tool=call_tool,
        has_tool=lambda n: False,
        rules_paths=(ROOT / "config" / "routing.default.toml", tmp_path / "routing.toml"),
        profiles_dir=public,
        extensions_dir=private,
        sign_mixer=lambda args: mixer.sign_mix_step(cfg, args),
    )
    out = asyncio.run(a.ask("What does a hard limiter make from 120 Hz and 210 Hz, a2=0.4?"))
    assert out["results"]["mixer"]["random_run"]["status"] == "done"
    assert "Sign mixer: strongest lines 120 Hz" in out["relay"]
    assert "settled at K 2.90" in out["relay"]
    assert "**Sign mixer**" in out["relay_full"]
    plain = asyncio.run(a.ask("What is pink noise?"))
    assert "mixer" not in plain["results"]
