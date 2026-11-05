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


def test_audio_arguments_are_checked() -> None:
    ref = "spaces/workbench~songs~super-8-orchestral@2e9efc7c820f83f6"
    argv = mixer._audio_argv(
        {
            "spaces": ref,
            "path": "cello.wav",
            "mixer": "voltage",
            "kind": "tanh",
            "params": {"g": 2, "v0": 0.2},
        }
    )
    assert argv[:5] == ["run", "--spaces", ref, "--path", "cello.wav"]
    assert "--param" in argv
    assert "g=2.0" in argv
    for bad in (
        {"spaces": "s3://elsewhere", "path": "a.wav"},
        {"spaces": ref, "path": "--file=/etc/passwd"},
        {"spaces": ref, "path": "../../secret.wav"},
        {"file": "relative.wav"},
        {"file": "/etc/hosts"},
        {"spaces": ref, "path": "a.wav", "mixer": "voltage", "kind": "laser"},
        {"spaces": ref, "path": "a.wav", "mixer": "voltage", "params": {"cmd": 1}},
        {"action": "delete"},
    ):
        with pytest.raises(ValueError, match=r"\w"):
            mixer._audio_argv(bad)
    assert mixer._audio_argv({"action": "list", "match": "cello"})[:3] == ["list", "--limit", "50"]
    assert mixer._audio_argv({"action": "ledger", "last": 5}) == ["ledger", "--last", "5"]


def test_audio_without_config(tmp_path: Path) -> None:
    out = mixer.audio_step(tmp_path / "none.json", {"action": "list"})
    assert out["status"] == "not configured"


def test_audio_run_through_a_configured_command(tmp_path: Path) -> None:
    script = tmp_path / "fake_audio.py"
    script.write_text(
        "import json, sys\n"
        "a = sys.argv[1:]\n"
        "if a[0] == 'ledger':\n"
        "    print(json.dumps({'table': '| t |', 'entries': [{'kind': 'audio-sign'}]}))\n"
        "else:\n"
        "    print(json.dumps({'summary': {'passed': True, 'K_settled': 2.88}, 'inputs': {'file': "
        "'x.wav'},"
        " 'settings': {'mixer': 'sign', 'argv': a}, 'run': 'runs/audio-mix/x',"
        " 'channels': [{'channel': 0, 'checks': {'disagree': 0.158}, 'settle': {'K_settled': 2.88},"
        " 'keyed_vs_twin': {'peaks': [{'f': 199.0}]}, 'big': list(range(5))}]}))\n"
    )
    cfg = tmp_path / "gateway.json"
    cfg.write_text(
        json.dumps(
            {"calculators": {"audio_mixer": {"command": sys.executable, "args": [str(script)]}}}
        )
    )
    ref = "spaces/workbench~songs~super-8-orchestral@2e9efc7c820f83f6"
    out = mixer.audio_step(cfg, {"spaces": ref, "path": "cello.wav"})
    assert out["status"] == "done"
    assert out["summary"]["K_settled"] == 2.88
    assert out["channels"][0]["keyed_vs_twin"] == [{"f": 199.0}]
    assert "big" not in out["channels"][0]
    assert "--json" in out["settings"]["argv"]
    assert mixer.audio_step(cfg, {"action": "ledger"})["table"] == "| t |"


def test_ngspice_arguments_are_checked(tmp_path: Path) -> None:
    ref = "spaces/workbench~songs~super-8-orchestral@2e9efc7c820f83f6"
    base = {"spaces": ref, "path": "cello.wav", "mixer": "voltage", "engine": "ngspice"}
    argv = mixer._audio_argv({**base, "circuit": "detector", "params": {"C": 2.2e-7}})
    assert argv[argv.index("--engine") + 1] == "ngspice"
    assert argv[argv.index("--circuit") + 1] == "detector"
    assert "C=2.2e-07" in argv
    net = tmp_path / "mine.cir"
    net.write_text("* mine\nR1 in out 1k\n.end\n")
    argv = mixer._audio_argv(
        {**base, "circuit": "netlist", "netlist": str(net), "out_vector": "v(out)"}
    )
    assert argv[argv.index("--netlist") + 1] == str(net)
    for bad in (
        {**base, "circuit": "laser"},
        {**base, "circuit": "netlist", "netlist": "mine.cir", "out_vector": "v(out)"},
        {**base, "circuit": "netlist", "netlist": str(net), "out_vector": "v(out); shell rm"},
        {
            **base,
            "circuit": "netlist",
            "netlist": str(tmp_path / "none.cir"),
            "out_vector": "v(out)",
        },
        {**base, "engine": "spectre"},
    ):
        with pytest.raises(ValueError, match=r"\w"):
            mixer._audio_argv(bad)


def test_sigma_delta_arguments_are_checked() -> None:
    ref = "spaces/workbench~songs~super-8-orchestral@2e9efc7c820f83f6"
    base = {"spaces": ref, "path": "full-mix.wav", "mixer": "sigma-delta"}
    argv = mixer._audio_argv(
        {**base, "provider": "ngspice", "sigma_th": 3e-4, "params": {"k_latch": 0.05}}
    )
    assert argv[argv.index("--provider") + 1] == "ngspice"
    assert argv[argv.index("--sigma-th") + 1] == "0.0003"
    assert "k_latch=0.05" in argv
    for bad in (
        {**base, "provider": "brian"},
        {**base, "params": {"rm -rf": 1}},
    ):
        with pytest.raises(ValueError, match=r"\w"):
            mixer._audio_argv(bad)


def test_surprise_arguments_are_checked() -> None:
    ref = "spaces/workbench~songs~super-8-orchestral@2e9efc7c820f83f6"
    base = {"spaces": ref, "path": "full-mix.wav", "mixer": "surprise"}
    argv = mixer._audio_argv({**base, "noise": 1.0, "oversample": 4, "min_bits": 2})
    assert argv[argv.index("--mixer") + 1] == "surprise"
    assert argv[argv.index("--noise") + 1] == "1.0"
    assert argv[argv.index("--oversample") + 1] == "4"
    assert argv[argv.index("--min-bits") + 1] == "2.0"
    for bad in ({**base, "oversample": 64}, {**base, "noise": -1}, {**base, "lags": "x"}):
        with pytest.raises(ValueError, match=r"\w"):
            mixer._audio_argv(bad)
