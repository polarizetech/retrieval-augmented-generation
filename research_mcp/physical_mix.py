"""Real hardware as a nonlinear mixer: polarizetech/physical-nonlinear-mixing (pnm), one tool.

The tool runs `pnm` as configured under `calculators.physical_mixing` (a command and its working
folder, the tool's checkout) and reads what it recorded:

- ``calc``: tones (``"1000:-20,1370:-20"``) or an equation in t, through a digital model
  (``via=model``, exact) or the audio interface's loop (``via=loop``: sent out of the interface,
  through the devices in the loop, and read back on its input). The answer lists every line with
  its level, phase and the combination m·f1 + n·f2 that made it, and, through the loop, whether
  each line passed the test across bursts. Measured lines are exploratory, never findings.
- ``calibrate``: the loop's delay, polarity and gain with a quiet noise burst.
- ``runs``: the recorded runs in the checkout's ``runs/`` (``name`` reads one).
- ``setup``: the interfaces and devices on record (``interfaces/``, ``devices/``).

Arguments are checked here and passed as a list (no shell). Hardware runs need the interface on
this Mac; the Olimex readings of the pickle are recorded runs, read with ``runs``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from research_pipeline.upstream import spec_for

SECTION = "calculators"
NAME = "physical_mixing"
SOURCE = "polarizetech/physical-nonlinear-mixing (pnm)"
_NUM = r"-?\d+(\.\d+)?"
_TONE = rf"\d+(\.\d+)?(:{_NUM}(:{_NUM})?)?"
_TONES = re.compile(rf"^\s*{_TONE}(\s*,\s*{_TONE})*\s*$")
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_SET = re.compile(r"^[A-Za-z0-9._-]+\.[A-Za-z0-9_-]+=[A-Za-z0-9:._ -]{1,40}$")
_EXPR = re.compile(r"^[0-9a-zA-Z_+\-*/(). ,%<>=]{1,300}$")
MAX_READ = 200_000


def _spec(config_path: Path | None) -> dict[str, Any] | None:
    return spec_for(config_path, NAME, SECTION) if config_path is not None else None


def _not_configured() -> dict[str, Any]:
    return {
        "status": "not configured",
        "note": f"add `{SECTION}.{NAME}` to the gateway config "
        "(see config/mcp-gateway.example.json)",
    }


def _hardware(arguments: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if arguments.get("interface"):
        out += ["--interface", _slug(arguments["interface"], "interface")]
        for d in arguments.get("devices") or []:
            out += ["--device", _slug(d, "device")]
        for s in arguments.get("set") or []:
            if not _SET.match(str(s)):
                raise ValueError(f"set {s!r}: give SLUG.NAME=VALUE")
            out += ["--set", str(s)]
        if arguments.get("loop") is not None:
            out += ["--loop", _slug(str(arguments["loop"]), "loop")]
    elif arguments.get("devices") or arguments.get("set") or arguments.get("loop") is not None:
        raise ValueError("devices, set and loop go with interface")
    return out


def _slug(value: Any, what: str) -> str:
    v = str(value)
    if not _SLUG.match(v):
        raise ValueError(f"{what} {v!r} is not a name from the tool's files")
    return v


def calc_argv(arguments: dict[str, Any]) -> list[str]:
    """`pnm calc` arguments from the tool call, every value checked."""
    tones, expr = arguments.get("tones"), arguments.get("expr")
    if (tones is None) == (expr is None):
        raise ValueError("give tones or expr, not both")
    out = ["calc"]
    if tones is not None:
        if not _TONES.match(str(tones)):
            raise ValueError("tones: HZ[:DBFS[:PHASE_DEG]],... e.g. 1000:-20,1370:-20")
        out += ["--tones", str(tones).replace(" ", "")]
    else:
        if not _EXPR.match(str(expr)):
            raise ValueError("expr: an equation in t: numbers, + - * / ** ( ) and function names")
        out += ["--expr", str(expr)]
    via = arguments.get("via", "model")
    if via not in ("model", "loop"):
        raise ValueError("via is model or loop")
    out += ["--via", via]
    if via == "model":
        model = str(arguments.get("model", "cube"))
        if not _EXPR.match(model):
            raise ValueError(
                "model: linear, square, cube, sign, hard[:DBFS], tanh[:DB], poly:a,b,... or y in x"
            )
        out += ["--model", model]
    out += ["--order", str(int(arguments.get("order", 5)))]
    if via == "loop":
        out += ["--repeats", str(max(3, min(64, int(arguments.get("repeats", 8)))))]
        out += _hardware(arguments)
    return [*out, "--json"]


def calibrate_argv(arguments: dict[str, Any]) -> list[str]:
    level = float(arguments.get("level", -30.0))
    if not -80 <= level <= -6:
        raise ValueError("level is -80 to -6 dBFS RMS")
    return ["loop", "calibrate", "--level", str(level), *_hardware(arguments), "--json"]


def _run(spec: dict[str, Any], argv: list[str], timeout: float) -> dict[str, Any]:
    cmd = [spec["command"], *(str(v) for v in spec.get("args", [])), *argv]
    env = {**os.environ, **{str(k): str(v) for k, v in spec.get("env", {}).items()}}
    try:
        done = subprocess.run(  # noqa: S603 (the configured command; arguments checked above)
            cmd,
            cwd=spec.get("cwd"),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "note": f"{NAME}: {exc}"}
    if done.returncode != 0:
        tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
        return {"status": "failed", "note": f"pnm exited {done.returncode}: {tail[0]}"}
    try:
        return {"status": "done", "answer": json.loads(done.stdout)}
    except ValueError:
        return {"status": "failed", "note": "pnm did not print JSON"}


def _folder(spec: dict[str, Any], sub: str) -> Path:
    root = Path(spec.get("root") or spec.get("cwd") or ".")
    return root / sub


def _runs(spec: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    folder = _folder(spec, "runs")
    if not folder.is_dir():
        return {"status": "failed", "note": f"no runs folder at {folder}"}
    name = arguments.get("name")
    if name:
        if not re.match(r"^[A-Za-z0-9._-]+\.json$", str(name)):
            raise ValueError("name: a .json file in runs/, as listed")
        path = folder / str(name)
        if not path.is_file():
            return {"status": "failed", "note": f"no run {name}"}
        text = path.read_text()
        out = {"status": "done", "name": name, "bytes": len(text)}
        if len(text) > MAX_READ:
            return out | {"truncated": True, "text": text[:MAX_READ]}
        return out | {"run": json.loads(text)}
    match = str(arguments.get("match", "")).lower()
    rows = []
    for p in sorted(folder.glob("*.json")):
        if match and match not in p.name.lower():
            continue
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        head = d if isinstance(d, dict) else {}
        rows.append(
            {
                "name": p.name,
                "kind": head.get("kind") or head.get("input", {}).get("kind"),
                "recorded": head.get("recorded"),
                "stage": head.get("stage"),
                "invalid": head.get("invalid"),
            }
        )
    note = "read one with name; recorded runs are exploratory, not findings"
    return {"status": "done", "runs": rows, "note": note}


def _setup(spec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"status": "done"}
    for sub in ("interfaces", "devices"):
        folder = _folder(spec, sub)
        files = sorted(folder.glob("*.toml")) if folder.is_dir() else []
        out[sub] = {p.stem: p.read_text() for p in files}
    return out


def step(config_path: Path | None, arguments: dict[str, Any]) -> dict[str, Any]:
    """The tool call: calc, calibrate, runs or setup."""
    spec = _spec(config_path)
    if not spec:
        return _not_configured()
    action = arguments.get("action", "calc")
    try:
        if action == "calc":
            out = _run(spec, calc_argv(arguments), timeout=600)
        elif action == "calibrate":
            out = _run(spec, calibrate_argv(arguments), timeout=120)
        elif action == "runs":
            return _runs(spec, arguments) | {"source": SOURCE}
        elif action == "setup":
            return _setup(spec) | {"source": SOURCE}
        else:
            raise ValueError("action is calc, calibrate, runs or setup")
    except ValueError as exc:
        return {"status": "refused", "note": str(exc)}
    measured = arguments.get("via") == "loop" or action == "calibrate"
    tier = "MEASURED, exploratory" if measured else "DERIVED (digital model)"
    return out | {"source": SOURCE, "tier": tier}
