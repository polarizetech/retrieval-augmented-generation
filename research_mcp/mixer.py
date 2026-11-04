"""The sign mixer: what a hard limiter makes from two tones. Deterministic, no model, no numpy.

A port of the nonlinear-mixer calculator in polarizetech/polarity-calculator
(calculators/nonlinear-mixer: Julia reference SignMixer.jl, Python sign_mixer.py, checked there
against a closed form, a direct 2-D integral, sampled audio and keyed streams under five keys).

    x(t) = a1 cos θ1 + a2 cos θ2,  y = sign(x) = Σ c(m, n) e^{i(mθ1 + nθ2)}

Lines sit only at |m f1 + n f2| with m + n odd, because sign is odd. With θ1 integrated out
exactly (sign(cos θ1 + b) is +1 on |θ1| < α = arccos(−b)), each coefficient is a smooth periodic
1-D integral,

    c(m, n) = (1/4π²) ∫ A_m(α(θ2)) cos(nθ2) dθ2,   A_m = 4 sin(mα)/m (m ≠ 0), 4α − 2π (m = 0),

which the trapezoid rule converges on exponentially for r < 1. The keyed stream
p = sign(sign(x) + k), k ~ N(0, 1), has E[p | x] = erf(1/√2) sign(x): the same lines scaled
by g = 0.6827.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from research_pipeline.upstream import spec_for

SOURCE = (
    "polarizetech/polarity-calculator calculators/nonlinear-mixer "
    "(Julia reference SignMixer.jl; this is a pure-Python port)"
)
MAX_ORDER = 40
KEYED_GAIN = math.erf(1 / math.sqrt(2))


def _table(order: int, r: float) -> list[list[float]]:
    """c[|m|][|n|] for 0 ≤ |m|, |n| ≤ order, strong tone first (0 ≤ r ≤ 1)."""
    k = max(1024, 16 * (order + 1))
    theta = [2 * math.pi * j / k for j in range(k)]
    alpha = [math.acos(max(-1.0, min(1.0, -r * math.cos(t)))) for t in theta]
    cos_n = [[math.cos(n * t) for t in theta] for n in range(order + 1)]
    out = [[0.0] * (order + 1) for _ in range(order + 1)]
    for m in range(order + 1):
        a = (
            [4 * al - 2 * math.pi for al in alpha]
            if m == 0
            else [4 * math.sin(m * al) / m for al in alpha]
        )
        for n in range(1 - m % 2, order + 1, 2):  # m + n odd only
            out[m][n] = sum(x * c for x, c in zip(a, cos_n[n], strict=True)) / (2 * math.pi * k)
    return out


def coefficient(m: int, n: int, r: float) -> float:
    """Coefficient of e^{i(mθ1 + nθ2)} in sign(cos θ1 + r cos θ2), r ≥ 0."""
    if r < 0:
        raise ValueError("r is a ratio of amplitudes, r >= 0")
    if (m + n) % 2 == 0:
        return 0.0
    if r > 1:
        return coefficient(n, m, 1 / r)
    return _table(max(abs(m), abs(n)), r)[abs(m)][abs(n)]


def sign_mix(  # noqa: PLR0917
    f1: float,
    a1: float,
    f2: float,
    a2: float,
    order: int = 15,
    rs: float | None = None,
    keyed: bool = False,
    limit: int = 20,
) -> dict[str, Any]:
    """The strongest lines of sign(a1 cos 2πf1t + a2 cos 2πf2t), with the terms that build each.

    rs folds every product into [0, rs/2] as sampling does. Phases are taken as zero, so lines that
    two terms share add in phase; with other phases their sum can differ (each term is listed).
    """
    if f1 <= 0 or f2 <= 0:
        raise ValueError("frequencies must be positive")
    if a1 <= 0 or a2 <= 0:
        raise ValueError("amplitudes must be positive")
    if not 1 <= order <= MAX_ORDER:
        raise ValueError(f"order must be between 1 and {MAX_ORDER}")
    if rs is not None and rs <= 2 * max(f1, f2):
        raise ValueError("rs must exceed twice the higher tone")
    swap = a2 > a1
    r = a1 / a2 if swap else a2 / a1
    table = _table(order, r)
    g = KEYED_GAIN if keyed else 1.0
    acc: dict[float, float] = {}
    terms: dict[float, list[tuple[int, int, float]]] = {}
    for m in range(-order, order + 1):
        for n in range(-order, order + 1):
            if (m + n) % 2 == 0:
                continue
            f = m * f1 + n * f2
            if rs is None:
                if f <= 0:
                    continue
            else:
                f %= rs
                if f > rs / 2:
                    continue
            f = round(f, 9)
            c = table[abs(n)][abs(m)] if swap else table[abs(m)][abs(n)]
            if c == 0.0:
                continue
            acc[f] = acc.get(f, 0.0) + c
            terms.setdefault(f, []).append((m, n, c))
    lines = []
    for f, z in acc.items():
        edge = f == 0 or (rs is not None and f == rs / 2)
        amp = g * (abs(z) if edge else 2 * abs(z))
        ts = sorted(terms[f], key=lambda t: (abs(t[0]) + abs(t[1]), -abs(t[2])))
        lines.append(
            {
                "f_hz": f,
                "amplitude": round(amp, 9),
                "lowest_order": abs(ts[0][0]) + abs(ts[0][1]),
                "terms": [f"{m:+d}·f1 {n:+d}·f2" for m, n, _ in ts[:4]],
            }
        )
    lines.sort(key=lambda ln: -ln["amplitude"])
    strong, weak = (f2, f1) if swap else (f1, f2)
    return {
        "model": "y = sign(a1 cos 2πf1t + a2 cos 2πf2t)"
        + (", keyed: p = sign(sign(x) + k), k ~ N(0,1)" if keyed else ""),
        "inputs": {
            "f1": f1,
            "a1": a1,
            "f2": f2,
            "a2": a2,
            "order": order,
            "rs": rs,
            "keyed": keyed,
        },
        "lines": lines[:limit],
        "lines_total": len(lines),
        "capture": {
            "weak_over_strong_in": r,
            "weak_over_strong_out": round(abs(table[0][1] / table[1][0]), 9),
            "ratio": round(abs(table[0][1] / table[1][0]) / r, 9),
            "note": (
                f"the {weak:g} Hz tone comes out further below the {strong:g} Hz tone than it "
                "went in; the ratio tends to 1/2 (6 dB) as r → 0"
            ),
        },
        "rules": [
            "sign is odd: only m + n odd products appear; f1 + f2, f2 − f1, 2f1 and 2f2 "
            "appear only "
            "where a higher odd-order product lands on them",
            "keyed sign scales every line by erf(1/√2) = 0.6827 over a white floor; "
            "the key comes after "
            "the sign, so it keeps the line ratios",
            "the lattice is truncated at |m|, |n| ≤ order; "
            "sampled sign also folds the tail beyond it",
        ],
        "keyed_gain": KEYED_GAIN,
        "derived": True,
        "source": SOURCE,
    }


# ---------- the random run: the calculator itself, with a physical random key ----------

SECTION = "calculators"
NAME = "sign_mixer"


def random_run(
    config_path: Path | None,
    *,
    f1: float,
    a1: float,
    f2: float,
    a2: float,
    settle: bool = True,
    rs: float | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Run the calculator's random mix (polarity-calculator random_mix.py --json) as configured.

    Every random number in it (the key k and the added noise ξ) comes from a fresh entropy-fetch
    physical mix, and the edge-of-chaos gain (settle) is on unless settle=False. The command comes
    from the gateway config, `calculators.sign_mixer`; the arguments are numbers this function
    formats, passed as an argument list (no shell).
    """
    spec = spec_for(config_path, NAME, SECTION) if config_path is not None else None
    if not spec:
        return {
            "status": "not configured",
            "note": (
                f"add `{SECTION}.{NAME}` to the gateway config (see "
                "config/mcp-gateway.example.json) to run the mixer with a physical random key; "
                "until then only the prediction is returned"
            ),
        }
    argv = [spec["command"], *(str(v) for v in spec.get("args", []))]
    argv += [
        "--f1",
        repr(float(f1)),
        "--a1",
        repr(float(a1)),
        "--f2",
        repr(float(f2)),
        "--a2",
        repr(float(a2)),
    ]
    if rs is not None:
        argv += ["--rs", repr(float(rs))]
    if not settle:
        argv.append("--no-settle")
    env = {**os.environ, **{str(k): str(v) for k, v in spec.get("env", {}).items()}}
    try:
        done = subprocess.run(  # noqa: S603 (the configured command; arguments are numbers)
            argv,
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
        return {"status": "failed", "note": f"{NAME} exited {done.returncode}: {tail[0]}"}
    try:
        out = json.loads(done.stdout)
    except ValueError:
        return {"status": "failed", "note": f"{NAME} did not print JSON"}
    keep = (
        "passed",
        "key_source",
        "sources",
        "key",
        "noise",
        "lines",
        "lines_pass",
        "twin_pass",
        "twin_band",
        "settle",
        "run",
        "made",
    )
    return {"status": "done", **{k: out[k] for k in keep if k in out}}


def sign_mix_step(config_path: Path | None, arguments: dict[str, Any]) -> dict[str, Any]:
    """The step: the derived line table, then (by default) the calculator's random run."""
    f1, f2 = float(arguments["f1"]), float(arguments["f2"])
    a1, a2 = float(arguments.get("a1", 1.0)), float(arguments.get("a2", 1.0))
    rs = arguments.get("rs")
    rs = float(rs) if rs is not None else None
    out = {
        "prediction": sign_mix(
            f1,
            a1,
            f2,
            a2,
            order=int(arguments.get("order", 15)),
            rs=rs,
            keyed=bool(arguments.get("keyed", True)),
            limit=int(arguments.get("limit", 20)),
        )
    }
    if arguments.get("random_run", True):
        out["random_run"] = random_run(
            config_path,
            f1=f1,
            a1=a1,
            f2=f2,
            a2=a2,
            settle=bool(arguments.get("settle", True)),
            rs=rs,
        )
    return out


FREQ = re.compile(r"(\d+(?:\.\d+)?)\s*(k?hz)\b", re.IGNORECASE)
AMP = re.compile(r"\b(a[12])\s*=\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_tones(question: str) -> dict[str, float] | None:
    """Two tones from plain words: '120 Hz and 210 Hz', optionally 'a2=0.4'. None without two."""
    freqs = [
        float(v) * (1000 if unit.lower() == "khz" else 1) for v, unit in FREQ.findall(question)
    ]
    if len(freqs) < 2:
        return None
    out: dict[str, float] = {"f1": freqs[0], "f2": freqs[1]}
    for name, value in AMP.findall(question):
        out[name.lower()] = float(value)
    return out


# ---------- audio: audio_mix.py (files on disk or in Spaces), and the run ledger ----------

AUDIO_NAME = "audio_mixer"
AUDIO_EXT = (".wav", ".flac", ".aif", ".aiff", ".mp3", ".m4a", ".ogg", ".caf")
SPACES_REF = re.compile(r"^spaces/[A-Za-z0-9._~-]+@[0-9a-f]{8,64}$")
KINDS = ("diode", "diode_r", "tanh", "poly", "sign")


def _safe(value: str, what: str) -> str:
    """A value passed as one argument: never an option, never a parent-directory hop."""
    if not value or value.startswith("-") or ".." in Path(value).parts or "\x00" in value:
        raise ValueError(f"{what} is not usable: {value!r}")
    return value


def _audio_argv(arguments: dict[str, Any]) -> list[str]:
    action = arguments.get("action", "run")
    if action == "list":
        out = ["list", "--limit", str(int(arguments.get("limit", 50)))]
        if arguments.get("match"):
            out += ["--match", _safe(str(arguments["match"]), "match")]
        return out + (["--refresh"] if arguments.get("refresh") else [])
    if action == "ledger":
        out = ["ledger", "--last", str(int(arguments.get("last", 20)))]
        if arguments.get("match"):
            out += ["--match", _safe(str(arguments["match"]), "match")]
        return out
    if action != "run":
        raise ValueError("action is list, run or ledger")
    out = ["run"]
    if arguments.get("spaces"):
        ref = str(arguments["spaces"])
        if not SPACES_REF.match(ref):
            raise ValueError("spaces must be a pinned ref: spaces/<accession>@<snapshot>")
        path = _safe(str(arguments.get("path", "")), "path")
        out += ["--spaces", ref, "--path", path]
    elif arguments.get("file"):
        file = Path(str(arguments["file"])).expanduser()
        if not file.is_absolute() or file.suffix.lower() not in AUDIO_EXT or not file.is_file():
            raise ValueError(
                "file must be an existing audio file, given by its full path ("
                + ", ".join(AUDIO_EXT)
                + ")"
            )
        out += ["--file", str(file)]
    else:
        raise ValueError("run needs spaces + path, or file")
    mixer_kind = arguments.get("mixer", "sign")
    if mixer_kind not in ("sign", "voltage"):
        raise ValueError("mixer is sign or voltage")
    out += ["--mixer", mixer_kind]
    if arguments.get("start") is not None:
        out += ["--start", repr(float(arguments["start"]))]
    seconds = float(arguments.get("seconds", 30))
    if not 0 <= seconds <= 600:
        raise ValueError("seconds is 0 (whole file, up to 10 min) to 600")
    out += ["--seconds", repr(seconds), "--channel", str(int(arguments.get("channel", -1)))]
    if mixer_kind == "voltage":
        kind = arguments.get("kind", "diode_r")
        if kind not in KINDS:
            raise ValueError(f"kind is one of {', '.join(KINDS)}")
        out += [
            "--kind",
            kind,
            "--full-scale",
            repr(float(arguments.get("full_scale", 0.6))),
            "--sigma",
            repr(float(arguments.get("sigma", 0.02))),
        ]
        for name, value in (arguments.get("params") or {}).items():
            if name not in ("Is", "n", "T", "R", "g", "v0", "a1", "a2", "a3"):
                raise ValueError(f"unknown parameter {name}")
            out += ["--param", f"{name}={float(value)!r}"]
    if not arguments.get("settle", True):
        out.append("--no-settle")
    return out


def audio_step(config_path: Path | None, arguments: dict[str, Any]) -> dict[str, Any]:
    """List Spaces audio, run a file through a mixer, or read the run ledger (audio_mix.py)."""
    spec = spec_for(config_path, AUDIO_NAME, SECTION) if config_path is not None else None
    if not spec:
        return {
            "status": "not configured",
            "note": f"add `{SECTION}.{AUDIO_NAME}` to the gateway config "
            "(see config/mcp-gateway.example.json)",
        }
    argv = [
        spec["command"],
        *(str(v) for v in spec.get("args", [])),
        *_audio_argv(arguments),
        "--json",
    ]
    env = {**os.environ, **{str(k): str(v) for k, v in spec.get("env", {}).items()}}
    try:
        done = subprocess.run(  # noqa: S603 (the configured command; arguments validated above)
            argv,
            cwd=spec.get("cwd"),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "note": f"{AUDIO_NAME}: {exc}"}
    if done.returncode != 0:
        tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
        return {"status": "failed", "note": f"{AUDIO_NAME} exited {done.returncode}: {tail[0]}"}
    try:
        out = json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"status": "failed", "note": f"{AUDIO_NAME} did not print JSON"}
    action = arguments.get("action", "run")
    if action == "list":
        return {"status": "done", "files": out}
    if action == "ledger":
        return {"status": "done", "table": out["table"], "entries": out["entries"]}
    chans = [
        {
            k: v
            for k, v in ch.items()
            if k in ("channel", "checks", "settle", "silent", "residual_mean", "residual_ok")
        }
        | {
            name: ch[name]["peaks"][:8]
            for name in ("keyed_vs_twin", "settled_vs_twin", "output_vs_input")
            if name in ch
        }
        for ch in out["channels"]
    ]
    return {
        "status": "done",
        "summary": out["summary"],
        "inputs": out["inputs"],
        "settings": out["settings"],
        "channels": chans,
        "run": out["run"],
        "note": (
            "logged in the calculator's run ledger (action=ledger); audio and outputs stay local"
        ),
    }
