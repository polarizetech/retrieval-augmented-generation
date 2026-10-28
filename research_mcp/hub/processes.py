"""Stale processes: reported on the board, never killed by the hub (docs/hub-plan.md, section 3).

Three kinds are worth knowing about: a stdio gateway (it holds its own copy of the paper library,
1.7 to 3 GB; `--connect` would make it a small bridge instead), a gateway or bridge started before
the code it runs last changed (an old version, until its session restarts), and an upstream server
whose gateway has gone (an orphan, reparented to launchd/init).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from research_pipeline.config import ROOT

UPSTREAM_NAMES = ("paper-fetch-mcp", "research-pipeline-mcp", "dataset-fetch-mcp", "patent_mcp")
CODE_DIRS = ("research_mcp", "research_pipeline")


def ps() -> str:
    out = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,lstart=,command="],  # noqa: S607 - fixed argv, no shell
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    return out.stdout


def code_changed(root: Path = ROOT, *, bridge: bool = False) -> float:
    """When the gateway's code last changed on disk (newest .py mtime), as a Unix time. A bridge
    only proxies, so for it only the bridge's own file counts."""
    if bridge:
        return (root / "research_mcp" / "bridge.py").stat().st_mtime
    newest = 0.0
    for folder in CODE_DIRS:
        for path in (root / folder).rglob("*.py"):
            newest = max(newest, path.stat().st_mtime)
    return newest


def parse(listing: str) -> list[dict[str, Any]]:
    rows = []
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) < 8 or not parts[0].isdigit():
            continue
        try:
            started = datetime.strptime(" ".join(parts[2:7]), "%a %b %d %H:%M:%S %Y").timestamp()
        except ValueError:
            continue
        rows.append(
            {
                "pid": int(parts[0]),
                "ppid": int(parts[1]),
                "started": started,
                "command": " ".join(parts[7:]),
            }
        )
    return rows


def stale(
    listing: Callable[[], str] = ps,
    changed: Callable[..., float] = code_changed,
    own_pid: int | None = None,
) -> list[dict[str, Any]]:
    """Gateways and upstreams worth a look, each with why. Wrappers (the app's helper that
    launches a gateway) are skipped; their child is listed."""
    try:
        rows = parse(listing())
        newest = changed()
        newest_bridge = changed(bridge=True)
    except (OSError, subprocess.SubprocessError):
        return []
    by_pid = {r["pid"]: r for r in rows}
    out = []
    for row in rows:
        cmd = row["command"]
        if row["pid"] == own_pid or "disclaimer" in cmd.split()[0]:
            continue
        why: list[str] = []
        if "research-rag-gateway" in cmd:
            if "--connect" in cmd:
                kind = "bridge"
            elif "--http" in cmd:
                kind = "http gateway"
            else:
                kind = "stdio gateway"
                why.append("holds its own copy of the paper library; --connect would share one")
            if row["started"] < (newest_bridge if kind == "bridge" else newest):
                why.append("started before the gateway's code last changed: an old version")
        elif any(name in cmd for name in UPSTREAM_NAMES):
            kind = "upstream"
            parent = by_pid.get(row["ppid"])
            if row["ppid"] == 1 or parent is None:
                why.append("its gateway has gone: an orphan")
        else:
            continue
        if why:
            out.append(
                {
                    "pid": row["pid"],
                    "kind": kind,
                    "started": datetime.fromtimestamp(row["started"]).isoformat(timespec="minutes"),
                    "why": why,
                }
            )
    return out
