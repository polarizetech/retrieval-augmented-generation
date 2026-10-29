"""Persistent pipeline sessions and compact, reviewable research capsules."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
create table if not exists sessions (
    run_id text primary key,
    external_session_id text,
    client text,
    project text,
    strictness text not null,
    question text not null,
    status text not null,
    started_at real not null,
    ended_at real,
    run_dir text,
    answer_sha256 text,
    capsule_path text,
    error text
);
create table if not exists session_events (
    id integer primary key autoincrement,
    run_id text not null,
    at real not null,
    kind text not null,
    payload text not null
);
"""


class SessionStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(SCHEMA)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("pragma journal_mode=wal")
        return db

    def start(
        self,
        run_id: str,
        question: str,
        *,
        client: str,
        project: str,
        strictness: str,
        external_session_id: str,
        route: dict[str, Any],
    ) -> None:
        now = time.time()
        with self._db() as db:
            db.execute(
                "insert or replace into sessions values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    external_session_id or None,
                    client or None,
                    project or None,
                    strictness,
                    question,
                    "running",
                    now,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            )
            db.execute(
                "insert into session_events(run_id,at,kind,payload) values (?,?,?,?)",
                (run_id, now, "started", json.dumps({"route": route}, ensure_ascii=False)),
            )

    def fail(self, run_id: str, error: str) -> None:
        with self._db() as db:
            db.execute(
                "update sessions set status='failed', ended_at=?, error=? where run_id=?",
                (time.time(), error[:1000], run_id),
            )

    def finish(self, run_id: str, log: dict[str, Any], metadata: dict[str, Any]) -> Path:
        run_dir = Path(log["run_dir"])
        capsule = capsule_markdown(run_id, log, metadata)
        capsule_path = run_dir / "capsule.md"
        capsule_path.write_text(capsule)
        session_path = run_dir / "session.json"
        session = {
            "schema_version": 1,
            "run_id": run_id,
            **metadata,
            "run_dir": str(run_dir),
            "answer_sha256": log["answer_sha256"],
            "capsule_path": str(capsule_path),
        }
        session_path.write_text(json.dumps(session, indent=2, ensure_ascii=False))
        with self._db() as db:
            db.execute(
                "update sessions set status='done', ended_at=?, run_dir=?, answer_sha256=?, "
                "capsule_path=? where run_id=?",
                (time.time(), str(run_dir), log["answer_sha256"], str(capsule_path), run_id),
            )
            db.execute(
                "insert into session_events(run_id,at,kind,payload) values (?,?,?,?)",
                (
                    run_id,
                    time.time(),
                    "finished",
                    json.dumps(
                        {"run_dir": str(run_dir), "answer_sha256": log["answer_sha256"]},
                        ensure_ascii=False,
                    ),
                ),
            )
        return capsule_path


def _bottom_line(answer: str) -> str:
    marker = "## Bottom line\n"
    if marker not in answer:
        return ""
    text = answer.split(marker, 1)[1].split("\n## ", 1)[0]
    return text.strip()


def capsule_markdown(run_id: str, log: dict[str, Any], metadata: dict[str, Any]) -> str:
    kept = [c for c in log.get("claims", []) if c.get("verdict") in ("SUPPORTED", "DISPUTED")]
    evidence_ids = sorted({eid for claim in kept for eid in claim.get("supported_by", [])})
    companions = log.get("companions", {}).get("results", {})
    route = metadata.get("agent_route") or {}
    lines = [
        f"# Research session: {log['question']}",
        "",
        "> REVIEW REQUIRED. This is a verified pipeline artifact, not accepted corpus content. ",
        "> Promote it only through the research repository's proposal/PR gate.",
        "",
        "## Session",
        "",
        f"- Run: `{run_id}`",
        f"- Client: `{metadata.get('client') or 'unknown'}`",
        f"- External session: `{metadata.get('external_session_id') or 'not supplied'}`",
        f"- Strictness: `{metadata.get('strictness') or 'grounded'}`",
        f"- Suggested project: `{metadata.get('project') or 'unassigned'}`",
        f"- Agent profiles: {', '.join(route.get('roles', [])) or 'researcher'}",
        f"- Specialist: `{route.get('specialist') or 'none'}`",
        f"- Profile commit: `{route.get('commit') or 'unavailable'}`",
        f"- Answer SHA-256: `{log.get('answer_sha256', '')}`",
        "",
        "## Bottom line",
        "",
        log.get("bottom_line")
        or _bottom_line(log.get("answer", ""))
        or "No verified bottom line was produced.",
        "",
        "## Promotion candidates",
        "",
    ]
    if kept:
        for claim in kept:
            ids = claim.get("supported_by") or claim.get("evidence_ids") or []
            lines.append(
                f"- [{claim.get('verdict')}] {claim.get('text')} "
                f"(evidence: {', '.join(f'`{eid}`' for eid in ids) or 'none'})"
            )
    else:
        lines.append("- None. No claim survived verification.")
    lines += ["", "## Companion records", ""]
    if companions:
        for name, result in sorted(companions.items()):
            count = len(result.get("items", [])) if isinstance(result, dict) else 0
            lines.append(f"- `{name}`: {count} listed item(s); see `run.json` for the records.")
    else:
        lines.append("- No dataset or patent companion was used.")
    lines += [
        "",
        "## Open questions and limits",
        "",
        *(f"- {item}" for item in log.get("notes", [])),
        *(
            f"- Insufficient evidence for `{item}`."
            for item in log.get("insufficient_subquestions", [])
        ),
        "",
        "## Provenance",
        "",
        f"- Full answer: `{Path(log['run_dir']) / 'answer.md'}`",
        f"- Complete machine log: `{Path(log['run_dir']) / 'run.json'}`",
        f"- Evidence IDs promoted here: {', '.join(f'`{eid}`' for eid in evidence_ids) or 'none'}",
        "",
    ]
    return "\n".join(lines)
