"""The request ledger: what is on the go, in a small SQLite database (docs/hub-plan.md, section 1).

Every `ask` gets an id and a row: who asked (client, session, project), what it routed to, a
fingerprint of the question, the files it names, status, timings, providers, cost and problems.
A research run's row follows the run as its calls pass through the gateway, so `hub__result`
can return its answer after the call that started it has gone. The database survives restarts.
Per-session fallback decisions live here too.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

ACTIVE = ("queued", "running", "working", "needs_decision")
# Columns added after the first release, with their types: added to an older database on open.
LATER_COLUMNS = {"attached_to": "TEXT"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    client TEXT, session TEXT, session_inferred INTEGER, project TEXT, repo TEXT,
    route TEXT, specialist TEXT, fingerprint TEXT, question TEXT, touches TEXT,
    status TEXT NOT NULL,
    seconds REAL, providers TEXT, tokens INTEGER, cost REAL, problems TEXT,
    run_id TEXT, result TEXT
);
CREATE INDEX IF NOT EXISTS requests_run ON requests(run_id);
CREATE INDEX IF NOT EXISTS requests_status ON requests(status);
CREATE TABLE IF NOT EXISTS decisions (
    session TEXT NOT NULL, component TEXT NOT NULL, choice TEXT NOT NULL, decided REAL NOT NULL,
    PRIMARY KEY (session, component)
);
"""
QUESTION_CHARS = 300
JSON_FIELDS = ("touches", "providers", "problems", "result")


def fingerprint(question: str) -> str:
    """The normalised question's hash: equal for the same words in any case or spacing."""
    words = re.findall(r"[a-z0-9]+", question.lower())
    return hashlib.sha256(" ".join(words).encode()).hexdigest()[:12]


class Ledger:
    def __init__(self, path: Path, clock=time.time):
        self.path = path
        self.clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            have = {r[1] for r in self._db.execute("PRAGMA table_info(requests)")}
            for column, kind in LATER_COLUMNS.items():
                if column not in have:
                    self._db.execute(f"ALTER TABLE requests ADD COLUMN {column} {kind}")
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _exec(self, sql: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
            self._db.commit()
            return rows

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        for key in JSON_FIELDS:
            if out.get(key):
                out[key] = json.loads(out[key])
        out["session_inferred"] = bool(out.get("session_inferred"))
        return out

    # -- requests ----------------------------------------------------------------------------
    def start(
        self,
        question: str,
        *,
        client: str = "",
        session: str = "",
        session_inferred: bool = False,
        project: str = "",
        repo: str = "",
        route: str = "",
        specialist: str = "",
        touches: list[str] | None = None,
        rid: str = "",
    ) -> str:
        rid = rid or uuid.uuid4().hex[:12]
        now = self.clock()
        self._exec(
            "INSERT INTO requests (id, created, updated, client, session, session_inferred, "
            "project, repo, route, specialist, fingerprint, question, touches, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')",
            (
                rid, now, now, client, session, int(session_inferred), project, repo, route,
                specialist, fingerprint(question), question[:QUESTION_CHARS],
                json.dumps(touches or []),
            ),
        )  # fmt: skip
        return rid

    def finish(
        self,
        rid: str,
        status: str,
        *,
        seconds: float | None = None,
        problems: list[dict[str, Any]] | None = None,
        result: dict[str, Any] | None = None,
        run_id: str | None = None,
        providers: list[str] | None = None,
        tokens: int | None = None,
        cost: float | None = None,
        attached_to: str | None = None,
    ) -> None:
        sets = ["status = ?", "updated = ?"]
        args: list[Any] = [status, self.clock()]
        for key, value in (
            ("seconds", seconds),
            ("problems", None if problems is None else json.dumps(problems)),
            ("result", None if result is None else json.dumps(result)),
            ("run_id", run_id),
            ("providers", None if providers is None else json.dumps(providers)),
            ("tokens", tokens),
            ("cost", cost),
            ("attached_to", attached_to),
        ):
            if value is not None:
                sets.append(f"{key} = ?")
                args.append(value)
        self._exec(f"UPDATE requests SET {', '.join(sets)} WHERE id = ?", (*args, rid))

    def follow_run(self, run_id: str, reply: dict[str, Any]) -> str | None:
        """Update the rows of the requests on `run_id` (the one that started it, and any that
        attached to it) from a pipeline reply. Returns the starting request's id."""
        rows = self._exec("SELECT id FROM requests WHERE run_id = ? ORDER BY created", (run_id,))
        if not rows:
            return None
        for row in rows[1:]:
            self._follow(row["id"], reply)
        return self._follow(rows[0]["id"], reply)

    def _follow(self, rid: str, reply: dict[str, Any]) -> str:
        status = str(reply.get("status", ""))
        if reply.get("error"):
            self.finish(rid, "failed", problems=[{"what_failed": str(reply["error"])[:500]}])
        elif status == "done":
            cost = reply.get("cost") or {}
            cost = cost if isinstance(cost, dict) else {}
            self.finish(
                rid,
                "degraded" if cost.get("ladder") else "done",
                result=reply,
                cost=cost.get("cost_usd"),
                providers=cost.get("providers"),
                tokens=cost.get("tokens"),
            )
        elif status:
            self.finish(rid, "working")
        return rid

    def research_on_the_go(self, exclude: str = "") -> list[dict[str, Any]]:
        """Research requests queued or running, oldest first, that a new one could attach to."""
        return [
            r for r in self.active() if r.get("route") == "research" and r["id"] != exclude
            and not r.get("attached_to") and r["status"] != "needs_decision"
        ]  # fmt: skip

    def typical_seconds(self, route: str, default: float) -> float:
        """The median time from start to finish of recent finished requests on `route`."""
        rows = self._exec(
            "SELECT updated - created AS s FROM requests WHERE route = ? AND status IN "
            "('done', 'degraded') ORDER BY created DESC LIMIT 20",
            (route,),
        )
        values = sorted(float(r["s"]) for r in rows)
        return values[len(values) // 2] if values else default

    def get(self, rid: str) -> dict[str, Any] | None:
        rows = self._exec("SELECT * FROM requests WHERE id = ? OR run_id = ?", (rid, rid))
        return self._row(rows[0]) if rows else None

    def active(self) -> list[dict[str, Any]]:
        marks = ", ".join("?" for _ in ACTIVE)
        rows = self._exec(
            f"SELECT * FROM requests WHERE status IN ({marks}) ORDER BY created", ACTIVE
        )
        return [self._row(r) for r in rows]

    def recent(self, seconds: float, **where: str) -> list[dict[str, Any]]:
        clauses = ["created >= ?"] + [f"{k} = ?" for k in where]
        rows = self._exec(
            f"SELECT * FROM requests WHERE {' AND '.join(clauses)} ORDER BY created DESC",
            (self.clock() - seconds, *where.values()),
        )
        return [self._row(r) for r in rows]

    def board(self, per_project: int = 5) -> dict[str, Any]:
        """Per project: counts by status, cost, and the latest requests."""
        rows = [self._row(r) for r in self._exec("SELECT * FROM requests ORDER BY created DESC")]
        projects: dict[str, dict[str, Any]] = {}
        for row in rows:
            name = row.get("project") or "(no project)"
            entry = projects.setdefault(
                name, {"counts": {}, "cost_usd": 0.0, "sessions": set(), "latest": []}
            )
            entry["counts"][row["status"]] = entry["counts"].get(row["status"], 0) + 1
            entry["cost_usd"] += row.get("cost") or 0.0
            entry["sessions"].add(row.get("session") or "")
            if len(entry["latest"]) < per_project:
                entry["latest"].append(
                    {k: row.get(k) for k in ("id", "status", "route", "question", "session")}
                )
        for entry in projects.values():
            entry["sessions"] = len(entry["sessions"] - {""})
            entry["cost_usd"] = round(entry["cost_usd"], 4)
        return projects

    def spend(self, seconds: float) -> float:
        rows = self._exec(
            "SELECT COALESCE(SUM(cost), 0) AS c FROM requests WHERE created >= ?",
            (self.clock() - seconds,),
        )
        return round(float(rows[0]["c"]), 4)

    # -- decisions ---------------------------------------------------------------------------
    def decisions(self, session: str) -> dict[str, str]:
        rows = self._exec("SELECT component, choice FROM decisions WHERE session = ?", (session,))
        return {r["component"]: r["choice"] for r in rows}

    def decide(self, session: str, component: str, choice: str) -> None:
        self._exec(
            "INSERT INTO decisions (session, component, choice, decided) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session, component) DO UPDATE SET choice = excluded.choice, "
            "decided = excluded.decided",
            (session, component, choice, self.clock()),
        )
