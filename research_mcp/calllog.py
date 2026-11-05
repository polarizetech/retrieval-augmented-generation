"""Every tool call through the gateway, recorded (docs/call-log.md).

Each call is written at once to a small local SQLite file, then copied to a hosted Postgres
database (Neon today) when CALL_LOG_DATABASE_URL is set. The local copy is only an outbox:
rows the database has are pruned after CALL_LOG_LOCAL_DAYS. Without a database URL nothing is
pruned, so no record is lost.

A record keeps who called (client, session, project), the tool, its arguments, the full
response, and its references: the research runs, saved reports, requests, DOIs and URLs the
call mentions. References make backlinks: the page for a run lists every call that mentions it.

The schema is plain tables and JSON columns with no extensions, so moving the database to
another Postgres host is one pg_dump and pg_restore.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, TextContent

log = logging.getLogger(__name__)

LOCAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id TEXT PRIMARY KEY,
    created REAL NOT NULL,
    client TEXT, session TEXT, project TEXT,
    tool TEXT NOT NULL,
    arguments TEXT,
    status TEXT NOT NULL,
    seconds REAL,
    summary TEXT,
    response TEXT,
    synced INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS calls_created ON calls(created);
CREATE INDEX IF NOT EXISTS calls_synced ON calls(synced);
CREATE TABLE IF NOT EXISTS call_refs (
    call_id TEXT NOT NULL, kind TEXT NOT NULL, ref TEXT NOT NULL,
    PRIMARY KEY (call_id, kind, ref)
);
CREATE INDEX IF NOT EXISTS call_refs_ref ON call_refs(kind, ref);
"""
REMOTE_SCHEMA = """
CREATE TABLE IF NOT EXISTS mcp_calls (
    id TEXT PRIMARY KEY,
    created TIMESTAMPTZ NOT NULL,
    client TEXT, session TEXT, project TEXT,
    tool TEXT NOT NULL,
    arguments JSONB,
    status TEXT NOT NULL,
    seconds DOUBLE PRECISION,
    summary TEXT,
    response JSONB
);
CREATE INDEX IF NOT EXISTS mcp_calls_created ON mcp_calls(created DESC);
CREATE INDEX IF NOT EXISTS mcp_calls_tool ON mcp_calls(tool);
CREATE TABLE IF NOT EXISTS mcp_call_refs (
    call_id TEXT NOT NULL REFERENCES mcp_calls(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    ref TEXT NOT NULL,
    PRIMARY KEY (call_id, kind, ref)
);
CREATE INDEX IF NOT EXISTS mcp_call_refs_ref ON mcp_call_refs(kind, ref);
"""
# Status and housekeeping tools: recorded, but their replies get no links.
QUIET_TOOLS = {"gateway_status", "hub__status", "hub__board", "hub__usage"}
SUMMARY_CHARS = 400
PUSH_BATCH = 200
RETRY_SECONDS = 60.0
URL = re.compile(r"https://(?:github\.com|doi\.org)/[^\s\"'<>)\]]+")
DOI = re.compile(r"^10\.\d{4,9}/\S+$")


# -- what a call refers to ---------------------------------------------------------------------
def response_json(result: Any) -> Any:
    """The response as plain JSON: a dict as it is, a CallToolResult as its fields."""
    if isinstance(result, CallToolResult):
        return result.model_dump(mode="json", exclude_none=True)
    return result


def _payload(result: Any) -> Any:
    """The response's data: a dict, or the JSON in a CallToolResult's first text block."""
    if isinstance(result, CallToolResult):
        content = result.content or []
        text = content[0].text if content and isinstance(content[0], TextContent) else ""
        try:
            return json.loads(text) if text.lstrip().startswith(("{", "[")) else text
        except json.JSONDecodeError:
            return text
    return result


def references(tool: str, arguments: dict[str, Any], result: Any) -> set[tuple[str, str]]:
    """(kind, ref) pairs for what a call mentions: run (a run folder's name), run_id, request,
    doi and url."""
    found: set[tuple[str, str]] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and item:
                    if key == "run_dir":
                        found.add(("run", Path(item).name))
                    elif key == "run_id":
                        found.add(("run_id", item))
                    elif key in ("request_id", "attached_to"):
                        found.add(("request", item))
                    elif key == "doi" and DOI.match(item):
                        found.add(("doi", item.lower()))
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            found.update(("url", u.rstrip(".,;")) for u in URL.findall(value))

    walk(arguments)
    walk(_payload(result))
    if tool == "hub__result" and arguments.get("id"):
        found.add(("request", str(arguments["id"])))
    return found


def summary(result: Any) -> str:
    """A line or two for lists: the reply's relay when it has one, else its opening text."""
    data = _payload(result)
    if isinstance(data, dict):
        text = str(data.get("relay") or data.get("error") or "")
        if not text:
            text = json.dumps(data, ensure_ascii=False)
    else:
        text = str(data)
    return " ".join(text.split())[:SUMMARY_CHARS]


def failed(result: Any) -> bool:
    if isinstance(result, CallToolResult):
        return bool(result.isError)
    return isinstance(result, dict) and bool(result.get("error"))


def _clean(value: Any) -> Any:
    """Postgres JSONB refuses the NUL character; drop it from every string."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _dumps(value: Any) -> str:
    return json.dumps(_clean(value), ensure_ascii=False, default=str)


# -- links --------------------------------------------------------------------------------------
class Links:
    """URLs of the record pages, under RECORDS_URL (a tailnet-only address)."""

    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def record(self, call_id: str) -> str:
        return f"{self.base}/records/{call_id}"

    def run(self, key: str) -> str:
        return f"{self.base}/records/run/{key}"

    def request(self, rid: str) -> str:
        return f"{self.base}/records/request/{rid}"

    def for_call(self, call_id: str, refs: Iterable[tuple[str, str]]) -> dict[str, Any]:
        refs = sorted(set(refs))
        runs = [r for k, r in refs if k == "run"] or [r for k, r in refs if k == "run_id"]
        out: dict[str, Any] = {"record": self.record(call_id)}
        if runs:
            out["research"] = [self.run(r) for r in runs]
        requests = [r for k, r in refs if k == "request"]
        if requests:
            out["requests"] = [self.request(r) for r in requests]
        return out

    @staticmethod
    def markdown(links: dict[str, Any]) -> str:
        lines = [f"- [Record of this call]({links['record']})"]
        lines += [f"- [Research doc and its references]({u})" for u in links.get("research", [])]
        lines += [f"- [Request history]({u})" for u in links.get("requests", [])]
        return "**Links**\n" + "\n".join(lines)


def with_links(result: Any, links: dict[str, Any]) -> Any:
    """The reply with its links: a `links` field on a dict (and on its relay), else on the JSON
    in a CallToolResult's first text block, else as one more text block."""
    block = Links.markdown(links)
    if isinstance(result, dict):
        out = {**result, "links": links}
        if isinstance(out.get("relay"), str):
            out["relay"] = f"{out['relay']}\n\n{block}"
        return out
    if isinstance(result, CallToolResult):
        content = list(result.content or [])
        data = _payload(result)
        if isinstance(data, dict) and content:
            data = {**data, "links": links}
            if isinstance(data.get("relay"), str):
                data["relay"] = f"{data['relay']}\n\n{block}"
            content[0] = TextContent(type="text", text=json.dumps(data, ensure_ascii=False))
        else:
            content.append(TextContent(type="text", text=block))
        return CallToolResult(
            isError=result.isError, content=content, structuredContent=result.structuredContent
        )
    return result


# -- storage ------------------------------------------------------------------------------------
class CallLog:
    def __init__(
        self,
        path: Path,
        links: Links,
        *,
        database_url: str = "",
        local_days: float = 7.0,
        connect: Any = None,
        clock=time.time,
    ):
        self.path = path
        self.links = links
        self.database_url = database_url
        self.local_days = local_days
        self.clock = clock
        self._connect = connect
        self._remote: Any = None
        self._schema_ready = False
        self._retry_at = 0.0
        self._pushing = False
        self.last_error = ""
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(LOCAL_SCHEMA)
            self._db.commit()

    @classmethod
    def from_env(cls, data_dir: Path, default_base: str) -> CallLog:
        path = Path(os.environ.get("CALL_LOG_LOCAL") or data_dir / "calls.sqlite")
        base = os.environ.get("RECORDS_URL", "").strip() or default_base
        return cls(
            path,
            Links(base),
            database_url=os.environ.get("CALL_LOG_DATABASE_URL", "").strip(),
            local_days=float(os.environ.get("CALL_LOG_LOCAL_DAYS", "7")),
        )

    @property
    def remote(self) -> bool:
        return bool(self.database_url or self._connect)

    def close(self) -> None:
        with self._lock:
            self._db.close()
            if self._remote is not None:
                with contextlib.suppress(Exception):
                    self._remote.close()

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    # -- writing --------------------------------------------------------------------------------
    def record(
        self,
        call_id: str,
        tool: str,
        arguments: dict[str, Any],
        result: Any,
        *,
        seconds: float,
        who: dict[str, str] | None = None,
    ) -> Any:
        """Store the call and return the reply with its links. Never raises: a broken log
        must not break the tool call."""
        try:
            refs = references(tool, arguments, result)
            who = who or {}
            with self._lock:
                self._db.execute(
                    "INSERT OR REPLACE INTO calls (id, created, client, session, project, tool,"
                    " arguments, status, seconds, summary, response, synced)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                    (
                        call_id,
                        self.clock(),
                        who.get("client", ""),
                        who.get("session", ""),
                        who.get("project", ""),
                        tool,
                        _dumps(arguments),
                        "error" if failed(result) else "ok",
                        round(seconds, 3),
                        summary(result),
                        _dumps(response_json(result)),
                    ),
                )
                self._db.executemany(
                    "INSERT OR IGNORE INTO call_refs (call_id, kind, ref) VALUES (?, ?, ?)",
                    [(call_id, k, r) for k, r in refs],
                )
                self._db.commit()
        except Exception as exc:  # noqa: BLE001 - logging is best effort
            log.warning("call log: could not record %s: %s", tool, exc)
            return result
        self._schedule_push()
        if tool in QUIET_TOOLS:
            return result
        return with_links(result, self.links.for_call(call_id, refs))

    # -- copying to the database ----------------------------------------------------------------
    def _schedule_push(self) -> None:
        if not self.remote or self._pushing or self.clock() < self._retry_at:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._pushing = True
        task = loop.create_task(asyncio.to_thread(self.push))
        task.add_done_callback(lambda _: setattr(self, "_pushing", False))

    def _connection(self) -> Any:
        if self._remote is None:
            if self._connect is not None:
                self._remote = self._connect()
            else:
                import psycopg  # optional: `uv sync --extra calllog`

                self._remote = psycopg.connect(self.database_url, connect_timeout=10)
        if not self._schema_ready:
            with self._remote.cursor() as cur:
                cur.execute(REMOTE_SCHEMA)
            self._remote.commit()
            self._schema_ready = True
        return self._remote

    def push(self) -> int:
        """Copy unsynced calls to the database; prune old synced ones. The number copied."""
        if not self.remote:
            return 0
        copied = 0
        try:
            conn = self._connection()
            while True:
                with self._lock:
                    rows = self._db.execute(
                        "SELECT * FROM calls WHERE synced = 0 ORDER BY created LIMIT ?",
                        (PUSH_BATCH,),
                    ).fetchall()
                    ids = [r["id"] for r in rows]
                    refs = (
                        self._db.execute(
                            f"SELECT call_id, kind, ref FROM call_refs WHERE call_id IN "
                            f"({','.join('?' * len(ids))})",
                            ids,
                        ).fetchall()
                        if ids
                        else []
                    )
                if not rows:
                    break
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO mcp_calls (id, created, client, session, project, tool,"
                        " arguments, status, seconds, summary, response) VALUES (%s, %s, %s, %s,"
                        " %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb) ON CONFLICT (id) DO NOTHING",
                        [
                            (
                                r["id"],
                                datetime.fromtimestamp(r["created"], UTC),
                                r["client"],
                                r["session"],
                                r["project"],
                                r["tool"],
                                r["arguments"],
                                r["status"],
                                r["seconds"],
                                r["summary"],
                                r["response"],
                            )
                            for r in rows
                        ],
                    )
                    if refs:
                        cur.executemany(
                            "INSERT INTO mcp_call_refs (call_id, kind, ref) VALUES (%s, %s, %s)"
                            " ON CONFLICT DO NOTHING",
                            [(r["call_id"], r["kind"], r["ref"]) for r in refs],
                        )
                conn.commit()
                with self._lock:
                    self._db.executemany(
                        "UPDATE calls SET synced = 1 WHERE id = ?", [(i,) for i in ids]
                    )
                    self._db.commit()
                copied += len(rows)
            self._prune()
            self.last_error = ""
        except Exception as exc:  # noqa: BLE001 - any driver or network error: retry later
            self.last_error = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("call log: copy to the database failed: %s", self.last_error)
            self._retry_at = self.clock() + RETRY_SECONDS
            if self._remote is not None:
                with contextlib.suppress(Exception):
                    self._remote.close()
            self._remote, self._schema_ready = None, False
        return copied

    def _prune(self) -> None:
        cutoff = self.clock() - self.local_days * 86400
        with self._lock:
            old = "SELECT id FROM calls WHERE synced = 1 AND created < ?"
            self._db.execute(f"DELETE FROM call_refs WHERE call_id IN ({old})", (cutoff,))
            self._db.execute("DELETE FROM calls WHERE synced = 1 AND created < ?", (cutoff,))
            self._db.commit()

    def status(self) -> dict[str, Any]:
        with self._lock:
            pending = self._db.execute("SELECT COUNT(*) FROM calls WHERE synced = 0").fetchone()[0]
        return {
            "database": "configured" if self.remote else "not configured (local only)",
            "unsynced": pending,
            "last_error": self.last_error,
        }

    # -- reading (for the record pages) ---------------------------------------------------------
    def _read(
        self, sql: str, args: tuple[Any, ...] = (), *, local: bool = False
    ) -> list[dict[str, Any]]:
        """Rows from the database when it is configured and answers, else (or with `local`)
        from the local file. `sql` uses {calls}, {refs} and {p} (the placeholder) so one query
        serves both."""
        if self.remote and not local:
            try:
                conn = self._connection()
                q = sql.format(calls="mcp_calls", refs="mcp_call_refs", p="%s")
                with conn.cursor() as cur:
                    cur.execute(q, args)
                    names = [d[0] for d in cur.description]
                    rows = [dict(zip(names, row, strict=True)) for row in cur.fetchall()]
                conn.commit()
                return [_normalise(r) for r in rows]
            except Exception as exc:  # noqa: BLE001 - fall back to the local copy
                self.last_error = f"{type(exc).__name__}: {exc}"[:300]
                self._remote, self._schema_ready = None, False
        q = sql.format(calls="calls", refs="call_refs", p="?")
        with self._lock:
            return [_normalise(dict(r)) for r in self._db.execute(q, args).fetchall()]

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._read(
            "SELECT id, created, client, session, project, tool, status, seconds, summary"
            " FROM {calls} ORDER BY created DESC LIMIT {p}",
            (limit,),
        )

    def get(self, call_id: str) -> dict[str, Any] | None:
        """One call with its refs; from the local copy when the database doesn't have it yet."""
        for local in (False, True):
            rows = self._read("SELECT * FROM {calls} WHERE id = {p}", (call_id,), local=local)
            if rows:
                row = rows[0]
                row["refs"] = [
                    (r["kind"], r["ref"])
                    for r in self._read(
                        "SELECT kind, ref FROM {refs} WHERE call_id = {p} ORDER BY kind, ref",
                        (call_id,),
                        local=local,
                    )
                ]
                return row
            if not self.remote:
                break
        return None

    def referencing(self, pairs: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
        """Calls that mention any of the (kind, ref) pairs, newest first."""
        pairs = sorted(set(pairs))
        if not pairs:
            return []
        where = " OR ".join("(r.kind = {p} AND r.ref = {p})" for _ in pairs)
        args = tuple(x for pair in pairs for x in pair)
        return self._read(
            "SELECT DISTINCT c.id, c.created, c.client, c.session, c.project, c.tool, c.status,"
            " c.seconds, c.summary FROM {calls} c JOIN {refs} r ON r.call_id = c.id"
            f" WHERE {where} ORDER BY c.created DESC LIMIT 200",
            args,
        )

    def related_refs(self, kind: str, ref: str, want: str) -> list[str]:
        """Refs of kind `want` on calls that also mention (kind, ref): e.g. a run's run ids."""
        return [
            r["ref"]
            for r in self._read(
                "SELECT DISTINCT b.ref FROM {refs} a JOIN {refs} b ON b.call_id = a.call_id"
                " WHERE a.kind = {p} AND a.ref = {p} AND b.kind = {p}",
                (kind, ref, want),
            )
        ]


def _normalise(row: dict[str, Any]) -> dict[str, Any]:
    """The same shape from either store: `created` as epoch seconds, JSON fields as text."""
    created = row.get("created")
    if isinstance(created, datetime):
        row["created"] = created.timestamp()
    for key in ("arguments", "response"):
        if key in row and not isinstance(row[key], str | type(None)):
            row[key] = json.dumps(row[key], ensure_ascii=False)
    return row
