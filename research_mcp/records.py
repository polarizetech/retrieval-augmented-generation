"""Read-only pages for the call log (docs/call-log.md): recent calls, one call, and a research
run or saved report with every call that refers to it (its backlinks).

The pages are served by the HTTP gateway under /records, but only to requests that reach it
from this machine and not through Tailscale Funnel: the public MCP host never shows them. To
open them from another device, publish the gateway on the tailnet only with `tailscale serve`
and set RECORDS_URL to that address.
"""

from __future__ import annotations

import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, Response
from starlette.routing import Route

from research_mcp.calllog import CallLog

LOOPBACK = ("127.0.0.1", "::1")
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")
# The documents a run folder may hold, the most useful first.
RUN_DOCS = ("answer.md", "capsule.md", "manifest.json")

STYLE = """
body{font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:60rem;margin:2rem auto;
padding:0 1rem;color:#1d1d1f}
a{color:#0b57d0}table{border-collapse:collapse;width:100%}
td,th{padding:.35rem .5rem;border-bottom:1px solid #e5e5e5;text-align:left;vertical-align:top}
th{font-weight:600;color:#555}pre{background:#f6f6f6;padding:.75rem;overflow-x:auto;
white-space:pre-wrap;word-break:break-word;font-size:13px}.muted{color:#777}
.err{color:#b3261e}h1{font-size:1.4rem}h2{font-size:1.1rem;margin-top:2rem}
"""


def allowed(request: Request, public_host: str) -> bool:
    """From this machine (directly or through tailnet-only `tailscale serve`), never Funnel."""
    if "tailscale-funnel-request" in request.headers:
        return False
    host = request.headers.get("host", "").rsplit(":", 1)[0]
    if public_host and host == public_host:
        return False
    client = request.client.host if request.client else ""
    return client in LOOPBACK


def _when(epoch: float | None) -> str:
    if not epoch:
        return ""
    return datetime.fromtimestamp(float(epoch), UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_e(title)}</title><style>{STYLE}</style></head><body>"
        f"<p class='muted'><a href='/records'>All records</a></p>{body}</body></html>"
    )


def _calls_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class='muted'>No calls.</p>"
    lines = ["<table><tr><th>When</th><th>Tool</th><th>Session</th><th>Summary</th></tr>"]
    for r in rows:
        status = " <span class='err'>(error)</span>" if r.get("status") == "error" else ""
        lines.append(
            f"<tr><td><a href='/records/{_e(r['id'])}'>{_e(_when(r.get('created')))}</a></td>"
            f"<td>{_e(r.get('tool'))}{status}</td>"
            f"<td class='muted'>{_e(r.get('client'))} {_e(r.get('session'))}</td>"
            f"<td>{_e(r.get('summary'))[:240]}</td></tr>"
        )
    return "\n".join(lines) + "</table>"


def _pretty(text: str | None) -> str:
    if not text:
        return ""
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return text


def _ref_link(kind: str, ref: str) -> str:
    if kind in ("run", "run_id"):
        return f"<a href='/records/run/{_e(ref)}'>{_e(ref)}</a>"
    if kind == "request":
        return f"<a href='/records/request/{_e(ref)}'>{_e(ref)}</a>"
    if kind == "doi":
        return f"<a href='https://doi.org/{_e(ref)}'>doi:{_e(ref)}</a>"
    if kind == "url":
        return f"<a href='{_e(ref)}'>{_e(ref)}</a>"
    return _e(ref)


REF_LABELS = {
    "run": "Research docs",
    "run_id": "Research runs",
    "request": "Requests",
    "doi": "Papers",
    "url": "Links",
}


def routes(calls: CallLog, runs_dir: Path, public_host: str = "") -> list[Route]:
    def guard(request: Request) -> Response | None:
        if not allowed(request, public_host):
            return Response("Not found", status_code=404)
        return None

    def index(request: Request) -> Response:
        if (refused := guard(request)) is not None:
            return refused
        status = calls.status()
        note = (
            f"<p class='muted'>Database: {_e(status['database'])}; "
            f"{status['unsynced']} call(s) not copied yet"
            + (
                f"; <span class='err'>last error: {_e(status['last_error'])}</span>"
                if status["last_error"]
                else ""
            )
            + "</p>"
        )
        return _page("Records", "<h1>Recent tool calls</h1>" + note + _calls_table(calls.recent()))

    def one(request: Request) -> Response:
        if (refused := guard(request)) is not None:
            return refused
        row = calls.get(request.path_params["call_id"])
        if row is None:
            return _page("Not found", "<h1>No such record</h1>")
        refs: dict[str, list[str]] = {}
        for kind, ref in row["refs"]:
            refs.setdefault(kind, []).append(ref)
        ref_html = "".join(
            f"<tr><th>{_e(REF_LABELS.get(kind, kind))}</th><td>"
            + ", ".join(_ref_link(kind, r) for r in items)
            + "</td></tr>"
            for kind, items in refs.items()
        )
        related = [
            r
            for r in calls.referencing(
                (k, v) for k, v in row["refs"] if k in ("run", "run_id", "request")
            )
            if r["id"] != row["id"]
        ]
        body = (
            f"<h1>{_e(row['tool'])}</h1><table>"
            f"<tr><th>When</th><td>{_e(_when(row.get('created')))}</td></tr>"
            f"<tr><th>Who</th><td>{_e(row.get('client'))} · session {_e(row.get('session'))}"
            f" · project {_e(row.get('project'))}</td></tr>"
            f"<tr><th>Status</th><td>{_e(row.get('status'))} in {_e(row.get('seconds'))} s"
            "</td></tr>"
            f"<tr><th>Record id</th><td>{_e(row['id'])}</td></tr>{ref_html}</table>"
            f"<h2>Summary</h2><p>{_e(row.get('summary'))}</p>"
            f"<h2>Related calls</h2>{_calls_table(related)}"
            f"<h2>Arguments</h2><pre>{_e(_pretty(row.get('arguments')))}</pre>"
            f"<h2>Full response</h2><pre>{_e(_pretty(row.get('response')))}</pre>"
        )
        return _page(f"{row['tool']} {row['id']}", body)

    def run(request: Request) -> Response:
        if (refused := guard(request)) is not None:
            return refused
        key = request.path_params["key"]
        if not RUN_NAME.match(key):
            return _page("Not found", "<h1>No such run</h1>")
        # `key` is a run folder's name or a run id; link the two through the calls that
        # mention both.
        folder = runs_dir / key
        names = [key] if folder.is_dir() else calls.related_refs("run_id", key, "run")
        run_ids = {key} if not folder.is_dir() else set(calls.related_refs("run", key, "run_id"))
        for name in names:
            run_ids.update(calls.related_refs("run", name, "run_id"))
        docs = ""
        for name in names:
            path = runs_dir / name
            if not RUN_NAME.match(name) or not path.is_dir():
                continue
            for doc in RUN_DOCS:
                if (path / doc).is_file():
                    text = (path / doc).read_text(errors="replace")
                    docs += f"<h2>{_e(doc)} <span class='muted'>({_e(name)})</span></h2>"
                    docs += f"<pre>{_e(_pretty(text) if doc.endswith('.json') else text)}</pre>"
                    break
        if not docs:
            docs = (
                "<p class='muted'>No document for this run yet: it may still be running, or "
                "its folder is not on this machine.</p>"
            )
        backlinks = calls.referencing(
            [("run", n) for n in names] + [("run_id", r) for r in run_ids]
        )
        ids = ", ".join(sorted(run_ids)) or "none recorded"
        body = (
            f"<h1>Research: {_e(names[0] if names else key)}</h1>"
            f"<p class='muted'>Run id(s): {_e(ids)}</p>"
            f"<h2>Referenced in {len(backlinks)} call(s)</h2>{_calls_table(backlinks)}{docs}"
        )
        return _page(f"Research {key}", body)

    def request_page(request: Request) -> Response:
        if (refused := guard(request)) is not None:
            return refused
        rid = request.path_params["rid"]
        rows = calls.referencing([("request", rid)])
        body = f"<h1>Request {_e(rid)}</h1>{_calls_table(rows)}"
        return _page(f"Request {rid}", body)

    return [
        Route("/records", index),
        Route("/records/run/{key}", run),
        Route("/records/request/{rid}", request_page),
        Route("/records/{call_id}", one),
    ]
