"""The call log: what it records, the links it adds, copying to a database, and the pages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, TextContent
from starlette.applications import Starlette
from starlette.testclient import TestClient

from research_mcp import records
from research_mcp.calllog import CallLog, Links, references

BASE = "https://mac.tailnet.ts.net:8892"


def make_log(tmp_path: Path, **kwargs: Any) -> CallLog:
    return CallLog(tmp_path / "calls.sqlite", Links(BASE), **kwargs)


def text_result(data: Any, error: bool = False) -> CallToolResult:
    body = data if isinstance(data, str) else json.dumps(data)
    return CallToolResult(isError=error, content=[TextContent(type="text", text=body)])


def test_references_find_runs_requests_dois_and_urls() -> None:
    reply = {
        "request_id": "abc123",
        "results": {"research": {"run_id": "r1"}},
        "passages": [{"doi": "10.1121/10.0015244"}, {"doi": "not a doi"}],
        "note": "see https://github.com/polarizetech/research/pull/7.",
    }
    found = references("ask", {"question": "q"}, reply)
    assert found == {
        ("request", "abc123"),
        ("run_id", "r1"),
        ("doi", "10.1121/10.0015244"),
        ("url", "https://github.com/polarizetech/research/pull/7"),
    }
    done = text_result({"run_id": "r1", "run_dir": "/x/runs/20261105-cochlea"})
    assert ("run", "20261105-cochlea") in references("pipeline__research_continue", {}, done)
    assert ("request", "abc123") in references("hub__result", {"id": "abc123"}, {})


def test_a_dict_reply_gets_links_on_it_and_on_its_relay(tmp_path: Path) -> None:
    calls = make_log(tmp_path)
    out = calls.record(
        "c1", "ask", {"question": "q"}, {"relay": "**Results**\n- x", "request_id": "abc"},
        seconds=0.5, who={"client": "claude", "session": "s1", "project": "p"},
    )  # fmt: skip
    assert out["links"] == {
        "record": f"{BASE}/records/c1",
        "requests": [f"{BASE}/records/request/abc"],
    }
    assert out["relay"].endswith(
        f"**Links**\n- [Record of this call]({BASE}/records/c1)\n"
        f"- [Request history]({BASE}/records/request/abc)"
    )
    row = calls.get("c1")
    assert row is not None
    assert (row["tool"], row["client"], row["session"], row["status"]) == (
        "ask", "claude", "s1", "ok",
    )  # fmt: skip
    assert row["refs"] == [("request", "abc")]
    assert json.loads(row["response"])["request_id"] == "abc"


def test_a_tool_result_gets_links_in_its_json_or_as_a_new_block(tmp_path: Path) -> None:
    calls = make_log(tmp_path)
    out = calls.record(
        "c2", "pipeline__research_continue", {"run_id": "r1"},
        text_result({"status": "done", "relay": "**Results**", "run_dir": "/runs/run-a"}),
        seconds=1.0,
    )  # fmt: skip
    data = json.loads(out.content[0].text)
    assert data["links"]["research"] == [f"{BASE}/records/run/run-a"]
    assert "[Research doc and its references]" in data["relay"]

    plain = calls.record("c3", "papers__text", {}, text_result("plain words"), seconds=0.1)
    assert plain.content[0].text == "plain words"
    assert plain.content[1].text.startswith("**Links**\n- [Record of this call]")

    failed = calls.record("c4", "papers__fetch", {}, text_result("boom", error=True), seconds=0)
    assert failed.isError
    assert calls.get("c4")["status"] == "error"  # type: ignore[index]


def test_status_tools_are_recorded_without_links(tmp_path: Path) -> None:
    calls = make_log(tmp_path)
    assert calls.record("c5", "gateway_status", {}, {"connected": {}}, seconds=0) == {
        "connected": {}
    }
    assert calls.get("c5") is not None


class FakeCursor:
    def __init__(self, conn: FakeConnection):
        self.conn = conn
        self.description: list[tuple[str]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, args: Any = None) -> None:
        if self.conn.fail:
            raise OSError("network down")
        self.conn.statements.append(sql)

    def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        if self.conn.fail:
            raise OSError("network down")
        table = "mcp_call_refs" if "mcp_call_refs" in sql else "mcp_calls"
        self.conn.rows.setdefault(table, []).extend(rows)


class FakeConnection:
    def __init__(self) -> None:
        self.fail = False
        self.statements: list[str] = []
        self.rows: dict[str, list[tuple[Any, ...]]] = {}
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        return None


def test_push_copies_calls_and_refs_then_prunes_old_synced_ones(tmp_path: Path) -> None:
    conn = FakeConnection()
    now = [1_000_000.0]
    calls = make_log(tmp_path, connect=lambda: conn, local_days=7, clock=lambda: now[0])
    calls.record("c1", "ask", {}, {"request_id": "abc", "relay": "r"}, seconds=0)
    assert calls.push() == 1
    assert "CREATE TABLE IF NOT EXISTS mcp_calls" in conn.statements[0]
    assert [r[0] for r in conn.rows["mcp_calls"]] == ["c1"]
    assert conn.rows["mcp_call_refs"] == [("c1", "request", "abc")]
    assert calls.status()["unsynced"] == 0
    assert calls.push() == 0  # nothing new

    now[0] += 8 * 86400  # a week later the synced row leaves this machine
    calls.record("c2", "ask", {}, {"relay": "r"}, seconds=0)
    assert calls.push() == 1
    assert calls.get("c1") is None or calls.remote  # read goes to the database when configured
    assert calls._read("SELECT id FROM {calls}", local=True) == [{"id": "c2"}]


def test_a_failed_push_keeps_everything_and_retries_later(tmp_path: Path) -> None:
    conn = FakeConnection()
    conn.fail = True
    calls = make_log(tmp_path, connect=lambda: conn)
    calls.record("c1", "ask", {}, {"relay": "r"}, seconds=0)
    assert calls.push() == 0
    assert "network down" in calls.status()["last_error"]
    assert calls.status()["unsynced"] == 1
    conn.fail = False
    assert calls.push() == 1
    assert calls.status()["last_error"] == ""


def test_without_a_database_nothing_is_pruned(tmp_path: Path) -> None:
    now = [1_000_000.0]
    calls = make_log(tmp_path, clock=lambda: now[0])
    calls.record("c1", "ask", {}, {"relay": "r"}, seconds=0)
    now[0] += 365 * 86400
    calls.record("c2", "ask", {}, {"relay": "r"}, seconds=0)
    assert calls.push() == 0
    assert {r["id"] for r in calls.recent()} == {"c1", "c2"}


def page_client(calls: CallLog, runs: Path, public_host: str = "") -> TestClient:
    app = Starlette(routes=records.routes(calls, runs, public_host))
    return TestClient(app, client=("127.0.0.1", 50000))


def test_a_research_doc_page_shows_the_doc_and_every_call_that_mentions_it(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    (runs / "run-a").mkdir(parents=True)
    (runs / "run-a" / "answer.md").write_text("# Cochlear distortion\n\nBottom line: <b>13%</b>")
    calls = make_log(tmp_path)
    calls.record("c1", "ask", {"question": "q"}, {"request_id": "abc",
                 "results": {"research": {"run_id": "r1"}}}, seconds=0)  # fmt: skip
    calls.record("c2", "pipeline__research_continue", {"run_id": "r1"},
                 text_result({"status": "working"}), seconds=0)  # fmt: skip
    done = text_result({"status": "done", "run_dir": str(runs / "run-a")})
    calls.record("c3", "pipeline__research_continue", {"run_id": "r1"}, done, seconds=0)
    calls.record("c4", "papers__search", {"query": "other"}, {"results": []}, seconds=0)
    web = page_client(calls, runs)

    by_folder = web.get("/records/run/run-a")
    assert by_folder.status_code == 200
    assert "Referenced in 3 call(s)" in by_folder.text  # c1 (run id), c2, c3; not c4
    assert "/records/c1" in by_folder.text
    assert "/records/c4" not in by_folder.text
    assert "&lt;b&gt;13%&lt;/b&gt;" in by_folder.text  # the doc, escaped
    by_run_id = web.get("/records/run/r1")
    assert "Referenced in 3 call(s)" in by_run_id.text
    assert "Cochlear distortion" in by_run_id.text

    record = web.get("/records/c3")
    assert "<a href='/records/run/run-a'>run-a</a>" in record.text
    assert "/records/c1" in record.text  # related through the run id
    assert web.get("/records/request/abc").text.count("/records/c1") == 1
    assert "Recent tool calls" in web.get("/records").text
    assert "No such record" in web.get("/records/nope").text
    assert web.get("/records/run/..%2F..%2Fetc").status_code == 404
    assert "No such run" in web.get("/records/run/.hidden").text


def test_record_pages_are_refused_through_funnel_and_off_this_machine(tmp_path: Path) -> None:
    calls = make_log(tmp_path)
    web = page_client(calls, tmp_path, public_host="polarize-mcp.example.ts.net")
    assert web.get("/records").status_code == 200
    assert web.get("/records", headers={"Tailscale-Funnel-Request": "?1"}).status_code == 404
    assert web.get("/records", headers={"Host": "polarize-mcp.example.ts.net"}).status_code == 404
    remote = TestClient(
        Starlette(routes=records.routes(calls, tmp_path)), client=("100.64.0.9", 50000)
    )
    assert remote.get("/records").status_code == 404
