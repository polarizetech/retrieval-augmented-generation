"""Hub phase 2 (docs/hub-plan.md, sections 2 and 3): queues and limits, reply deadlines,
memory pressure, duplicate and near-duplicate research, and stale processes. Each is forced on
purpose and the reply checked."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from research_mcp.hub.ledger import Ledger
from research_mcp.hub.processes import stale
from research_mcp.hub.queue import Lane
from research_pipeline.config import Settings
from research_pipeline.ladder import Ladder
from research_pipeline.usage import Meters
from tests.test_hub import Rig, Rung, World, hub_config


async def settle(rig: Rig) -> None:
    """Let work that outlived its reply finish."""
    await asyncio.gather(*list(rig.asker._tasks), return_exceptions=True)


# -- lanes ------------------------------------------------------------------------------------
def test_a_lane_lets_its_limit_in_and_queues_the_rest_in_order() -> None:
    async def go() -> None:
        lane = Lane("retrieval", 2, typical_seconds=10, memory=lambda: 50)
        gate = asyncio.Event()
        order: list[str] = []

        async def hold(rid: str) -> None:
            async with lane.slot(rid):
                order.append(rid)
                await gate.wait()

        tasks = [asyncio.create_task(hold(r)) for r in ("a", "b", "c", "d")]
        await asyncio.sleep(0.05)
        assert sorted(lane.running) == ["a", "b"]
        assert lane.position("c") == 1
        assert lane.position("d") == 2
        assert lane.estimate("d") == 10.0  # two ahead of it, two at a time
        gate.set()
        await asyncio.gather(*tasks)
        assert order == ["a", "b", "c", "d"]
        assert lane.as_dict()["waiting"] == 0

    asyncio.run(go())


def test_a_heavy_lane_waits_while_memory_is_short() -> None:
    async def go() -> None:
        free = [5]
        lane = Lane("retrieval", 2, heavy=True, memory=lambda: free[0], min_free=10)
        entered = asyncio.Event()

        async def enter() -> None:
            async with lane.slot("a"):
                entered.set()

        task = asyncio.create_task(enter())
        await asyncio.sleep(0.05)
        assert not entered.is_set()
        assert lane.held_by_memory
        free[0] = 40
        await asyncio.wait_for(task, 5)
        assert entered.is_set()

    asyncio.run(go())


# -- deadlines --------------------------------------------------------------------------------
def test_slow_work_answers_working_at_the_deadline_and_finishes_into_the_ledger(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    rig.library.slow = True
    rig.hub.deadline = 0.1

    async def go() -> None:
        started = time.monotonic()
        out = await rig.asker.ask("What is pink noise?", session="s1")
        assert time.monotonic() - started < 0.5
        assert out["status"] == "working"
        assert out["relay"].startswith("**Still working after")
        assert f"`hub__result` with id `{out['request_id']}`" in out["relay"]
        assert (rig.ledger.get(out["request_id"]) or {})["status"] == "working"
        await settle(rig)
        row = rig.ledger.get(out["request_id"]) or {}
        assert row["status"] == "done"
        assert "**Passages**" in row["result"]["relay"]

    asyncio.run(go())


def test_a_busy_retrieval_lane_answers_queued_with_position_and_estimate(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.library.slow = True
    rig.hub.deadline = 0.3

    async def go() -> None:
        outs = await asyncio.gather(
            *(rig.asker.ask(f"What is noise number {i}?", session="s1") for i in range(3))
        )
        queued = [o for o in outs if o["status"] == "queued"]
        assert len(queued) == 1
        assert queued[0]["error"]["component"] == "queue"
        assert "Retrieval is busy (2 at a time): position 1 in line" in queued[0]["relay"]
        assert queued[0]["error"]["retry_after"] >= 5
        await settle(rig)
        assert all((rig.ledger.get(o["request_id"]) or {})["status"] == "done" for o in outs)

    asyncio.run(go())


def test_low_memory_holds_retrieval_and_says_so(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.free_memory = 3
    rig.hub.deadline = 0.2

    async def go() -> None:
        out = await rig.asker.ask("What is pink noise?", session="s1")
        assert out["status"] == "queued"
        assert "Memory is low, so heavy work waits." in out["relay"]
        rig.free_memory = 60
        await settle(rig)
        assert (rig.ledger.get(out["request_id"]) or {})["status"] == "done"

    asyncio.run(go())


def test_a_full_pipeline_queues_research_and_starts_it_when_a_slot_frees(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    full = {"error": "4 runs at once is the limit; wait for one to finish"}
    replies = [full, full, {"run_id": "r9", "status": "working"}]
    rig.run_reply = replies  # type: ignore[assignment]
    rig.hub.deadline = 0.02

    async def go() -> None:
        out = await rig.asker.ask("research: is pink noise safe?", session="s1")
        assert out["status"] == "queued"
        assert "Position 1 in line" in out["relay"]
        await settle(rig)
        row = rig.ledger.get(out["request_id"]) or {}
        assert row["status"] == "working"
        assert row["run_id"] == "r9"
        assert list(rig.hub.lanes.research) == []

    asyncio.run(go())


# -- duplicates and near overlaps -------------------------------------------------------------
def test_the_same_research_question_attaches_to_the_run_on_the_go(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    first = rig.ask("research: does pink noise improve sleep?", session="a")
    second = rig.ask("Research: does PINK noise improve  sleep", session="b")
    starts = [c for c in rig.calls if c[0] == "pipeline__research_start"]
    assert len(starts) == 1
    assert second["status"] == "working"
    assert second["results"]["research"]["attached_to"] == first["request_id"]
    assert f"**Attached** to request `{first['request_id']}` (run `r1`)" in second["relay"]
    assert "asked by session a" in second["relay"]
    # When the run finishes, both requests have its answer.
    rig.ledger.follow_run("r1", {"status": "done", "answer": "Maybe."})
    assert (rig.ledger.get(second["request_id"]) or {})["status"] == "done"
    assert (rig.ledger.get(first["request_id"]) or {})["result"]["answer"] == "Maybe."


def test_a_paraphrase_attaches_and_a_near_question_runs_with_a_warning(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.vectors = {
        "does pink noise improve sleep?": [1.0, 0.0, 0.0],
        "can pink noise help people sleep better?": [0.97, 0.24, 0.0],
        "does brown noise improve sleep?": [0.85, 0.53, 0.0],
    }
    first = rig.ask("research: does pink noise improve sleep?", session="a")
    para = rig.ask("research: can pink noise help people sleep better?", session="b")
    assert para["results"]["research"]["attached_to"] == first["request_id"]
    assert "bge-m3 similarity" in para["relay"]

    near = rig.ask("research: does brown noise improve sleep?", session="c")
    assert "attached_to" not in near["results"]["research"]
    assert any("A similar research request is on the go" in p for p in near["progress"])
    assert [c[0] for c in rig.calls].count("pipeline__research_start") == 2


def test_without_embeddings_only_the_same_words_attach(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    def down(url: str, body: dict[str, Any]) -> Any:
        raise OSError("ollama down")

    rig.hub.overlap.post = down
    rig.ask("research: does pink noise improve sleep?", session="a")
    other = rig.ask("research: can pink noise help people sleep better?", session="b")
    assert "attached_to" not in other["results"]["research"]
    same = rig.ask("research: does pink noise improve sleep?", session="c")
    assert same["results"]["research"]["how"] == "the same words"


# -- the ledger -------------------------------------------------------------------------------
def test_an_older_ledger_gains_the_new_column_on_open(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE requests (id TEXT PRIMARY KEY, created REAL NOT NULL, updated REAL NOT NULL, "
        "client TEXT, session TEXT, session_inferred INTEGER, project TEXT, repo TEXT, route TEXT, "
        "specialist TEXT, fingerprint TEXT, question TEXT, touches TEXT, status TEXT NOT NULL, "
        "seconds REAL, providers TEXT, tokens INTEGER, cost REAL, problems TEXT, run_id TEXT, "
        "result TEXT)"
    )
    db.execute("INSERT INTO requests (id, created, updated, status) VALUES ('x', 1, 1, 'done')")
    db.commit()
    db.close()
    ledger = Ledger(path)
    ledger.finish("x", "done", attached_to="y")
    assert (ledger.get("x") or {})["attached_to"] == "y"


# -- stale processes --------------------------------------------------------------------------
PS = """\
  100     1 Tue Oct 27 05:00:00 2026 /venv/bin/python /venv/bin/research-rag-gateway --config c.json
  101   100 Tue Oct 27 05:00:01 2026 /tools/paper-fetch/bin/python /home/.local/bin/paper-fetch-mcp
  200     1 Wed Oct 28 13:00:00 2026 /venv/bin/python /venv/bin/research-rag-gateway --config c.json --connect http://127.0.0.1:8891/
  300     1 Tue Oct 27 06:00:00 2026 /tools/paper-fetch/bin/python /home/.local/bin/paper-fetch-mcp
  400   399 Wed Oct 28 13:00:00 2026 /Applications/Claude.app/Contents/Helpers/disclaimer --pgroup -- research-rag-gateway
  500     1 Wed Oct 28 12:00:00 2026 /venv/bin/python /venv/bin/research-rag-gateway --http
  600   599 Wed Oct 28 13:00:00 2026 /usr/bin/vim notes.md
"""  # noqa: E501


def test_stale_processes_are_found_and_explained() -> None:
    changed = time.mktime(time.strptime("2026-10-28 12:30", "%Y-%m-%d %H:%M"))
    found = {
        r["pid"]: r
        for r in stale(lambda: PS, lambda bridge=False: changed - 3600 * bridge, own_pid=500)
    }
    assert set(found) == {100, 300}
    assert found[100]["kind"] == "stdio gateway"
    assert any("--connect would share one" in w for w in found[100]["why"])
    assert any("old version" in w for w in found[100]["why"])
    assert found[300]["why"] == ["its gateway has gone: an orphan"]


# -- per-provider limits ----------------------------------------------------------------------
class SlowRung(Rung):
    def __init__(self, settings: Settings, name: str, gauge: dict[str, int]):
        super().__init__(settings, name)
        self.gauge = gauge

    def chat_json(self, task: str, system: str, user: str, schema: dict[str, Any], **_: Any):
        self.gauge["now"] += 1
        self.gauge["most"] = max(self.gauge["most"], self.gauge["now"])
        time.sleep(0.05)
        self.gauge["now"] -= 1
        return {"by": self.name}


def test_the_local_model_takes_one_call_at_a_time_across_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("research_pipeline.ladder._SLOTS", {})
    config = hub_config(tmp_path)
    config["ladders"] = {"writing": ["local"]}
    config["limits"] = {"providers": {"local": 1}}
    settings = replace(
        Settings(), endpoint_url="https://openrouter.ai/api/v1", endpoint_model="qwen/qwen3.8-27b"
    )
    gauge = {"now": 0, "most": 0}
    meters = Meters(config, settings, fetch=World().fetch)
    runs = [
        Ladder(settings, config, meters=meters, rungs={"local": SlowRung(settings, "local", gauge)})
        for _ in range(3)
    ]
    threads = [
        threading.Thread(target=r.chat_json, args=("synthesise", "s", "u", {})) for r in runs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert gauge["most"] == 1
