"""The pipeline behind MCP, with the calling model as its model.

A scripted "client" answers every task it is handed through research_start / research_continue,
exactly as a chat model would, and the run must reach the same verdicts as the local-model run in
test_pipeline.py, in a small number of turns.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from paper_fetch import Library

from research_pipeline import mcp_server, verify
from research_pipeline import pipeline as pipeline_module
from research_pipeline.llm import Ollama
from tests.conftest import PAPER, LibraryAdapter, hold, hold_w1, papers_library
from tests.test_pipeline import QUESTION, ScriptedModel


class NoChat(Ollama):
    """In client mode nothing may reach Ollama: the client writes, the library embeds."""

    def resolve(self, model: str) -> tuple[str, str | None]:
        return model, None

    def chat_json(self, *_: Any, **__: Any) -> dict[str, Any]:
        raise AssertionError("client mode must not send chat calls to Ollama")


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Library:
    for key, value in {
        "PIPELINE_DATA_DIR": str(tmp_path / "data"),
        "PIPELINE_RUNS_DIR": str(tmp_path / "runs"),
        "PIPELINE_RERANKER": "none",
        "PIPELINE_VERIFIER_MODEL": "",
        "PIPELINE_MCP_LLM": "client",
        "PIPELINE_CLIENT_TURN_WAIT": "5",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(mcp_server, "Ollama", NoChat)
    monkeypatch.setattr(mcp_server, "RUNS", {})
    library = papers_library(tmp_path)
    hold_w1(library)
    library.index_works()
    monkeypatch.setattr(pipeline_module, "PaperLibrary", lambda *_: LibraryAdapter(library))
    return library


class CorrectingModel(ScriptedModel):
    """Like a chat model: after its answer is refused, it fixes it on the next turn."""

    def __init__(self) -> None:
        super().__init__()
        self.refused = False

    def synthesise(self, user: str) -> dict[str, Any]:
        got = super().synthesise(user)
        if self.refused:  # drop the citation to an evidence id that was never retrieved
            got["claims"] = [c for c in got["claims"] if c["evidence_ids"] != ["S1-E99"]]
        return got


def drive(model: ScriptedModel) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Play the client: answer every task until the run is done. Returns (final, all turns)."""

    async def conversation() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        turns = [json.loads(await mcp_server.research_start(QUESTION, offline=True))]
        while turns[-1]["status"] in ("tasks", "working"):
            reply = turns[-1]
            if reply.get("errors") and isinstance(model, CorrectingModel):
                model.refused = True
            answers = [
                {"task_id": t["task_id"], "result": getattr(model, t["task"])(t["input"])}
                for t in reply.get("tasks", [])
            ]
            turns.append(json.loads(await mcp_server.research_continue(reply["run_id"], answers)))
            assert len(turns) < 40, "the run should not need this many turns"
        return turns[-1], turns

    return asyncio.run(conversation())


def test_the_client_runs_every_model_step(env: Library) -> None:
    final, turns = drive(CorrectingModel())
    assert final["status"] == "done", final
    # The draft citing an unretrieved evidence id was refused by the schema, then corrected.
    assert any("is not one of" in e for t in turns for e in t.get("errors", []))
    assert final["stats"]["model_backend"] == "mcp-client"
    log = json.loads((Path(final["run_dir"]) / "run.json").read_text())
    verdicts = {c["text"]: c["verdict"] for c in log["claims"]}
    claim = "Aerobic training lowered resting systolic blood pressure by {} mmHg."
    assert verdicts == {
        claim.format(5): "SUPPORTED",
        claim.format(9): verify.VERDICT_NUMBERS,
    }
    assert log["models"]["text"]["name"].startswith("mcp-client:")
    assert {c["model"] for c in log["llm_calls"]} == {log["models"]["text"]["name"]}
    assert "shares the writer's blind spots" in final["answer"]


def test_model_calls_arrive_in_batches(env: Library) -> None:
    _, turns = drive(CorrectingModel())
    task_turns = [t for t in turns if t["status"] == "tasks"]
    handed_out = sum(len(t["tasks"]) for t in task_turns)
    # plan, extraction+classification, synthesis (+1 refused and corrected), falsification pass,
    # critique, verification, summary, summary check: a handful of turns for many calls.
    assert len(task_turns) <= 12
    assert handed_out > len(task_turns)
    assert all(set(t["prompts"]) == {task["task"] for task in t["tasks"]} for t in task_turns)
    # Each distinct schema is sent once per turn and every task names one that was sent.
    for turn in task_turns:
        assert {task["output_schema"] for task in turn["tasks"]} == set(turn["schemas"])
        assert len(turn["schemas"]) <= len(turn["tasks"])
    extraction = next(t for t in task_turns if any(x["task"] == "extract" for x in t["tasks"]))
    assert len(extraction["schemas"]) < len(extraction["tasks"])


def test_a_wrong_answer_is_refused_and_asked_again(env: Library) -> None:
    async def conversation() -> dict[str, Any]:
        first = json.loads(await mcp_server.research_start(QUESTION, offline=True))
        [task] = first["tasks"]
        assert task["task"] == "plan"
        bad = json.loads(
            await mcp_server.research_continue(
                first["run_id"], [{"task_id": task["task_id"], "result": {"mode": "?"}}]
            )
        )
        assert bad["status"] == "tasks"
        assert "does not match output_schema" in bad["errors"][0]
        assert [t["task_id"] for t in bad["tasks"]] == [task["task_id"]]
        return bad

    asyncio.run(conversation())


def test_one_run_at_a_time(env: Library) -> None:
    async def conversation() -> None:
        first = json.loads(await mcp_server.research_start(QUESTION, offline=True))
        second = json.loads(await mcp_server.research_start(QUESTION, offline=True))
        assert second["active_run_id"] == first["run_id"]
        status = json.loads(await mcp_server.research_status(first["run_id"]))
        assert status["done"] is False
        run = mcp_server.RUNS[first["run_id"]]
        assert run.client is not None
        run.client.close("test over")
        assert run.task is not None
        await run.task
        assert json.loads(await mcp_server.research_continue(first["run_id"]))["status"] == "failed"

    asyncio.run(conversation())


def test_unknown_runs_are_reported(env: Library) -> None:
    reply = json.loads(asyncio.run(mcp_server.research_continue("nope", [])))
    assert "unknown run_id" in reply["error"]


def test_concurrent_model_calls_never_touch_the_note_store(
    env: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model calls run concurrently; the SQLite note store must not. (Found by a live run.)"""
    import threading

    from research_pipeline.notes import NoteStore

    hold(
        env,
        "W2",
        PAPER.replace("48 adults", "52 adults"),
        doi="10.1000/w2",
        title="Another trial",
        year=2021,
    )
    env.index_works()
    threads: set[str] = set()
    for name in ("get", "set"):
        original = getattr(NoteStore, name)

        def spy(self: Any, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            threads.add(threading.current_thread().name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(NoteStore, name, spy)
    final, _ = drive(CorrectingModel())
    assert final["status"] == "done", final
    assert threads  # the notes were used
    # Stages run one after another (in the event loop or an asyncio.to_thread worker); only the
    # model-call pool runs concurrently, and it must never reach the note store.
    assert not any(name.startswith("ThreadPoolExecutor") for name in threads), threads
