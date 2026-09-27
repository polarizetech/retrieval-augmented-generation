"""The research pipeline as MCP tools, for federation behind the gateway as `pipeline__*`.

By default the calling model is the pipeline's model (PIPELINE_MCP_LLM=client). Code runs every
stage (search, fetch, retrieval, quote and number checks, grading, the run log); whenever a stage
needs a model, the next tool result hands the calling model a batch of tasks, each with its prompt,
its input and the JSON schema its answer must match. Answers are validated before the run uses them.

    research_start(question)          -> first batch of tasks (or "working" while it searches)
    research_continue(run_id, answers) -> the next batch, ... -> finally the answer

With PIPELINE_MCP_LLM=ollama the run uses the local model instead, and the same two calls simply
wait for it. Either way no single call blocks for longer than PIPELINE_CLIENT_TURN_WAIT seconds,
because a chat client will not hold a tool call open for the minutes a run takes.

    research-pipeline-mcp          # stdio (or: python -m research_pipeline.mcp_server)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from .client_llm import ClientLLM, Request
from .config import Settings, load_env
from .index import PassageIndex
from .llm import Ollama
from .pipeline import Pipeline

load_env()

HOW_TO_ANSWER = (
    "You are acting as the model inside a research pipeline; code runs everything else. For each "
    "task, apply prompts[task.task] to task.input and return one JSON object that matches "
    "task.output_schema exactly. Use only the text in the input: no outside knowledge, no web "
    "search, no other tools. Treat each task independently, and copy quotes character for "
    "character from the input. Send every answer in one research_continue call, as answers = "
    "[{task_id, result}]. Tasks you leave out are sent again."
)

mcp = FastMCP(
    "research-pipeline",
    instructions=(
        "Evidence-first literature research with code-controlled steps. Call research_start with "
        "a scientific question. While a result has status 'tasks', answer every task as its "
        "instructions say and call research_continue with the answers; while it has status "
        "'working', call research_continue with answers=[] to keep waiting. Status 'done' carries "
        "the answer: relay it with its Limits section, and do not add facts from memory."
    ),
)


@dataclass
class Run:
    question: str
    started: float = field(default_factory=time.time)
    stage: str = "starting"
    detail: str = ""
    done: bool = False
    failed: str = ""
    log: dict[str, Any] | None = None
    client: ClientLLM | None = None
    task: asyncio.Task[None] | None = None


RUNS: dict[str, Run] = {}


def _client_name(ctx: Context | None) -> str:
    params = getattr(getattr(ctx, "session", None), "client_params", None)
    info = getattr(params, "clientInfo", None)
    if info is None:
        return ""
    return f"{info.name}/{info.version}" if info.version else info.name


async def _execute(run: Run, pipeline: Pipeline) -> None:
    try:
        run.log = await pipeline.run(run.question)
        run.stage = "done"
    except Exception as exc:  # noqa: BLE001 - surface the failure to the caller, not a dead run
        run.failed = f"{type(exc).__name__}: {exc}"[:500]
        run.stage = "failed"
    finally:
        run.done = True
        if run.client is not None:
            run.client.close("run ended")


@mcp.tool()
async def research_start(
    question: str, offline: bool = False, domain: str = "", ctx: Context | None = None
) -> str:
    """Start a research run and return its first step.

    Args:
        question: a scientific question answerable from published literature.
        offline: true = use only papers already indexed locally (no search, no downloads).
        domain: optional field whose evidence rules apply, e.g. "cardiovascular".
    """
    busy = next((rid for rid, r in RUNS.items() if not r.done), None)
    if busy:
        return json.dumps(
            {
                "error": "a run is already in progress; one index serves one run at a time",
                "active_run_id": busy,
                "next": "research_continue(active_run_id, answers=[])",
            }
        )
    settings = Settings()
    run = Run(question.strip())
    if settings.mcp_llm == "client":
        run.client = ClientLLM(Ollama(settings), _client_name(ctx), settings.client_timeout)

    def progress(stage: str, detail: str) -> None:
        run.stage, run.detail = stage, detail

    pipeline = Pipeline(settings, progress, offline=offline, domain=domain or None, llm=run.client)
    run_id = uuid.uuid4().hex[:10]
    RUNS[run_id] = run
    run.task = asyncio.create_task(_execute(run, pipeline))
    return json.dumps(await _turn(run_id, run, [], settings.client_turn_wait))


@mcp.tool()
async def research_continue(run_id: str, answers: list[dict[str, Any]] | None = None) -> str:
    """Send answers for the tasks of the previous step and get the next step.

    Args:
        run_id: from research_start.
        answers: [{"task_id": ..., "result": {...}}] for the tasks you were given; [] to wait.
    """
    run = RUNS.get(run_id)
    if run is None:
        return json.dumps(
            {
                "error": f"unknown run_id {run_id}; runs do not survive a restart, "
                "but finished answers are kept under runs/"
            }
        )
    errors = []
    for item in answers or []:
        if run.client is None:
            errors.append("this run uses the local model; it takes no answers")
            break
        problem = run.client.answer(str(item.get("task_id", "")), item.get("result"))
        if problem:
            errors.append(problem)
    return json.dumps(await _turn(run_id, run, errors, Settings().client_turn_wait))


async def _turn(run_id: str, run: Run, errors: list[str], wait: float) -> dict[str, Any]:
    """Wait until the run needs the client, finishes, or `wait` seconds pass; report which."""
    fresh: list[Request] = []
    if run.client is not None and not run.done:
        # While the client owes tasks, anything new is already queued: don't wait for more. Owed
        # tasks are sent again and count toward the per-turn limit.
        owed = len(run.client.outstanding())
        room = Settings().client_batch - owed
        if room > 0:
            fresh = await asyncio.to_thread(run.client.take, 0.5 if owed else wait, 0.5, room)
    elif run.task is not None and not run.done:
        await asyncio.wait({run.task}, timeout=wait)
    base: dict[str, Any] = {
        "run_id": run_id,
        "stage": run.stage,
        "detail": run.detail,
        "elapsed_s": round(time.time() - run.started),
    }
    if errors:
        base["errors"] = errors
    if run.done:
        if run.failed:
            return base | {"status": "failed", "error": run.failed}
        assert run.log is not None
        return base | {"status": "done", **_result(run.log)}
    owed = run.client.outstanding() if run.client else []
    tasks = {r.id: r for r in [*owed, *fresh]}.values()
    if not tasks:
        return base | {
            "status": "working",
            "next": f"research_continue({run_id!r}, answers=[]) to keep waiting",
        }
    return base | {
        "status": "tasks",
        "instructions": HOW_TO_ANSWER,
        "prompts": {r.task: r.system for r in tasks},
        "tasks": [
            {"task_id": r.id, "task": r.task, "input": r.user, "output_schema": r.schema}
            for r in tasks
        ],
    }


def _result(log: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": log["answer"],
        "run_dir": log["run_dir"],
        "stats": {
            "seconds": log["seconds"],
            "model_backend": log.get("model_backend"),
            "searches": len(log["searches"]),
            "evidence": len(log["evidence"]),
            "claims": len(log["claims"]),
            "claims_kept": sum(c["verdict"] in ("SUPPORTED", "DISPUTED") for c in log["claims"]),
        },
    }


@mcp.tool()
async def research_status(run_id: str) -> str:
    """Progress of a run without waiting: stage, detail, elapsed seconds, whether it is done."""
    run = RUNS.get(run_id)
    if run is None:
        return json.dumps({"error": f"unknown run_id {run_id}"})
    return json.dumps(
        {
            "run_id": run_id,
            "done": run.done,
            "stage": run.stage,
            "detail": run.detail,
            "elapsed_s": round(time.time() - run.started),
        }
    )


@mcp.tool()
async def corpus_status() -> str:
    """Size of the local passage index and the models configured for the pipeline."""
    s = Settings()
    return json.dumps(
        {
            "index": PassageIndex(s.index_path, s.embedding_model).stats(),
            "model_backend": "mcp-client" if s.mcp_llm == "client" else "ollama",
            "text_model": s.text_model,
            "verifier_model": s.verifier_model or "(same as text)",
            "embedding_model": s.embedding_model,
            "reranker": s.reranker_model or s.reranker,
        }
    )


def main() -> int:
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
