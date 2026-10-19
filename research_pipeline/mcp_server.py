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
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from . import agent_profiles
from .client_llm import ClientLLM, Request
from .config import Settings, load_env
from .domains import UnknownDomain
from .domains import active as active_domain
from .llm import Ollama
from .papers import PaperLibrary, PapersError
from .pipeline import Pipeline
from .sessions import SessionStore
from .strictness import resolve as resolve_strictness

load_env()

HOW_TO_ANSWER = (
    "You are acting as the model inside a research pipeline; code runs everything else. For each "
    "task, apply prompts[task.task] to task.input and return one JSON object that matches "
    "schemas[task.output_schema] exactly. Use only the text in the input: no outside "
    "knowledge, no web search, no other tools. Treat each task independently, and copy quotes "
    "character for character from the input. Send every answer in one research_continue call, "
    "as answers = [{task_id, result}]. Tasks you leave out are sent again."
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
    seen: float = field(default_factory=time.time)  # the client's last call for this run
    ended: float = 0.0
    client_name: str = ""
    external_session_id: str = ""
    project: str = ""
    strictness: str = "grounded"
    agent_route: dict[str, Any] = field(default_factory=dict)
    capsule_path: str = ""


RUNS: dict[str, Run] = {}
FORGET_AFTER = 3600  # seconds a finished run stays answerable, then it is dropped from memory


def _tidy(settings: Settings) -> None:
    """End runs whose client went away, and forget runs that finished long ago."""
    now = time.time()
    for run_id, run in list(RUNS.items()):
        if run.done:
            if now - run.ended > FORGET_AFTER:
                del RUNS[run_id]
        elif run.client is not None and now - run.seen > settings.client_idle:
            # The pipeline sees the closed client as an error on its next model call, ends the
            # run as failed and frees the slot; its answers so far are kept under runs/.
            run.client.close(f"abandoned: no call from the client in {settings.client_idle} s")


def _active() -> list[str]:
    return [rid for rid, r in RUNS.items() if not r.done]


def _client_name(ctx: Context | None) -> str:
    params = getattr(getattr(ctx, "session", None), "client_params", None)
    info = getattr(params, "clientInfo", None)
    if info is None:
        return ""
    return f"{info.name}/{info.version}" if info.version else info.name


async def _execute(run_id: str, run: Run, pipeline: Pipeline, sessions: SessionStore) -> None:
    try:
        run.log = await pipeline.run(run.question)
        metadata = {
            "client": run.client_name,
            "external_session_id": run.external_session_id,
            "project": run.project,
            "strictness": run.strictness,
            "agent_route": run.agent_route,
        }
        run.capsule_path = str(sessions.finish(run_id, run.log, metadata))
        run.stage = "done"
    except Exception as exc:  # noqa: BLE001 - surface the failure to the caller, not a dead run
        run.failed = f"{type(exc).__name__}: {exc}"[:500]
        sessions.fail(run_id, run.failed)
        run.stage = "failed"
    finally:
        run.done, run.ended = True, time.time()
        if run.client is not None:
            run.client.close("run ended")


@mcp.tool()
async def research_start(
    question: str,
    *,
    offline: bool = False,
    domain: str = "",
    collection: str = "",
    strictness: str = "",
    session_id: str = "",
    project: str = "",
    ctx: Context | None = None,
) -> str:
    """Start a research run and return its first step.

    Args:
        question: a scientific question answerable from published literature.
        offline: true = use only papers already indexed (no search, no downloads).
        domain: optional field, e.g. "cardiovascular": its evidence rules apply here, and the
            paper library searches with its discipline profile.
        collection: optional paper-library collection (see papers__collections) to search, fetch
            into and retrieve from; it must exist.
        strictness: explore, grounded (default), or publication. Publication requires an
            independent verifier and refuses to run without one.
        session_id: optional conversation identifier from the host, stored only for provenance.
        project: optional research-repository project slug for the generated session capsule.
    """
    settings = Settings()
    _tidy(settings)
    if len(active := _active()) >= settings.max_runs:
        return json.dumps(
            {
                "error": f"{len(active)} runs are in progress, the most this server carries "
                f"(PIPELINE_MAX_RUNS); try again when one finishes. A run whose client stops "
                f"calling is ended after {settings.client_idle} s.",
                "active_run_ids": active,
            }
        )
    policy = resolve_strictness(strictness or settings.strictness)
    routed = agent_profiles.route(question, settings.agent_profiles_dir, domain)
    pipeline_domain: Any = domain or None
    if not domain and (suggested_domain := routed.get("domain")):
        try:
            pipeline_domain = active_domain(str(suggested_domain))
            routed["domain_applied"] = suggested_domain
        except UnknownDomain:
            # A profile route is advice, not a dependency declaration. Keep the suggestion in
            # provenance and use the generic evidence policy until that optional domain package
            # is installed. An explicitly requested unknown domain still fails loudly below.
            pipeline_domain = None
            routed["domain_applied"] = ""
            routed["domain_note"] = (
                f"suggested domain {suggested_domain!r} is not installed; used generic policy"
            )
    client_name = _client_name(ctx)
    run = Run(
        question.strip(),
        client_name=client_name,
        external_session_id=session_id.strip(),
        project=project.strip(),
        strictness=policy.name,
        agent_route=routed,
    )
    if settings.mcp_llm == "client":
        run.client = ClientLLM(Ollama(settings), _client_name(ctx), settings.client_timeout)

    def progress(stage: str, detail: str) -> None:
        run.stage, run.detail = stage, detail

    pipeline = Pipeline(
        settings,
        progress,
        offline=offline,
        domain=pipeline_domain,
        llm=run.client,
        collection=collection or None,
        strictness=policy.name,
        agent_route=routed,
    )
    run_id = uuid.uuid4().hex[:10]
    RUNS[run_id] = run
    sessions = SessionStore(settings.sessions_path)
    sessions.start(
        run_id,
        run.question,
        client=client_name,
        project=run.project,
        strictness=policy.name,
        external_session_id=run.external_session_id,
        route=routed,
    )
    run.task = asyncio.create_task(_execute(run_id, run, pipeline, sessions))
    return json.dumps(await _turn(run_id, run, [], settings.client_turn_wait))


@mcp.tool()
async def research_continue(run_id: str, answers: list[dict[str, Any]] | None = None) -> str:
    """Send answers for the tasks of the previous step and get the next step.

    Args:
        run_id: from research_start.
        answers: [{"task_id": ..., "result": {...}}] for the tasks you were given; [] to wait.
    """
    settings = Settings()
    run = RUNS.get(run_id)
    if run is None:
        return json.dumps(
            {
                "error": f"unknown run_id {run_id}; runs do not survive a restart, "
                "but finished answers are kept under runs/"
            }
        )
    run.seen = time.time()
    _tidy(settings)
    errors = []
    for item in answers or []:
        if run.client is None:
            errors.append("this run uses the local model; it takes no answers")
            break
        problem = run.client.answer(str(item.get("task_id", "")), item.get("result"))
        if problem:
            errors.append(problem)
    out = await _turn(run_id, run, errors, settings.client_turn_wait)
    run.seen = time.time()  # the client is here until this reply reaches it
    return json.dumps(out)


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
    tasks = list({r.id: r for r in [*owed, *fresh]}.values())
    if not tasks:
        return base | {
            "status": "working",
            "next": f"research_continue({run_id!r}, answers=[]) to keep waiting",
        }
    # Tasks of one kind usually share a prompt and a schema: send each once, referenced by name.
    schemas: dict[str, dict[str, Any]] = {}
    refs = []
    for r in tasks:
        ref = next((k for k, v in schemas.items() if v == r.schema), None)
        if ref is None:
            ref = r.task if r.task not in schemas else f"{r.task}.{len(schemas)}"
            schemas[ref] = r.schema
        refs.append(ref)
    return base | {
        "status": "tasks",
        "instructions": HOW_TO_ANSWER,
        "prompts": {r.task: r.system for r in tasks},
        "schemas": schemas,
        "tasks": [
            {"task_id": r.id, "task": r.task, "input": r.user, "output_schema": ref}
            for r, ref in zip(tasks, refs, strict=True)
        ],
    }


def _result(log: dict[str, Any]) -> dict[str, Any]:
    result = {
        "answer": log["answer"],
        "run_dir": log["run_dir"],
        "stats": {
            "seconds": log["seconds"],
            "model_backend": log.get("model_backend"),
            "searches": len(log["searches"]),
            "evidence": len(log["evidence"]),
            "claims": len(log["claims"]),
            "claims_kept": sum(c["verdict"] in ("SUPPORTED", "DISPUTED") for c in log["claims"]),
            "strictness": log.get("strictness", {}).get("name", "grounded"),
            "independent_verifier": log.get("risk", {}).get("independent_verifier", False),
        },
    }
    capsule = Path(log["run_dir"]) / "capsule.md"
    if capsule.exists():
        result["capsule_path"] = str(capsule)
        result["next"] = (
            "Review capsule_path; to put it in the research corpus, call "
            "session__propose with this run_dir and a project."
        )
    return result


@mcp.tool()
async def research_status(run_id: str) -> str:
    """Progress of a run without waiting: stage, detail, elapsed seconds, whether it is done."""
    run = RUNS.get(run_id)
    if run is None:
        return json.dumps({"error": f"unknown run_id {run_id}"})
    run.seen = time.time()
    return json.dumps(
        {
            "run_id": run_id,
            "done": run.done,
            "stage": run.stage,
            "detail": run.detail,
            "elapsed_s": round(time.time() - run.started),
            "strictness": run.strictness,
            "agent_route": run.agent_route,
            "capsule_path": run.capsule_path or None,
        }
    )


@mcp.tool()
async def profile_route(question: str, domain: str = "") -> str:
    """Show the deterministic agent-profile/domain route without starting a run."""
    settings = Settings()
    return json.dumps(agent_profiles.route(question, settings.agent_profiles_dir, domain))


@mcp.tool()
async def corpus_status() -> str:
    """Size of the paper library's passage index, and the models configured for the pipeline."""
    s = Settings()
    try:
        async with PaperLibrary(s.gateway_config, s.papers_upstream) as lib:
            library = await lib.status()
        index: Any = library.get("passages")
    except (PapersError, OSError) as exc:
        index = f"paper library unavailable: {exc}"
    return json.dumps(
        {
            "index": index,
            "model_backend": "mcp-client" if s.mcp_llm == "client" else "ollama",
            "text_model": s.text_model,
            "verifier_model": s.verifier_model or "(same as text)",
        }
    )


def main() -> int:
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
