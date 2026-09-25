"""The research pipeline as MCP tools, for federation behind the gateway as `pipeline__*`.

A run takes minutes on a local model, far longer than a chat client will hold a tool call open, so
the surface is start / status / result rather than one blocking call. The calling model needs to
make exactly one well-formed call to begin and one to collect: the multi-step work that small
models cannot sequence reliably happens inside the run, in code.

    research-pipeline-mcp          # stdio (or: python -m research_pipeline.mcp_server)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from mcp.server.fastmcp import FastMCP

from .config import Settings, load_env
from .index import PassageIndex
from .pipeline import Pipeline

load_env()
mcp = FastMCP(
    "research-pipeline",
    instructions=(
        "Evidence-first literature research. Call research_start with a scientific question, poll "
        "research_status until done, then call research_result. The result is assembled only from "
        "claims that were verified against retrieved full-text passages; treat its 'Limits' "
        "section "
        "as part of the answer and do not add facts from memory when relaying it."
    ),
)

RUNS: dict[str, dict[str, Any]] = {}
_active: asyncio.Task[None] | None = None


async def _execute(run_id: str, question: str, offline: bool) -> None:
    state = RUNS[run_id]

    def progress(stage: str, detail: str) -> None:
        state.update(stage=stage, detail=detail, updated=time.time())

    try:
        log = await Pipeline(Settings(), progress, offline=offline).run(question)
        state.update(
            done=True,
            stage="done",
            detail="",
            run_dir=log["run_dir"],
            answer=log["answer"],
            stats={
                "seconds": log["seconds"],
                "searches": len(log["searches"]),
                "evidence": len(log["evidence"]),
                "claims": len(log["claims"]),
                "claims_kept": sum(
                    c["verdict"] in ("SUPPORTED", "DISPUTED") for c in log["claims"]
                ),
            },
        )
    except Exception as exc:  # noqa: BLE001 - surface the failure to the caller, not a dead run
        state.update(done=True, stage="failed", detail=f"{type(exc).__name__}: {exc}"[:500])


@mcp.tool()
async def research_start(question: str, offline: bool = False) -> str:
    """Start a research run. Returns a run_id immediately; the run takes several minutes.

    Args:
        question: a scientific question answerable from published literature.
        offline: true = use only papers already indexed locally (no search, no downloads).
    """
    global _active  # noqa: PLW0603 - one local model serves one run; this is that guard
    if _active is not None and not _active.done():
        busy = next((k for k, v in RUNS.items() if not v.get("done")), "?")
        return json.dumps(
            {
                "error": "a run is already in progress; one local model serves one run",
                "active_run_id": busy,
            }
        )
    run_id = uuid.uuid4().hex[:10]
    RUNS[run_id] = {
        "question": question,
        "started": time.time(),
        "updated": time.time(),
        "stage": "starting",
        "detail": "",
        "done": False,
    }
    _active = asyncio.create_task(_execute(run_id, question, offline))
    return json.dumps({"run_id": run_id, "next": "poll research_status(run_id) every 30-60 s"})


@mcp.tool()
async def research_status(run_id: str) -> str:
    """Progress of a run: current stage, detail, elapsed seconds, and whether it is done."""
    state = RUNS.get(run_id)
    if state is None:
        return json.dumps(
            {
                "error": f"unknown run_id {run_id}; runs do not survive a server restart, "
                "but finished answers are kept under runs/"
            }
        )
    return json.dumps(
        {
            "run_id": run_id,
            "done": state["done"],
            "stage": state["stage"],
            "detail": state["detail"],
            "elapsed_s": round(time.time() - state["started"]),
        }
    )


@mcp.tool()
async def research_result(run_id: str) -> str:
    """The finished answer (markdown) with its verification statistics and run-log location."""
    state = RUNS.get(run_id)
    if state is None:
        return json.dumps({"error": f"unknown run_id {run_id}"})
    if not state["done"]:
        return json.dumps({"error": "not finished", "stage": state["stage"]})
    if state["stage"] == "failed":
        return json.dumps({"error": state["detail"]})
    return json.dumps(
        {"answer": state["answer"], "stats": state["stats"], "run_dir": state["run_dir"]}
    )


@mcp.tool()
async def corpus_status() -> str:
    """Size of the local passage index and the models configured for the pipeline."""
    s = Settings()
    return json.dumps(
        {
            "index": PassageIndex(s.index_path, s.embedding_model).stats(),
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
