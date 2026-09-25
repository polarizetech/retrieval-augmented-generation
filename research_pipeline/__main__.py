"""Command line.

    research-pipeline ask "Does regular aerobic exercise lower resting blood pressure in adults?"
    research-pipeline ask --domain cardiovascular "..."
    research-pipeline ask --offline "..."     # indexed corpus only: no search, no fetch
    research-pipeline index                   # index every full text the library holds
    research-pipeline domains                 # what fields are installed
    research-pipeline status

`python -m research_pipeline` works the same way.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from . import domains as domain_registry
from . import safety
from .config import Settings, load_env
from .index import PassageIndex
from .llm import Ollama
from .papers import PaperLibrary


def _progress(stage: str, detail: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {stage:<9} {detail}", file=sys.stderr, flush=True)


async def _index(settings: Settings, limit: int | None) -> int:
    llm = Ollama(settings)
    index = PassageIndex(settings.index_path, settings.embedding_model)
    async with PaperLibrary(settings.gateway_config, settings.papers_upstream) as lib:
        held = [h for h in await lib.held() if h.get("full_text")]
        todo = [h for h in held if not index.has(h["work"])][:limit]
        _progress("index", f"{len(held)} held with full text; {len(todo)} to index")
        for n, rec in enumerate(todo, 1):
            if rec.get("is_retracted"):
                _progress("index", f"skip retracted {rec['work']}")
                continue
            text = await lib.full_text(rec["work"])
            if hidden := safety.scan_hidden(text):
                _progress("index", f"{rec['work']}: {hidden}; stripped before indexing")
            added = await asyncio.to_thread(index.add, rec, safety.clean(text), llm.embed)
            _progress("index", f"{n}/{len(todo)} {rec['work']}: {added} passages")
    print(index.stats())
    return 0


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(prog="research-pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="run the full pipeline on one question")
    ask.add_argument("question")
    ask.add_argument(
        "--domain", help="field to research, e.g. cardiovascular (default PIPELINE_DOMAIN)"
    )
    ask.add_argument("--offline", action="store_true", help="use the indexed corpus only")
    ask.add_argument("--rounds", type=int, help="override PIPELINE_MAX_ROUNDS")
    idx = sub.add_parser("index", help="index the library's held full texts")
    idx.add_argument("--limit", type=int)
    sub.add_parser("domains", help="installed domain extensions and their status")
    sub.add_parser("status", help="index size and configuration")
    args = parser.parse_args()

    settings = Settings()
    if args.cmd == "index":
        return asyncio.run(_index(settings, args.limit))
    if args.cmd == "domains":
        rows = domain_registry.discover()
        print(
            json.dumps(rows, indent=1)
            if rows
            else "No domain extensions installed. Runs use the generic policy.\n"
            "Install one with: pip install -e domains/cardiovascular"
        )
        return 0 if all(r["status"] == "ok" for r in rows) else 1
    if args.cmd == "status":
        index = PassageIndex(settings.index_path, settings.embedding_model)
        print(
            {
                "index": index.stats(),
                "index_path": str(settings.index_path),
                "domain": settings.domain or "(generic)",
                "text_model": settings.text_model,
                "verifier_model": settings.verifier_model or "(same as text)",
                "embedding_model": settings.embedding_model,
                "reranker": settings.reranker,
            }
        )
        return 0

    from .pipeline import Pipeline

    if args.rounds:
        settings.max_rounds = args.rounds
    log = asyncio.run(
        Pipeline(settings, _progress, offline=args.offline, domain=args.domain).run(args.question)
    )
    print(log["answer"])
    print(f"run log: {log['run_dir']}  ({log['seconds']} s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
