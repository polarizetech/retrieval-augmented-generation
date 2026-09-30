"""Command line.

    research-pipeline ask "Does regular aerobic exercise lower resting blood pressure in adults?"
    research-pipeline ask --domain cardiovascular "..."
    research-pipeline ask --offline "..."     # indexed corpus only: no search, no fetch
    research-pipeline ask --collection my-review "..."   # work in a paper-library collection
    research-pipeline index                   # have the library index every full text it holds
    research-pipeline domains                 # what fields are installed
    research-pipeline status
    research-pipeline novelty --id CAND-0007 "statement" --established "term" --queries A B C D

`python -m research_pipeline` works the same way.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from . import domains as domain_registry
from .config import Settings, load_env
from .papers import PaperLibrary


def _progress(stage: str, detail: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {stage:<9} {detail}", file=sys.stderr, flush=True)


async def _index(settings: Settings, limit: int | None) -> int:
    """Have the paper library index the full texts it holds (it skips what it has indexed)."""
    async with PaperLibrary(settings.gateway_config, settings.papers_upstream) as lib:
        held = [h["work"] for h in await lib.held() if h.get("full_text")][:limit]
        _progress("index", f"{len(held)} held with full text")
        for n in range(0, len(held), 10):
            report = await lib.index(held[n : n + 10])
            _progress(
                "index",
                f"{min(n + 10, len(held))}/{len(held)}: {report['passages_added']} new passages",
            )
        print(json.dumps((await lib.status())["passages"]))
    return 0


async def _status(settings: Settings) -> dict[str, object]:
    async with PaperLibrary(settings.gateway_config, settings.papers_upstream) as lib:
        library = await lib.status()
    return {
        "passage_index": library.get("passages"),
        "library_works": library.get("works"),
        "domain": settings.domain or "(generic)",
        "collection": settings.collection or None,
        "text_model": settings.text_model,
        "verifier_model": settings.verifier_model or "(same as text)",
        "reranker": settings.reranker,
    }


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
    ask.add_argument(
        "--collection", help="paper-library collection to work in (default PIPELINE_COLLECTION)"
    )
    ask.add_argument("--rounds", type=int, help="override PIPELINE_MAX_ROUNDS")
    idx = sub.add_parser("index", help="index the library's held full texts")
    idx.add_argument("--limit", type=int)
    nov = sub.add_parser(
        "novelty",
        help="adversarial prior-art search that reads the nearest papers; writes a dossier "
        "into the research repository",
    )
    nov.add_argument("statement", help="the candidate claim, written out in full")
    nov.add_argument("--id", required=True, dest="cid", help="candidate id, e.g. CAND-0007")
    nov.add_argument(
        "--established",
        nargs="+",
        default=[],
        help="the closest ESTABLISHED terminology (required; counts toward the five phrasings)",
    )
    nov.add_argument("--queries", nargs="+", default=[], help="further phrasings")
    nov.add_argument("--research-dir", help="research repository checkout (default $RESEARCH_REPO)")
    nov.add_argument("--offline", action="store_true", help="read the indexed corpus only")
    nov.add_argument("--domain", help="field policy, as for ask")
    sub.add_parser("domains", help="installed domain extensions and their status")
    sub.add_parser("status", help="the library's passage index and this configuration")
    args = parser.parse_args()

    settings = Settings()
    if args.cmd == "novelty":
        import os

        from . import novelty

        try:
            novelty.check_inputs(args.cid, args.statement, args.queries, args.established)
            research = novelty.research_dir(args.research_dir, dict(os.environ))
        except novelty.NoveltyError as exc:
            print(f"novelty: {exc}", file=sys.stderr)
            return 2
        probe = novelty.NoveltyProbe(settings, _progress, offline=args.offline, domain=args.domain)
        log = asyncio.run(probe.run(args.cid, args.statement, args.queries, args.established))
        out = novelty.write(log, research)
        print(f"{log['id']}: {log['verdict']} — {log['why']}")
        print(f"dossier: {out / 'prior-art.md'}", file=sys.stderr)
        return 0
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
        print(json.dumps(asyncio.run(_status(settings)), indent=1))
        return 0

    from .pipeline import Pipeline

    if args.rounds:
        settings.max_rounds = args.rounds
    pipe = Pipeline(
        settings, _progress, offline=args.offline, domain=args.domain, collection=args.collection
    )
    log = asyncio.run(pipe.run(args.question))
    print(log["answer"])
    print(f"run log: {log['run_dir']}  ({log['seconds']} s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
