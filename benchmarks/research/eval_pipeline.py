"""End-to-end regression cases for the research pipeline, scored from run logs with no LLM judge.

    python benchmarks/research/eval_pipeline.py                  # offline, indexed corpus only
    python benchmarks/research/eval_pipeline.py --online         # allow search and fetching
    python benchmarks/research/eval_pipeline.py --only exercise-blood-pressure
    python benchmarks/research/eval_pipeline.py --score-only     # re-score the last results file

Offline is the default so that a change in score reflects a change in the pipeline, not in what
the search providers returned that day. Index the expected papers first (research-pipeline index).

Measured per case:
  source_recall   expected papers that ended up as evidence (retrieval, reranking and extraction)
  kept / drafted  claims that survived verification
  conflict_shown  contradiction cases: first-hand results in both directions, from different papers
  abstained       unanswerable cases: no claim survived (answering anyway is the failure)
  forbidden       kept claims matching a must_not_claim pattern (e.g. causal language for an
                  observational association)

A handful of cases measures nothing statistically: with n = 10 a proportion's 95% interval is
about +/-30 points. These cases are a smoke test and a template, not an accuracy estimate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from research_pipeline.config import ROOT, Settings, load_env
from research_pipeline.pipeline import Pipeline

KEPT = ("SUPPORTED", "DISPUTED")


def score(case: dict[str, Any], log: dict[str, Any]) -> dict[str, Any]:
    kept = [c for c in log["claims"] if c["verdict"] in KEPT]
    evidence = log["evidence"]
    papers = log.get("papers", {})
    evidence_dois = {(papers.get(e["work"], {}).get("doi") or "").lower() for e in evidence}
    expected = [d.lower() for d in case.get("expect_dois", [])]
    found = [d for d in expected if d in evidence_dois]
    row: dict[str, Any] = {
        "id": case["id"],
        "kind": case["kind"],
        "seconds": log["seconds"],
        "drafted": len(log["claims"]),
        "kept": len(kept),
        "source_recall": round(len(found) / len(expected), 2) if expected else None,
    }
    if case.get("expect_conflict"):
        first_hand = [e for e in evidence if e["role"] == "E"]
        yes = {e["work"] for e in first_hand if e["direction"] == "affirms"}
        no = {e["work"] for e in first_hand if e["direction"] == "denies"}
        row["conflict_shown"] = bool(yes and no and yes != no)
    if not case.get("answerable", True):
        row["abstained"] = not kept
    patterns = [re.compile(p, re.IGNORECASE) for p in case.get("must_not_claim", [])]
    row["forbidden"] = sum(any(p.search(c["text"]) for p in patterns) for c in kept)
    row["pass"] = all(
        [
            row["source_recall"] in (None, 1.0),
            row.get("conflict_shown", True),
            row.get("abstained", True),
            row["forbidden"] == 0,
            bool(kept) or not case.get("answerable", True),
        ]
    )
    return row


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(Path(__file__).with_name("pipeline_cases.jsonl")))
    parser.add_argument("--only")
    parser.add_argument("--online", action="store_true", help="allow discovery and fetching")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    results_dir = ROOT / "benchmarks/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]

    if args.score_only:
        latest = sorted(results_dir.glob("research-pipeline-*.jsonl"))[-1]
        rows = [json.loads(line) for line in latest.read_text().splitlines()]
    else:
        out = results_dir / f"research-pipeline-{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
        rows = []
        for case in cases:
            print(f"== {case['id']}", file=sys.stderr, flush=True)
            pipeline = Pipeline(Settings(), offline=not args.online)
            log = asyncio.run(pipeline.run(case["question"]))
            row = score(case, log) | {
                "run_dir": log["run_dir"],
                "text_model": log["models"]["text"],
                "verifiers": log["models"]["verifiers"],
                "reranker": log["reranker"],
                "prompt_version": log["prompt_version"],
                "index": log["index"],
            }
            rows.append(row)
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        print(f"results: {out}", file=sys.stderr)

    header = f"{'case':<28}{'pass':<6}{'recall':<8}{'kept/drafted':<14}{'conflict':<10}"
    print(header + f"{'abstain':<9}{'forbid':<8}sec")
    for r in rows:
        print(
            f"{r['id']:<28}{'yes' if r['pass'] else 'NO':<6}{r['source_recall']!s:<8}"
            f"{str(r['kept']) + '/' + str(r['drafted']):<14}{r.get('conflict_shown', '-')!s:<10}"
            f"{r.get('abstained', '-')!s:<9}{r['forbidden']:<8}{r['seconds']}"
        )
    passed = sum(r["pass"] for r in rows)
    print(f"\n{passed}/{len(rows)} cases pass. n is far too small for a rate; see the docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
