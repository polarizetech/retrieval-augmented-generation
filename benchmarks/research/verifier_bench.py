"""Score claim verifiers against labelled (claim, passage) pairs. No LLM judge: labels are human.

    python benchmarks/research/verifier_bench.py
    python benchmarks/research/verifier_bench.py --models qwen3:4b-instruct-2507,bespoke-minicheck

A verifier is only worth running if it rejects what it should. The number that matters most is
false accepts: unsupported claims passed as SUPPORTED, because those reach the reader.

Each claim is scored the way the pipeline treats it: the model's verdict, overridden to
NOT_SUPPORTED when the claim states a number its passage does not contain (the pipeline removes
such a claim). The cases are synthetic passages, so they test the verifier, not a literature.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from research_pipeline import verify
from research_pipeline.config import Settings, load_env
from research_pipeline.llm import Ollama


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(Path(__file__).with_name("verifier_cases.jsonl")))
    parser.add_argument("--models", default="")
    args = parser.parse_args()
    settings = Settings()
    llm = Ollama(settings)
    names = [
        m.strip()
        for m in (args.models or settings.verifier_model or settings.text_model).split(",")
    ]
    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]

    for name in filter(None, names):
        model, digest = llm.resolve(name)
        right = false_accepts = 0
        started = time.time()
        for case in cases:
            verdict, _ = verify.judge(llm, model, case["claim"], case["passage"])
            missing = verify.unsupported_numbers(case["claim"], [case["passage"]])
            final = "NOT_SUPPORTED" if verdict == "SUPPORTED" and missing else verdict
            ok = final == case["expected"] or final in case.get("also_accept", [])
            right += ok
            false_accepts += final == "SUPPORTED" and case["expected"] != "SUPPORTED"
            overrode = " (model said SUPPORTED; number check overrode)" if final != verdict else ""
            mark = "ok  " if ok else "MISS"
            print(
                f"  {mark} {case['id']:<20} expected {case['expected']:<20} got {final}{overrode}"
            )
        print(
            f"{model} [{(digest or '')[:12]}]: {right}/{len(cases)} correct, "
            f"{false_accepts} false accept(s), {time.time() - started:.0f} s\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
