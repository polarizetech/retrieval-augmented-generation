"""Run logs: one folder per run, even for runs started in the same second."""

from __future__ import annotations

import json
from pathlib import Path

from research_pipeline.config import Settings
from research_pipeline.pipeline import write_run


def test_runs_started_in_the_same_second_keep_separate_logs(tmp_path: Path) -> None:
    settings = Settings(runs_dir=tmp_path)
    logs = [
        {"question": "Same question?", "date": "2026-10-25T19:13:38Z", "answer": a, "n": a}
        for a in ("first", "second")
    ]
    dirs = [write_run(settings, log) for log in logs]
    assert dirs[0] != dirs[1]
    assert dirs[1].name.endswith("-2")
    assert [json.loads((d / "run.json").read_text())["n"] for d in dirs] == ["first", "second"]
