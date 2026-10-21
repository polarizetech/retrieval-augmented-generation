from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from research_pipeline.sessions import SessionStore


def test_session_store_writes_capsule_and_index(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "answer.md").write_text("answer")
    (run_dir / "run.json").write_text("{}")
    log = {
        "question": "What changed?",
        "run_dir": str(run_dir),
        "answer_sha256": "abc123",
        "answer": "# Answer\n\n## Bottom line\n\nA checked result.\n\n## Limits\n\nOne limit.",
        "claims": [{"text": "A checked result.", "verdict": "SUPPORTED", "supported_by": ["E1"]}],
        "companions": {"results": {"datasets": {"items": [{"id": "D1"}]}}},
        "notes": ["One limit."],
        "insufficient_subquestions": [],
    }
    metadata = {
        "client": "test/1",
        "external_session_id": "chat-7",
        "project": "demo",
        "strictness": "publication",
        "agent_route": {"roles": ["researcher"], "commit": "deadbeef"},
    }
    store = SessionStore(tmp_path / "sessions.sqlite")
    store.start(
        "r1",
        log["question"],
        client="test/1",
        project="demo",
        strictness="publication",
        external_session_id="chat-7",
        route=metadata["agent_route"],
    )
    capsule = store.finish("r1", log, metadata)

    text = capsule.read_text()
    assert "REVIEW REQUIRED" in text
    assert "A checked result." in text
    assert "`E1`" in text
    assert json.loads((run_dir / "session.json").read_text())["external_session_id"] == "chat-7"
    with sqlite3.connect(store.path) as db:
        assert db.execute("select status from sessions where run_id='r1'").fetchone() == ("done",)
