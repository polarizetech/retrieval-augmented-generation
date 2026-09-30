"""What the classifier decided about each paper: its study design and population.

Evidence-side judgments belong to this application, not to the paper library, so they live in a
small SQLite file of their own (`PIPELINE_DATA_DIR/notes.sqlite`). A note is keyed by the paper's
work id and the SHA-256 of the text it was made from: when the library re-indexes a changed text,
the old note no longer applies and the paper is classified again.
"""

from __future__ import annotations

import sqlite3
import weakref
from pathlib import Path

SCHEMA = """
create table if not exists paper_notes (
    work text primary key, text_sha256 text, study_type text, population text, model text
);
"""


class NoteStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Classification runs on a stage's own thread; model calls run elsewhere and never
        # touch this connection.
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.executescript(SCHEMA)
        self._closer = weakref.finalize(self, self.db.close)

    def get(self, work: str, text_sha256: str | None = None) -> dict[str, str] | None:
        row = self.db.execute(
            "select text_sha256, study_type, population, model from paper_notes where work=?",
            (work,),
        ).fetchone()
        if not row or (text_sha256 and row[0] and row[0] != text_sha256):
            return None
        return dict(zip(("study_type", "population", "model"), row[1:], strict=True))

    def set(
        self, work: str, study_type: str, population: str, model: str, text_sha256: str = ""
    ) -> None:
        with self.db:
            self.db.execute(
                "insert or replace into paper_notes values (?,?,?,?,?)",
                (work, text_sha256, study_type, population, model),
            )

    def close(self) -> None:
        self._closer()
