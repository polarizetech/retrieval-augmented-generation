"""The passage index: chunking, lexical and dense retrieval, and its guarantees about offsets."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

from research_pipeline.index import PassageIndex, chunk, fts_query
from tests.conftest import PAPER, fake_embed


class TestChunking:
    def test_offsets_index_into_the_original_text(self) -> None:
        pieces = chunk(PAPER)
        assert len(pieces) > 1
        for start, end, body in pieces:
            assert PAPER[start:end] == body

    def test_the_reference_list_is_not_indexed(self) -> None:
        assert not any("Someone A." in body for _, _, body in chunk(PAPER))

    def test_fragments_too_short_to_ground_a_claim_are_dropped(self) -> None:
        assert chunk("Too short.\n\nAlso short.") == []


def test_fts_query_drops_stopwords_and_cannot_inject_syntax() -> None:
    query = fts_query('Does the "pressure" OR rate NEAR(x) change?')
    assert '"the"' not in query
    assert '"pressure"' in query
    assert all(part.startswith('"') and part.endswith('"') for part in query.split(" OR "))


class TestIndex:
    def test_hybrid_search_ranks_the_matching_passage_first(self, index: PassageIndex) -> None:
        hits = index.search("lowered systolic pressure", fake_embed(["lowered"])[0], limit=3)
        assert "lowered" in hits[0].text
        assert hits[0].bm25_rank is not None
        assert hits[0].dense_rank is not None

    def test_lexical_search_works_without_a_query_vector(self, index: PassageIndex) -> None:
        hits = index.search("heart rate", None, limit=3)
        assert hits
        assert all(h.dense_rank is None for h in hits)

    def test_reindexing_identical_text_is_a_no_op(self, index: PassageIndex) -> None:
        assert index.add({"work": "W1"}, PAPER, fake_embed) == 0

    def test_changed_text_replaces_the_earlier_copy(self, index: PassageIndex) -> None:
        before = index.stats()["passages"]
        added = index.add({"work": "W1"}, PAPER.replace("5 mmHg", "6 mmHg"), fake_embed)
        assert added == before
        assert index.stats() == {"papers": 1, "passages": before}

    def test_restricting_to_unknown_works_returns_nothing(self, index: PassageIndex) -> None:
        assert index.search("pressure", None, works=["W404"]) == []

    def test_passages_are_addressable_by_position(self, index: PassageIndex) -> None:
        first = index.passage_at("W1", 0)
        assert first is not None
        assert first.ord == 0
        assert index.passage(first.id) == first
        assert index.passage_at("W1", 999) is None

    def test_paper_metadata_round_trips(self, index: PassageIndex) -> None:
        paper = index.paper("W1")
        assert paper is not None
        assert paper["authors"] == ["Ada Lovelace", "Alan Turing"]
        assert paper["is_retracted"] is False
        assert index.paper("W404") is None

    def test_notes_are_stored_and_cleared_with_the_paper(self, index: PassageIndex) -> None:
        index.set_note("W1", "primary_human", "adults", "model-x")
        assert index.note("W1") == {
            "study_type": "primary_human",
            "population": "adults",
            "model": "model-x",
        }
        index.add(
            {"work": "W1"},
            PAPER + "\n\nAn addendum long enough to change the hash. " * 3,
            fake_embed,
        )
        assert index.note("W1") is None

    def test_a_different_embedding_model_is_refused(
        self, index: PassageIndex, tmp_path: Path
    ) -> None:
        with pytest.raises(RuntimeError, match="not comparable"):
            PassageIndex(tmp_path / "passages.sqlite", "another-model")


class _CommitsBetweenReadAndWrite:
    """Wraps a connection so another writer runs right after `_remove` reads a work's passages."""

    def __init__(self, db: Any, other_writer: threading.Thread) -> None:
        self._db = db
        self._other = other_writer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)

    def __enter__(self) -> Any:
        return self._db.__enter__()

    def __exit__(self, *exc: Any) -> Any:
        return self._db.__exit__(*exc)

    def execute(self, sql: str, *args: Any) -> Any:
        cur = self._db.execute(sql, *args)
        if sql.startswith("select id, text from passages where work=?") and not self._other.ident:
            self._other.start()
            # Long enough for an unblocked writer to commit; a writer that has to wait for this
            # connection's lock just keeps waiting, and is joined below.
            self._other.join(timeout=0.5)
        return cur


def test_two_processes_indexing_the_same_work_leave_one_consistent_copy(tmp_path: Path) -> None:
    """Regression for 2026-09-25: `research-pipeline index` and a concurrent `ask` both indexed
    the same work. The second writer read the work's passages before the first committed, then
    deleted the first writer's passages but not their vectors, and its own passages reused those
    rowids: UNIQUE constraint failed: vectors.passage_id."""
    path = tmp_path / "passages.sqlite"
    first = PassageIndex(path, "fake-embed")
    second = PassageIndex(path, "fake-embed")
    errors: list[sqlite3.Error] = []

    def first_writes() -> None:
        try:
            first.add({"work": "W1"}, PAPER, fake_embed)
        except sqlite3.Error as exc:  # surfaced in the main thread below
            errors.append(exc)

    other = threading.Thread(target=first_writes)
    second.db = _CommitsBetweenReadAndWrite(second.db, other)  # type: ignore[assignment]
    second.add({"work": "W1"}, PAPER, fake_embed)
    other.join()
    assert errors == []

    db = first.db
    n_passages = db.execute("select count(*) from passages").fetchone()[0]
    assert n_passages == len(chunk(PAPER))
    assert db.execute(
        "select count(*) from vectors where passage_id not in (select id from passages)"
    ).fetchone() == (0,)
    assert db.execute(
        "select count(*) from passages where id not in (select passage_id from vectors)"
    ).fetchone() == (0,)
    assert first.stats() == {"papers": 1, "passages": n_passages}
