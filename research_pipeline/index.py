"""Local passage index: SQLite FTS5 (BM25) + dense vectors, fused with reciprocal rank fusion.

The paper library answers "which papers"; it matches terms and does not rank. This index answers
"which passage", which is what grounding needs. Every passage keeps its character offsets into the
text as indexed (the library's full text with hidden characters removed; see safety.clean), and the
paper row keeps that text's SHA-256, so a quoted sentence can be traced to the exact stored copy.

A few hundred papers is ~10^4 passages, so dense search is an exact numpy matrix product. There is
no approximate-nearest-neighbour index to tune, drift, or explain.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
create table if not exists meta (key text primary key, value text);
create table if not exists papers (
    work text primary key, doi text, pmid text, pmcid text, title text, year integer,
    authors text, is_retracted integer, oa_status text, license text, route text, format text,
    text_sha256 text, n_chars integer, indexed_at text
);
create table if not exists passages (
    id integer primary key, work text not null, ord integer not null,
    start integer not null, "end" integer not null, text text not null
);
create index if not exists passages_work on passages(work);
create virtual table if not exists passages_fts
    using fts5(text, content='passages', content_rowid='id');
create table if not exists vectors (passage_id integer primary key, vec blob not null);
create table if not exists paper_notes (
    work text primary key, study_type text, population text, model text
);
"""

REFERENCES_HEADING = re.compile(
    r"^\s*(references|bibliography|literature cited|works cited)\s*$", re.I
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "not",
        "no",
        "than",
        "then",
        "there",
        "these",
        "those",
    ]
)


@dataclass
class Passage:
    id: int
    work: str
    ord: int
    start: int
    end: int
    text: str
    bm25_rank: int | None = None
    dense_rank: int | None = None
    fused: float = 0.0


def chunk(text: str, target: int = 1100, hard_max: int = 1600) -> list[tuple[int, int, str]]:
    """Split full text into (start, end, passage) on paragraph then sentence boundaries.

    Offsets index into `text` unchanged, so text[start:end] == passage always holds. The reference
    list is excluded: a bibliography entry matches every query about its topic and supports nothing.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    for para in re.split(r"(\n\s*\n)", text):
        start, pos = pos, pos + len(para)
        if not para.strip():
            continue
        if REFERENCES_HEADING.match(para):
            break
        if len(para) <= hard_max:
            spans.append((start, pos))
            continue
        cursor = start
        for piece in SENTENCE_END.split(para):
            at = text.find(piece, cursor, pos)
            if at < 0:
                continue
            spans.append((at, at + len(piece)))
            cursor = at + len(piece)

    out: list[tuple[int, int, str]] = []
    cur_start: int | None = None
    cur_end = 0
    for start, end in spans:
        if cur_start is not None and end - cur_start > hard_max:
            out.append((cur_start, cur_end, text[cur_start:cur_end]))
            cur_start = None
        if cur_start is None:
            cur_start = start
        cur_end = end
        if cur_end - cur_start >= target:
            out.append((cur_start, cur_end, text[cur_start:cur_end]))
            cur_start = None
    if cur_start is not None:
        out.append((cur_start, cur_end, text[cur_start:cur_end]))
    return [(s, e, t) for s, e, t in out if len(t.strip()) >= 80]


def fts_query(query: str) -> str:
    terms = [w.lower() for w in WORD.findall(query)]
    terms = [t for t in dict.fromkeys(terms) if t not in STOPWORDS]
    return " OR ".join(f'"{t}"' for t in terms)


class PassageIndex:
    def __init__(self, path: Path, embedding_model: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Stages run in worker threads, one at a time; access is serial, never concurrent.
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        known = self.db.execute("select value from meta where key='embedding_model'").fetchone()
        if known and known[0] != embedding_model:
            raise RuntimeError(
                f"index at {path} was built with {known[0]!r}, not {embedding_model!r}; "
                "vectors from "
                "different models are not comparable. Point PIPELINE_DATA_DIR elsewhere or rebuild."
            )
        self.db.execute(
            "insert or ignore into meta values ('embedding_model', ?)", (embedding_model,)
        )
        self.db.commit()
        self._matrix: tuple[np.ndarray, np.ndarray] | None = None

    # -- writing ---------------------------------------------------------------------------
    def has(self, work: str, text_sha256: str | None = None) -> bool:
        row = self.db.execute("select text_sha256 from papers where work=?", (work,)).fetchone()
        return bool(row) and (text_sha256 is None or row[0] == text_sha256)

    def add(
        self,
        record: dict[str, Any],
        text: str,
        embed: Callable[[list[str]], list[list[float]]],
        batch: int = 16,
    ) -> int:
        """Index one paper's full text. Replaces any earlier copy of the same work."""
        work = record["work"]
        digest = hashlib.sha256(text.encode()).hexdigest()
        if self.has(work, digest):
            return 0
        pieces = chunk(text)
        vectors: list[list[float]] = []
        for i in range(0, len(pieces), batch):
            vectors.extend(embed([p[2] for p in pieces[i : i + batch]]))
        with self.db:
            self._remove(work)
            self.db.execute(
                "insert into papers values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    work,
                    record.get("doi"),
                    record.get("pmid"),
                    record.get("pmcid"),
                    record.get("title"),
                    record.get("year"),
                    json.dumps(record.get("authors") or []),
                    int(bool(record.get("is_retracted"))),
                    record.get("oa_status"),
                    record.get("license"),
                    record.get("route"),
                    record.get("format"),
                    digest,
                    len(text),
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                ),
            )
            for ord_, ((start, end, body), vec) in enumerate(zip(pieces, vectors, strict=True)):
                cur = self.db.execute(
                    'insert into passages (work, ord, start, "end", text) values (?,?,?,?,?)',
                    (work, ord_, start, end, body),
                )
                self.db.execute(
                    "insert into passages_fts (rowid, text) values (?,?)", (cur.lastrowid, body)
                )
                self.db.execute(
                    "insert into vectors values (?,?)", (cur.lastrowid, _normalise(vec).tobytes())
                )
        self._matrix = None
        return len(pieces)

    def _remove(self, work: str) -> None:
        rows = self.db.execute("select id, text from passages where work=?", (work,)).fetchall()
        for pid, body in rows:
            self.db.execute(
                "insert into passages_fts (passages_fts, rowid, text) values ('delete',?,?)",
                (pid, body),
            )
            self.db.execute("delete from vectors where passage_id=?", (pid,))
        self.db.execute("delete from passages where work=?", (work,))
        self.db.execute("delete from papers where work=?", (work,))
        self.db.execute("delete from paper_notes where work=?", (work,))

    # -- reading ---------------------------------------------------------------------------
    def paper(self, work: str) -> dict[str, Any] | None:
        cur = self.db.execute("select * from papers where work=?", (work,))
        row = cur.fetchone()
        if not row:
            return None
        out = dict(zip([c[0] for c in cur.description], row, strict=True))
        out["authors"] = json.loads(out["authors"] or "[]")
        out["is_retracted"] = bool(out["is_retracted"])
        return out

    def opening(self, work: str, max_chars: int = 2500) -> str:
        rows = self.db.execute(
            "select text from passages where work=? order by ord limit 4", (work,)
        )
        return "\n\n".join(r[0] for r in rows)[:max_chars]

    def note(self, work: str) -> dict[str, str] | None:
        row = self.db.execute(
            "select study_type, population, model from paper_notes where work=?", (work,)
        ).fetchone()
        return dict(zip(("study_type", "population", "model"), row, strict=True)) if row else None

    def set_note(self, work: str, study_type: str, population: str, model: str) -> None:
        with self.db:
            self.db.execute(
                "insert or replace into paper_notes values (?,?,?,?)",
                (work, study_type, population, model),
            )

    def stats(self) -> dict[str, int]:
        papers = self.db.execute("select count(*) from papers").fetchone()[0]
        passages = self.db.execute("select count(*) from passages").fetchone()[0]
        return {"papers": papers, "passages": passages}

    def passage(self, passage_id: int) -> Passage | None:
        row = self.db.execute(
            'select id, work, ord, start, "end", text from passages where id=?', (passage_id,)
        ).fetchone()
        return Passage(*row) if row else None

    def passage_at(self, work: str, ord_: int) -> Passage | None:
        """A passage by its position in its paper, which survives re-indexing the same text."""
        row = self.db.execute(
            'select id, work, ord, start, "end", text from passages where work=? and ord=?',
            (work, ord_),
        ).fetchone()
        return Passage(*row) if row else None

    def has_vectors(self) -> bool:
        return self.db.execute("select 1 from vectors limit 1").fetchone() is not None

    def _vectors(self) -> tuple[np.ndarray, np.ndarray]:
        if self._matrix is None:
            rows = self.db.execute(
                "select passage_id, vec from vectors order by passage_id"
            ).fetchall()
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            mat = (
                np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
                if rows
                else np.zeros((0, 1), dtype=np.float32)
            )
            self._matrix = (ids, mat)
        return self._matrix

    def search(
        self,
        query: str,
        query_vector: list[float] | None,
        *,
        works: Iterable[str] | None = None,
        limit: int = 40,
        pool: int = 100,
        rrf_k: int = 60,
    ) -> list[Passage]:
        """Hybrid retrieval. Lexical and dense rankings are fused by rank, not by score, because
        BM25 scores and cosine similarities are on unrelated scales."""
        allowed: set[int] | None = None
        if works is not None:
            marks = ",".join("?" * len(wl)) if (wl := list(works)) else "''"
            allowed = {
                r[0]
                for r in self.db.execute(f"select id from passages where work in ({marks})", wl)
            }
            if not allowed:
                return []

        ranks: dict[int, dict[str, int]] = {}
        match = fts_query(query)
        if match:
            rows = self.db.execute(
                "select rowid from passages_fts where passages_fts match ? "
                "order by bm25(passages_fts) limit ?",
                (match, pool * (4 if allowed is not None else 1)),
            ).fetchall()
            kept = [r[0] for r in rows if allowed is None or r[0] in allowed][:pool]
            for rank, pid in enumerate(kept, 1):
                ranks.setdefault(pid, {})["bm25"] = rank

        if query_vector is not None:
            ids, mat = self._vectors()
            if len(ids):
                scores = mat @ _normalise(query_vector)
                if allowed is not None:
                    scores = np.where(np.isin(ids, list(allowed)), scores, -np.inf)
                top = np.argsort(-scores)[:pool]
                for rank, i in enumerate(top, 1):
                    if np.isfinite(scores[i]):
                        ranks.setdefault(int(ids[i]), {})["dense"] = rank

        fused = sorted(
            ((sum(1.0 / (rrf_k + r) for r in rk.values()), pid, rk) for pid, rk in ranks.items()),
            key=lambda t: -t[0],
        )[:limit]
        out = []
        for score, pid, rk in fused:
            p = self.passage(pid)
            if p:
                p.bm25_rank, p.dense_rank, p.fused = (
                    rk.get("bm25"),
                    rk.get("dense"),
                    round(score, 5),
                )
                out.append(p)
        return out


def _normalise(vec: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm else arr
