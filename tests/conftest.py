"""Shared fixtures. Every test runs offline: no Ollama, no paper library, no network."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from research_pipeline.index import PassageIndex
from research_pipeline.schema import Evidence

# A synthetic paper. The topic is deliberately ordinary; what matters is the structure: a result
# paragraph, a null-result paragraph, and a reference list that must not be indexed.
PAPER = (
    "Resting systolic blood pressure was measured in 48 adults before and after twelve weeks of "
    "supervised aerobic training. "
    * 8
    + "\n\n"
    + "Aerobic training lowered resting systolic blood pressure by 5 mmHg relative to control "
    "(p = 0.02). "
    * 10
    + "\n\n"
    + "Training did not change resting heart rate in the same participants (p = 0.41). " * 10
    + "\n\nReferences\n\n1. Someone A. An earlier trial of exercise and blood pressure. 2016."
)

Embed = Callable[[list[str]], list[list[float]]]


def fake_embed(texts: list[str]) -> list[list[float]]:
    """A two-dimensional bag of words: enough to make dense ranking deterministic."""
    return [
        [float(t.lower().count("lowered")) + 0.01, float(t.lower().count("heart")) + 0.01]
        for t in texts
    ]


@pytest.fixture
def index(tmp_path: Path) -> Iterator[PassageIndex]:
    idx = PassageIndex(tmp_path / "passages.sqlite", "fake-embed")
    idx.add(
        {
            "work": "W1",
            "doi": "10.1000/w1",
            "title": "Training and blood pressure",
            "year": 2020,
            "authors": ["Ada Lovelace", "Alan Turing"],
            "route": "fixture",
        },
        PAPER,
        fake_embed,
    )
    yield idx
    idx.db.close()


def make_evidence(eid: str, work: str, direction: str = "affirms", **overrides: Any) -> Evidence:
    fields: dict[str, Any] = {
        "id": eid,
        "subquestion": "S1",
        "work": work,
        "passage_id": 1,
        "start": 0,
        "end": 10,
        "direction": direction,
        "finding": "f",
        "quote": "q",
        "study_type": "primary_human",
        "population": "adults",
        "secondhand": False,
        "role": "E",
    }
    return Evidence(**(fields | overrides))
