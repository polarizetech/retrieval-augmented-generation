"""The novelty probe, offline: a scripted verifier over the fixture paper.

The verdict is the whole product, so each branch of it is pinned: a passage that states the
candidate is prior art only when its quote is in the stored text and its numbers are present; a
search that did not fully run can never produce CANDIDATE; and the dossier lands in the research
repository or nowhere.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from research_pipeline import novelty
from research_pipeline.config import Settings
from tests.conftest import PAPER, LibraryAdapter, hold_w1, papers_library

EFFECT = "Aerobic training lowered resting systolic blood pressure by 5 mmHg relative to control"
STATEMENT = "Aerobic training lowers resting systolic blood pressure in adults"
PHRASES = [
    "exercise training blood pressure",
    "aerobic exercise hypertension adults",
    "endurance training systolic pressure",
    "training lowers blood pressure",
]
ESTABLISHED = ["exercise antihypertensive effect"]


class Verifier:
    """Answers the verify task the way it is told to, and nothing else."""

    def __init__(self, verdict: str, closest: str = EFFECT) -> None:
        self.verdict, self.closest, self.calls = verdict, closest, []

    def resolve(self, model: str) -> tuple[str, str]:
        return model, "sha256:test"

    def chat_json(self, task: str, system: str, user: str, schema: dict, **_: Any) -> dict:
        self.calls.append({"task": task})
        assert task == "verify", task
        hit = "lowered" in user
        return {
            "closest_sentence": self.closest if hit else "",
            "claim_adds": "nothing" if hit else "everything",
            "verdict": self.verdict if hit else "NOT_SUPPORTED",
        }


def probe(tmp_path: Path, model: Verifier, statement: str = STATEMENT) -> dict[str, Any]:
    settings = Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        reranker="none",
        verifier_model="",
        text_model="stub-model",
    )
    library = papers_library(tmp_path)
    hold_w1(library)
    library.index_works()
    p = novelty.NoveltyProbe(settings, offline=True, llm=model)  # type: ignore[arg-type]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(novelty, "PaperLibrary", lambda *_: LibraryAdapter(library))
        return asyncio.run(p.run("CAND-0001", statement, PHRASES, ESTABLISHED))


@pytest.fixture
def research(tmp_path: Path) -> Path:
    root = tmp_path / "research"
    root.mkdir()
    for marker in novelty.RESEARCH_MARKERS:
        (root / marker).write_text("{}")
    return root


def test_a_passage_that_states_it_is_prior_art_with_its_quote(tmp_path: Path) -> None:
    log = probe(tmp_path, Verifier("SUPPORTED"))
    assert log["verdict"] == "PRIOR_ART"
    states = [r for r in log["readings"] if r["bearing"] == "states"]
    assert states
    assert all(r["quote"] and r["quote"] in PAPER for r in states)


def test_an_unanchored_quote_is_never_prior_art(tmp_path: Path) -> None:
    log = probe(tmp_path, Verifier("SUPPORTED", closest="A sentence the paper never wrote at all."))
    assert log["verdict"] == "PARTLY_KNOWN"
    assert not [r for r in log["readings"] if r["bearing"] == "states"]


def test_a_number_the_passage_lacks_downgrades_it(tmp_path: Path) -> None:
    log = probe(tmp_path, Verifier("SUPPORTED"), statement=STATEMENT + " by 9 mmHg")
    assert log["verdict"] == "PARTLY_KNOWN"
    assert any("9" in n for r in log["readings"] for n in r["missing_numbers"])


def test_a_narrower_version_is_partly_known(tmp_path: Path) -> None:
    assert probe(tmp_path, Verifier("PARTIALLY_SUPPORTED"))["verdict"] == "PARTLY_KNOWN"


def test_an_offline_run_can_never_be_a_candidate(tmp_path: Path) -> None:
    log = probe(tmp_path, Verifier("NOT_SUPPORTED"))
    assert log["verdict"] == "INCONCLUSIVE"
    assert "offline" in log["why"]


def test_a_refusing_index_blocks_candidate() -> None:
    v, why = novelty.decide([], {"core": ["unavailable: 429"]}, [], 8, offline=False)
    assert v == "INCONCLUSIVE"
    assert "core" in why


def test_too_few_works_read_blocks_candidate() -> None:
    v, why = novelty.decide([], {}, [], novelty.MIN_READ - 1, offline=False)
    assert v == "INCONCLUSIVE"
    assert "read in full" in why


def test_candidate_needs_every_index_and_enough_reading() -> None:
    v, _ = novelty.decide([], {}, [], novelty.MIN_READ, offline=False)
    assert v == "CANDIDATE"


@pytest.mark.parametrize(
    ("cid", "queries", "established", "msg"),
    [
        ("CAND-1", PHRASES[:2], ESTABLISHED, "at least 5"),
        ("CAND-1", PHRASES, [], "--established is required"),
        ("../x", PHRASES, ESTABLISHED, "letters, digits"),
        ("CAND-1", [PHRASES[0]] * 4, ESTABLISHED, "at least 5"),  # duplicates do not count
    ],
)
def test_inputs_are_refused(cid: str, queries: list[str], established: list[str], msg: str) -> None:
    with pytest.raises(novelty.NoveltyError, match=msg):
        novelty.check_inputs(cid, STATEMENT, queries, established)


def test_the_dossier_is_only_written_into_the_research_repository(tmp_path: Path) -> None:
    with pytest.raises(novelty.NoveltyError, match="RESEARCH_REPO"):
        novelty.research_dir(None, {})
    with pytest.raises(novelty.NoveltyError, match="not the research repository"):
        novelty.research_dir(str(tmp_path), {})


def test_the_dossier_lands_in_research_novelty(tmp_path: Path, research: Path) -> None:
    log = probe(tmp_path, Verifier("SUPPORTED"))
    out = novelty.write(log, novelty.research_dir(str(research), {}))
    assert out == research / "novelty" / "CAND-0001"
    md = (out / "prior-art.md").read_text()
    assert "## Verdict: PRIOR_ART" in md
    assert EFFECT in md
    assert "## How to kill this" in md
    runs = list((out / "runs").glob("*.json"))
    assert len(runs) == 1
    assert json.loads(runs[0].read_text())["verdict"] == "PRIOR_ART"


def test_the_dossier_never_announces_but_quotes_the_input_verbatim(
    tmp_path: Path, research: Path
) -> None:
    log = probe(tmp_path, Verifier("NOT_SUPPORTED"), statement="A novel discovery: " + STATEMENT)
    out = novelty.write(log, research)  # the operator's own words are quoted, not announced
    md = (out / "prior-art.md").read_text()
    assert "A novel discovery" in md
    assert not novelty.BANNED.search(novelty.template_text(md, log))


def test_the_keyword_rule_is_reported_beside_the_reading(tmp_path: Path) -> None:
    log = probe(tmp_path, Verifier("NOT_SUPPORTED"))
    assert "For comparison: the keyword rule" in novelty.render(log)
