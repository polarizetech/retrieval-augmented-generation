"""How the answer presents what the search for null results found."""

from __future__ import annotations

import pytest

from research_pipeline.pipeline import State
from research_pipeline.render import render
from research_pipeline.schema import Claim, Plan, SubQuestion
from tests.conftest import make_evidence

# What the paper library says about the cited work (render prints the reference from it).
LIBRARY = {
    "W1": {
        "doi": "10.1000/w1",
        "title": "Training and blood pressure",
        "year": 2020,
        "authors": ["Ada Lovelace", "Alan Turing"],
    }
}

MODELS = {"text": {"name": "m"}, "verifiers": [{"name": "m"}]}


def answer(direction: str) -> str:
    st = State("Does training lower blood pressure?")
    st.plan = Plan(
        st.question,
        "multi_paper_synthesis",
        st.question,
        [
            SubQuestion("S1", "Does it?", "evidence", ["q"]),
            SubQuestion("F1", "Does it?", "falsification", ["training no effect blood pressure"]),
        ],
    )
    st.evidence["F1-E1"] = make_evidence("F1-E1", "W1", direction, subquestion="F1")
    st.claims.append(
        Claim(
            "F1-C1",
            "F1",
            "A claim from the null-result search.",
            ["F1-E1"],
            verdict="SUPPORTED",
            supported_by=["F1-E1"],
            label="moderate",
        )
    )
    return render(st, LIBRARY, MODELS)


def conflicting_section(text: str) -> str:
    return text.split("## Conflicting or negative evidence")[1].split("## Limits", maxsplit=1)[0]


@pytest.mark.parametrize("direction", ["denies", "mixed"])
def test_a_null_result_is_shown_as_opposing_evidence(direction: str) -> None:
    section = conflicting_section(answer(direction))
    assert "Found by searching specifically for null results" in section
    assert "No opposing result was retrieved" not in section


def test_a_supporting_result_from_the_null_search_is_not_called_opposing() -> None:
    section = conflicting_section(answer("affirms"))
    assert "No opposing result was retrieved" in section
    assert "found supporting results instead" in section
    assert "Found by searching specifically" not in section


def test_a_claim_no_verifier_answered_is_counted_in_the_limits() -> None:
    from research_pipeline.verify import VERDICT_UNCHECKED

    st = State("Q?")
    st.plan = Plan(
        "Q?",
        "m",
        "Q?",
        [SubQuestion("S1", "Q?", "evidence", []), SubQuestion("F1", "Q?", "falsification", [])],
    )
    st.evidence = {"S1-E1": make_evidence("S1-E1", "W1")}
    st.claims = [
        Claim("S1-C1", "S1", "Training lowered pressure.", ["S1-E1"], verdict=VERDICT_UNCHECKED)
    ]
    text = render(st, LIBRARY, MODELS)
    limits = text.split("## Limits of this answer")[1]
    assert "1 drafted claim(s) were removed because the verifier returned no verdict" in limits
    assert "Training lowered pressure." not in text.split("## Limits")[0]
