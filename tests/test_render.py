"""How the answer presents what the search for null results found."""

from __future__ import annotations

import pytest

from research_pipeline.index import PassageIndex
from research_pipeline.pipeline import State
from research_pipeline.render import render
from research_pipeline.schema import Claim, Plan, SubQuestion
from tests.conftest import make_evidence

MODELS = {"text": {"name": "m"}, "verifiers": [{"name": "m"}]}


def answer(index: PassageIndex, direction: str) -> str:
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
    return render(st, index, MODELS)


def conflicting_section(text: str) -> str:
    return text.split("## Conflicting or negative evidence")[1].split("## Limits", maxsplit=1)[0]


@pytest.mark.parametrize("direction", ["denies", "mixed"])
def test_a_null_result_is_shown_as_opposing_evidence(index: PassageIndex, direction: str) -> None:
    section = conflicting_section(answer(index, direction))
    assert "Found by searching specifically for null results" in section
    assert "No opposing result was retrieved" not in section


def test_a_supporting_result_from_the_null_search_is_not_called_opposing(
    index: PassageIndex,
) -> None:
    section = conflicting_section(answer(index, "affirms"))
    assert "No opposing result was retrieved" in section
    assert "found supporting results instead" in section
    assert "Found by searching specifically" not in section
