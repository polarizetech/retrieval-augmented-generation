"""A thin answer says why it is thin, from the run's own records."""

from __future__ import annotations

from research_pipeline import thin
from research_pipeline.pipeline import State
from research_pipeline.render import render
from research_pipeline.schema import Candidate, Claim, Plan, SubQuestion
from tests.conftest import make_evidence

MODELS = {"text": {"name": "m"}, "verifiers": [{"name": "m"}]}


def state(works: list[str], label: str = "moderate", second: bool = False) -> State:
    """S1 has one kept claim per work; S2 has a claim only when `second`."""
    st = State("Does training lower blood pressure?")
    st.plan = Plan(
        st.question,
        "multi_paper_synthesis",
        st.question,
        [
            SubQuestion("S1", "Does it lower it?", "evidence", ["training blood pressure"]),
            SubQuestion("S2", "By how much?", "evidence", ["training blood pressure size"]),
            SubQuestion("F1", "Does it?", "falsification", ["training no effect"]),
        ],
    )
    for i, w in enumerate(works):
        st.evidence[f"S1-E{i}"] = make_evidence(f"S1-E{i}", w)
        st.claims.append(
            Claim(
                f"S1-C{i}",
                "S1",
                "c",
                [f"S1-E{i}"],
                verdict="SUPPORTED",
                supported_by=[f"S1-E{i}"],
                label=label,
            )
        )
    if second:
        st.evidence["S2-E1"] = make_evidence("S2-E1", works[0], subquestion="S2")
        st.claims.append(
            Claim(
                "S2-C1",
                "S2",
                "c",
                ["S2-E1"],
                verdict="SUPPORTED",
                supported_by=["S2-E1"],
                label=label,
            )
        )
    return st


def test_a_full_answer_is_not_thin() -> None:
    assert thin.diagnose(state(["W1", "W2", "W3"], second=True)) == []


def test_a_failing_provider_is_named_with_its_reason_and_remedy() -> None:
    st = state(["W1"], second=True)
    st.searches = [
        {
            "providers": {
                "openalex": {
                    "status": "unavailable",
                    "why": "OpenAlex rate limit reached (remaining $0.0)",
                }
            }
        },
        {
            "providers": {
                "openalex": {
                    "status": "unavailable",
                    "why": (
                        "not asked for 30 s, after: OpenAlex rate limit reached (remaining $0.0)"
                    ),
                },
                "pubmed": {
                    "status": "unavailable",
                    "why": "pubmed rate-limited us (HTTP 429); set NCBI_API_KEY",
                },
            }
        },
    ]
    lines = thin.diagnose(st)
    text = "\n".join(lines)
    assert "**openalex** did not answer 2 of 2 searches" in text
    assert "budget for the period is spent" in text
    assert "**pubmed** did not answer 1 of 2" in text
    assert "API key" in text
    assert "rest on 1 paper" in lines[0]


def test_an_empty_subquestion_says_where_its_trail_stopped() -> None:
    st = state(["W1", "W2", "W3"])
    st.evidence["S2-E1"] = make_evidence("S2-E1", "W4", "neutral", subquestion="S2")
    text = "\n".join(thin.diagnose(st))
    assert "“By how much?”" in text
    assert "read as neutral" in text
    assert "“training blood pressure size”" in text


def test_budget_and_closed_papers_are_counted() -> None:
    st = state(["W1"])
    for i, outcome in enumerate(["over_budget", "over_budget", "not_obtainable"]):
        st.candidates[f"k{i}"] = Candidate(f"k{i}", "t", 2020, {}, [], None, [], outcome=outcome)
    text = "\n".join(thin.diagnose(st, max_fetch=12))
    assert "at most 12 papers" in text
    assert "2 further candidate(s)" in text
    assert "1 relevant-looking paper(s)" in text


def test_no_technical_failure_says_so() -> None:
    text = "\n".join(thin.diagnose(state(["W1", "W2", "W3"])))
    assert "No search, fetch or verification failure was recorded" in text


def test_the_section_appears_only_when_thin() -> None:
    full = render(state(["W1", "W2", "W3"], second=True), {}, MODELS)
    sparse = render(state(["W1"]), {}, MODELS)
    assert "## Why this answer is thin" not in full
    assert "## Why this answer is thin" in sparse
    assert sparse.index("## Why this answer is thin") < sparse.index("## Findings")
