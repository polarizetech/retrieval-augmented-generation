from pathlib import Path

import pytest

from research_mcp.routing import classify, load_rules, profile_override

ROOT = Path(__file__).resolve().parents[1]
RULES = load_rules(ROOT / "config" / "routing.default.toml")


@pytest.mark.parametrize(
    ("request_text", "mode"),
    [
        # A question never recruits the coder, even about code.
        ("What is pink noise?", "quick"),
        ("How does the prompt hook work?", "quick"),
        ("How do I fix the failing test?", "quick"),
        ("Should we refactor the router?", "quick"),
        ("Why does research_mcp/rag.py over-fetch?", "quick"),
        ("Add a section on thermal noise to the brief", "quick"),
        # A task on code, or a code review, does.
        ("Fix the failing test in tests/test_rag.py", "code"),
        ("Can you fix the bug in the gateway?", "code"),
        ("Please review this PR", "code"),
        ("Can you do a code review of the router?", "code"),
        ("Implement stochastic resonance in the simulator", "code"),
        ("Let's refactor the router", "code"),
        ("Add a --dry-run flag to the cli", "code"),
        ("Build a noise generator app", "code"),
        # Evidence questions run the research pipeline.
        ("Write a summary of the literature on 1/f noise", "research"),
        ("What does the evidence say about stochastic resonance in hearing?", "research"),
        ("Is it true that pink noise improves sleep?", "research"),
        ("What do papers say about binaural beats?", "research"),
        # A noun alone is not an evidence request: it gets one retrieval.
        ("Find papers on pink noise", "quick"),
        ("Cite the study about pink noise and sleep", "quick"),
        ("Is there a mechanism for stochastic resonance?", "quick"),
        ("is it true that pink noise improves sleep", "research"),
        ("Is there a meta-analysis of binaural beats?", "research"),
        ("research: find papers on pink noise", "research"),
        ("deep: pink noise", "research"),
        # Asking for a paper to be fetched and indexed skips the pipeline.
        ("get the paper 10.1038/xyz", "fetch"),
        ("fetch: some title", "fetch"),
        ("index this paper 10.1000/abc", "fetch"),
        ("download the paper on binaural beats", "fetch"),
        ("get 10.1000/abc", "fetch"),
        ("index 10.1000/abc", "fetch"),
    ],
)
def test_each_request_recruits_only_what_it_needs(request_text: str, mode: str) -> None:
    assert classify(request_text, RULES).mode == mode


def test_a_doi_in_a_code_task_is_not_fetched() -> None:
    assert classify("fix the DOI 10.1000/abc in tests/x.py", RULES).mode != "fetch"


def test_prefixes_and_an_explicit_mode_win_and_say_why() -> None:
    route = classify("research: pink noise and memory", RULES)
    assert (route.mode, route.question) == ("research", "pink noise and memory")
    assert "starts with 'research:'" in route.reason
    assert classify("code: tidy the README", RULES).mode == "code"
    assert classify("What is pink noise?", RULES, mode="research").mode == "research"
    with pytest.raises(ValueError, match="mode must be one of"):
        classify("x", RULES, mode="everything")


def test_a_repo_turns_a_task_into_code_work() -> None:
    assert classify("Add a note about the release", RULES).mode == "quick"
    assert classify("Add a note about the release", RULES, repo="polarizetech/x").mode == "code"


def test_local_words_extend_the_defaults_and_pick_a_specialist(tmp_path: Path) -> None:
    local = tmp_path / "routing.toml"
    local.write_text(
        '[prefixes]\n"lit:" = "research"\n'
        '[research]\ntriggers = ["what do we know"]\n'
        '[profiles]\nnoise-engineering = ["hiss"]\n'
    )
    rules = load_rules(ROOT / "config" / "routing.default.toml", local)
    assert classify("lit: tinnitus maskers", rules).mode == "research"
    assert classify("What do we know about tape hiss?", rules).mode == "research"
    assert "is it true" in rules.research  # the defaults stay
    picked = profile_override("Where does tape hiss come from?", rules)
    assert picked == ("noise-engineering", "hiss")
    assert profile_override("What is a mixer?", rules) is None
    local.write_text('[prefixes]\n"x:" = "everything"\n')
    with pytest.raises(ValueError, match="unknown modes"):
        load_rules(ROOT / "config" / "routing.default.toml", local)
