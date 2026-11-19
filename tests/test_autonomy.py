"""Delegate and strict autonomy: how a failure is handled when the person has not chosen.

Under delegate (the default) a failure takes its labelled fallback and the request goes on, except
the library and the verifier, which still ask. Under strict every failure asks. A choice the person
made for the session always wins.
"""

from __future__ import annotations

import pytest

from research_mcp.brief import compact
from research_mcp.hub.contract import COMPONENTS
from research_pipeline.papers import PapersError
from tests.test_hub import Rig, assert_contract


# -- delegate: a failure takes its fallback ---------------------------------------------------
def test_delegate_takes_the_reranker_fallback_without_asking(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("research_mcp.ask.RETRIEVE_SECONDS", 0.05)
    rig = Rig(tmp_path)
    rig.hub.seen["reranker"] = "bge-reranker"
    rig.library.slow = True
    out = rig.ask("What is pink noise?", autonomy="delegate")
    assert out["status"] == "degraded"
    assert out["problems"][0]["status"] == "degraded"
    assert out["problems"][0]["component"] == "reranker"
    assert out["problems"][0]["fallback_used"] == COMPONENTS["reranker"].fallback
    assert "needs_decision" not in {p["status"] for p in out["problems"]}
    assert rig.ledger.decisions("s1") == {}  # nothing was stored for the session
    assert ("plain", "reranker") in rig.library.calls


def test_delegate_takes_the_embeddings_fallback_without_asking(tmp_path) -> None:
    rig = Rig(tmp_path)
    rig.library.error = PapersError("ollama embed failed: connection refused")
    out = rig.ask("What is pink noise?", autonomy="delegate")
    assert out["status"] == "degraded"
    assert out["problems"][0]["component"] == "embeddings"
    assert out["problems"][0]["status"] == "degraded"


def test_delegate_takes_the_provider_fallback_without_asking(tmp_path) -> None:
    rig = Rig(tmp_path)
    rig.world.balance = 0.5  # under the USD 1 threshold
    out = rig.ask("research: does pink noise improve sleep?", autonomy="delegate")
    assert out["status"] == "degraded"
    assert out["problems"][0]["component"] == "provider"
    assert [c[0] for c in rig.calls].count("pipeline__research_start") == 1


def test_delegate_takes_the_github_fallback_without_asking(tmp_path) -> None:
    rig = Rig(tmp_path)
    rig.github = {"authenticated": False, "detail": "run `gh auth login` on the host"}
    task = "Fix the failing test in tests/test_rag.py"
    out = rig.ask(task, repo="polarizetech/x", autonomy="delegate")
    assert out["status"] == "degraded"
    assert out["problems"][0]["component"] == "github"


def test_delegate_takes_the_corpus_fallback_without_asking(tmp_path) -> None:
    rig = Rig(tmp_path)
    rig.corpus = None
    out = rig.ask("research: is pink noise safe?", autonomy="delegate")
    assert out["status"] == "degraded"
    assert out["problems"][0]["component"] == "corpus"
    assert [c[0] for c in rig.calls].count("pipeline__research_start") == 1


# -- delegate: the library and the verifier still ask ------------------------------------------
def test_delegate_still_asks_when_the_library_fails(tmp_path) -> None:
    rig = Rig(tmp_path)
    rig.library.error = PapersError("passage index is locked")
    out = rig.ask("What is pink noise?", autonomy="delegate")
    assert_contract(out, "needs_decision", "library")


def test_delegate_still_asks_when_the_verifier_fails(tmp_path) -> None:
    rig = Rig(tmp_path)
    rig.world.models.remove("bespoke-minicheck:latest")
    out = rig.ask("research: is pink noise safe?", autonomy="delegate")
    assert_contract(out, "needs_decision", "verifier")


# -- strict: every failure asks ----------------------------------------------------------------
def test_strict_asks_when_the_reranker_fails(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("research_mcp.ask.RETRIEVE_SECONDS", 0.05)
    rig = Rig(tmp_path)
    rig.hub.seen["reranker"] = "bge-reranker"
    rig.library.slow = True
    out = rig.ask("What is pink noise?", autonomy="strict")
    assert_contract(out, "needs_decision", "reranker")
    assert ("plain", "reranker") not in rig.library.calls


# -- a choice the person made wins -------------------------------------------------------------
def test_a_session_stop_still_wins_over_delegate(tmp_path) -> None:
    rig = Rig(tmp_path)
    rig.library.error = PapersError("rerank model crashed")
    out = rig.ask("What is pink noise?", autonomy="delegate", decisions={"reranker": "stop"})
    assert_contract(out, "failed", "reranker")
    assert ("plain", "reranker") not in rig.library.calls


# -- the answer format follows the autonomy ----------------------------------------------------
def test_the_fresh_leads_instruction_follows_the_autonomy(tmp_path) -> None:
    rig = Rig(tmp_path)
    delegate = compact(rig.ask("What is pink noise?", autonomy="delegate"))["answer_format"]
    assert "fetch and index up to five of the closest leads with rag__index_paper" in delegate
    assert "then retrieve again" in delegate
    assert "list any lead that could not be fetched" in delegate
    assert "to fetch and read them" not in delegate

    strict = compact(rig.ask("What is pink noise?", autonomy="strict"))["answer_format"]
    assert "suggest asking again with `research:` to fetch and read them" in strict
    assert "rag__index_paper" not in strict
