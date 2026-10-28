"""Fault injection for the hub (docs/hub-plan.md, phase 1).

Every component is broken on purpose, and its fallback and message are checked: the first failure
in a session asks how to handle it, the answer holds for the rest of that session, and every
non-success reply carries the error contract with the problem first in the relay.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from research_mcp import bridge
from research_mcp.ask import Asker
from research_mcp.hub.contract import COMPONENTS, RESTART_HINT, Problem, error_result
from research_mcp.hub.health import Hub
from research_mcp.hub.ledger import Ledger, fingerprint
from research_pipeline.config import Settings
from research_pipeline.ladder import Ladder, make_ladder
from research_pipeline.llm import LLMError, ModelOutputError, Ollama
from research_pipeline.papers import PapersError
from research_pipeline.usage import Breaker, Meters, load_config, read_nearai, read_openrouter
from tests.test_ask import brief

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_FIELDS = {
    "status", "component", "what_failed", "impact", "fallback_used", "options",
    "recommendation", "retry_after",
}  # fmt: skip


def hub_config(tmp_path: Path) -> dict[str, Any]:
    key = tmp_path / "openrouter.key"
    key.write_text("test-key")
    return {
        "breaker": {"failures": 3, "cooldown_seconds": 120},
        "usage": {"ttl_seconds": 300},
        "providers": {
            "openrouter": {
                "kind": "openai",
                "label": "OpenRouter Qwen",
                "url": "https://openrouter.ai/api/v1",
                "model": "qwen/qwen3.8-27b",
                "key_file": str(key),
                "min_usd": 1.0,
                "max_used_fraction": 0.95,
            },
            "nearai": {
                "kind": "openai",
                "label": "NEAR AI direct",
                "url": "https://cloud-api.near.ai/v1",
                "model": "Qwen/Qwen3.8-27B",
                "key_file": str(tmp_path / "missing.key"),
                "reporting_file": str(tmp_path / "missing.env"),
                "min_usd": 1.0,
            },
            "local": {"kind": "ollama", "label": "local qwen3:4b", "model": "qwen3:4b"},
        },
        "ladders": {
            "planning": ["openrouter", "nearai", "local"],
            "writing": ["openrouter", "nearai", "local"],
        },
        "steps": {"plan": "planning", "synthesise": "writing"},
    }


class World:
    """The outside world the hub reads, with switches to break each part of it."""

    def __init__(self) -> None:
        self.balance = 20.0  # OpenRouter credits left
        self.ollama_up = True
        self.models = ["bge-m3:latest", "bespoke-minicheck:latest", "qwen3:4b"]
        self.reads: list[str] = []

    def fetch(self, url: str, headers: dict[str, str]) -> Any:
        self.reads.append(url)
        if url.endswith("/key"):
            return {"data": {"usage_daily": 0.5, "usage_weekly": 3.5, "usage_monthly": 9.0}}
        if url.endswith("/credits"):
            return {"data": {"total_credits": 30, "total_usage": 30 - self.balance}}
        if "/api/" in url:
            if not self.ollama_up:
                raise urllib.error.URLError("connection refused")
            if url.endswith("/api/ps"):
                return {"models": []}
            return {"models": [{"name": m} for m in self.models]}
        raise AssertionError(f"unexpected read {url}")


class Library:
    """The paper library as `ask` sees it: retrieve, retrieve with a part off, search."""

    def __init__(self) -> None:
        self.error: Exception | None = None
        self.slow = False
        self.reranker: str | None = "bge-reranker"
        self.retrieval = "hybrid"
        self.search_error: Exception | None = None
        self.calls: list[tuple[str, Any]] = []

    def found(self) -> dict[str, Any]:
        return {
            "retrieval": self.retrieval,
            "reranker": self.reranker,
            "n": 1,
            "excluded": [],
            "results": [{"evidence_id": "W1#p0.abcdef12", "text": "Pink noise.", "title": "T"}],
        }

    async def retrieve(self, query: str, limit: int, max_chars: int) -> dict[str, Any]:
        self.calls.append(("retrieve", query))
        if self.slow:
            await asyncio.sleep(1)
        if self.error is not None:
            raise self.error
        return self.found()

    async def plain(self, query: str, limit: int, max_chars: int, component: str) -> dict[str, Any]:
        self.calls.append(("plain", component))
        out = self.found()
        out["reranker"] = None
        return out

    async def search(self, query: str, limit: int) -> dict[str, Any]:
        self.calls.append(("search", query))
        if self.search_error is not None:
            raise self.search_error
        return {"results": [{"title": "A paper on noise", "year": 2020, "doi": "10.1/n"}]}


class Rig:
    def __init__(self, tmp_path: Path, *, switchable: bool = True):
        self.tmp = tmp_path
        self.world = World()
        self.library = Library()
        self.config = hub_config(tmp_path)
        self.settings = replace(
            Settings(),
            verifier_model="bespoke-minicheck:latest",
            strictness="grounded",
            endpoint_url="https://openrouter.ai/api/v1",
            endpoint_model="qwen/qwen3.8-27b",
        )
        self.ledger = Ledger(tmp_path / "hub.sqlite")
        self.connected = {"papers", "pipeline", "research", "repos"}
        self.errors: dict[str, str] = {}
        self.corpus: Path | None = tmp_path
        self.github = {"authenticated": True, "detail": "authenticated"}
        self.pipeline = True
        self.run_reply: dict[str, Any] = {"run_id": "r1", "status": "working"}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.hub = Hub(
            self.ledger,
            self.config,
            self.settings,
            upstreams=lambda: (self.connected, self.errors),
            call_tool=self.call_tool,
            has_tool=self.has_tool,
            corpus=lambda: self.corpus,
            fetch=self.world.fetch,
            post=self.embed,
            memory=lambda: self.free_memory,
        )
        self.hub.deadline = 5.0
        self.hub.research_retry = 0.05
        self.free_memory = 50
        # bge-m3 stand-in: each question's vector, set by a test; others are orthogonal.
        self.vectors: dict[str, list[float]] = {}
        public, private = tmp_path / "public", tmp_path / "private"
        brief(public, "researcher", "literature", "")
        brief(private, "noise-engineering", "noise colour pink brown", "Name the domain.")
        (tmp_path / "routing.toml").write_text("")
        self.asker = Asker(
            retrieve=self.library.retrieve,
            call_tool=self.call_tool,
            has_tool=self.has_tool,
            rules_paths=(ROOT / "config" / "routing.default.toml", tmp_path / "routing.toml"),
            profiles_dir=public,
            extensions_dir=private,
            hub=self.hub,
            search=self.library.search,
            plain_retrieve=self.library.plain,
            can_switch=lambda _c: switchable,
        )

    def embed(self, url: str, body: dict[str, Any]) -> Any:
        texts = body["input"]
        return {
            "embeddings": [
                self.vectors.get(t, [0.0] * i + [1.0] + [0.0] * (len(texts) - i))
                for i, t in enumerate(texts)
            ]
        }

    def has_tool(self, name: str) -> bool:
        if name == "pipeline__research_start":
            return self.pipeline
        return name == "repos__auth_status"

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        if name == "repos__auth_status":
            return self.github
        reply: Any = self.run_reply
        if isinstance(reply, list):  # a sequence of replies; the last one repeats
            reply = reply.pop(0) if len(reply) > 1 else reply[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def ask(self, question: str, **kw: Any) -> dict[str, Any]:
        kw.setdefault("session", "s1")
        return asyncio.run(self.asker.ask(question, **kw))


def assert_contract(out: dict[str, Any], status: str, component: str) -> None:
    assert out["status"] == status
    assert set(out["error"]) == CONTRACT_FIELDS
    assert out["error"]["component"] == component
    assert out["error"]["status"] == status


# -- the contract -----------------------------------------------------------------------------
def test_every_status_reads_plainly_and_carries_all_fields() -> None:
    for status in ("queued", "degraded", "refused", "failed", "needs_decision"):
        problem = Problem(status, "library", "It broke.", fallback_used="search", retry_after=30)
        assert set(problem.as_dict()) == CONTRACT_FIELDS
        assert problem.plain().startswith("**")
    result = error_result(Problem("failed", "upstream", "The pipeline died."))
    assert result.isError
    text = result.content[0].text  # type: ignore[union-attr]
    assert text.startswith("**Failed: an upstream process.** The pipeline died.")
    assert (
        json.loads(text.split("```json\n")[1].split("```")[0])["error"]["component"] == "upstream"
    )
    with pytest.raises(ValueError, match="unknown status"):
        Problem("broken", "library", "x")


# -- the ledger -------------------------------------------------------------------------------
def test_the_ledger_tracks_requests_runs_and_decisions_across_restarts(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "hub.sqlite")
    rid = ledger.start("What is  Pink noise?", session="s1", project="noise", route="research")
    assert [r["id"] for r in ledger.active()] == [rid]
    ledger.finish(rid, "working", run_id="run9", seconds=1.5)
    ledger.decide("s1", "reranker", "fallback")
    ledger.close()

    again = Ledger(tmp_path / "hub.sqlite")
    assert (again.get("run9") or {}).get("id") == rid  # by run id too
    assert again.decisions("s1") == {"reranker": "fallback"}
    assert again.decisions("s2") == {}
    reply = {"status": "done", "answer": "A.", "cost": {"cost_usd": 0.02, "providers": ["x"]}}
    assert again.follow_run("run9", reply) == rid
    row = again.get(rid)
    assert row is not None
    assert row["status"] == "done"
    assert row["result"]["answer"] == "A."
    assert row["cost"] == 0.02
    assert again.active() == []
    board = again.board()
    assert board["noise"]["counts"] == {"done": 1}
    assert board["noise"]["sessions"] == 1
    assert fingerprint("What is  Pink noise?") == fingerprint("what is pink NOISE")


# -- circuit breakers -------------------------------------------------------------------------
def test_a_breaker_opens_after_repeated_failures_then_probes_after_the_cool_down() -> None:
    now = [0.0]
    breaker = Breaker(failures=3, cooldown=120, clock=lambda: now[0])
    for _ in range(2):
        breaker.failure("boom")
    assert breaker.allow()
    breaker.failure("boom")
    assert breaker.state == "open"
    assert not breaker.allow()
    assert breaker.retry_after() == 120
    now[0] = 121
    assert breaker.state == "half-open"
    assert breaker.allow()
    breaker.failure("still broken")  # a failed probe restarts the cool-down
    assert not breaker.allow()
    now[0] = 300
    breaker.success()
    assert breaker.state == "closed"


# -- reranker ---------------------------------------------------------------------------------
def test_reranker_timeout_asks_once_per_session_then_retrieves_without_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("research_mcp.ask.RETRIEVE_SECONDS", 0.05)
    rig = Rig(tmp_path)
    rig.hub.seen["reranker"] = "bge-reranker"
    rig.library.slow = True

    first = rig.ask("What is pink noise?")
    assert_contract(first, "needs_decision", "reranker")
    assert first["relay"].startswith("**Your choice is needed: the reranker failed.**")
    assert 'decisions={"reranker": "<choice>"}' in first["relay"]
    assert {o["choice"] for o in first["error"]["options"]} == {"fallback", "stop"}
    assert (rig.ledger.get(first["request_id"]) or {}).get("status") == "needs_decision"

    second = rig.ask("What is pink noise?", decisions={"reranker": "fallback"})
    assert second["status"] == "degraded"
    assert ("plain", "reranker") in rig.library.calls
    assert "**Degraded: the reranker.**" in second["relay"]
    assert "retrieval without reranking" in second["relay"]
    assert "**Passages**" in second["relay"]

    # Same session, no decision passed: the stored choice applies without asking again.
    third = rig.ask("What is brown noise?")
    assert third["status"] == "degraded"
    # Another session is asked for itself.
    assert rig.ask("What is pink noise?", session="s2")["status"] == "needs_decision"


def test_a_reranker_the_library_dropped_by_itself_is_labelled(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    assert rig.ask("What is pink noise?")["status"] == "done"
    rig.library.reranker = None
    out = rig.ask("What is pink noise?")
    assert out["status"] == "degraded"
    assert out["problems"][0]["component"] == "reranker"
    assert "(detected)" in out["relay"]


def test_stop_means_stop_with_a_retry_time(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.library.error = PapersError("rerank model crashed")
    out = rig.ask("What is pink noise?", decisions={"reranker": "stop"})
    assert_contract(out, "failed", "reranker")
    assert out["error"]["retry_after"] == 120
    assert "Retry in about 120 s" in out["relay"]
    assert ("plain", "reranker") not in rig.library.calls


# -- embeddings -------------------------------------------------------------------------------
def test_embeddings_down_falls_back_to_lexical_retrieval(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.library.error = PapersError("ollama embed failed: connection refused")
    out = rig.ask("What is pink noise?", decisions={"embeddings": "fallback"})
    assert out["status"] == "degraded"
    assert ("plain", "embeddings") in rig.library.calls
    assert "**Degraded: embeddings (Ollama).**" in out["relay"]
    assert "lexical retrieval" in out["relay"]
    assert "misses paraphrases" in out["relay"]


def test_without_a_switch_embeddings_fall_back_to_scholarly_search(tmp_path: Path) -> None:
    rig = Rig(tmp_path, switchable=False)
    rig.library.error = PapersError("ollama embed failed")
    out = rig.ask("What is pink noise?", decisions={"embeddings": "fallback"})
    assert out["status"] == "degraded"
    assert "the library cannot switch off its embeddings yet" in out["relay"]
    assert "no verbatim passages" in out["relay"]
    assert "**Scholarly search results**" in out["relay"]


# -- paper library ----------------------------------------------------------------------------
def test_library_down_falls_back_to_scholarly_search_and_breaker_skips_it(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.library.error = PapersError("passage index is locked")
    first = rig.ask("What is pink noise?")
    assert_contract(first, "needs_decision", "library")

    out = rig.ask("What is pink noise?", decisions={"library": "fallback"})
    assert out["status"] == "degraded"
    assert "1. A paper on noise (2020, doi:10.1/n)" in out["relay"]
    assert "not citable" in out["relay"]

    rig.ask("What is pink noise?")  # third failure: the breaker opens
    tries = sum(c[0] == "retrieve" for c in rig.library.calls)
    skipped = rig.ask("What is pink noise?")
    assert sum(c[0] == "retrieve" for c in rig.library.calls) == tries  # not tried again
    assert "skipped for" in skipped["problems"][0]["what_failed"]


def test_library_and_its_search_both_down_fail_with_the_contract(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.library.error = PapersError("index gone")
    rig.library.search_error = PapersError("providers unreachable")
    out = rig.ask("What is pink noise?", decisions={"library": "fallback"})
    assert_contract(out, "failed", "library")
    assert out["relay"].startswith("**Failed: the paper library.**")
    assert RESTART_HINT in out["relay"]


# -- model provider ---------------------------------------------------------------------------
def test_low_credit_moves_research_down_the_ladder_after_asking(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.world.balance = 0.5  # under the USD 1 threshold
    first = rig.ask("research: does pink noise improve sleep?")
    assert_contract(first, "needs_decision", "provider")
    assert "OpenRouter is low ($0.50 left)" in first["error"]["what_failed"]
    assert not [c for c in rig.calls if c[0] == "pipeline__research_start"]

    out = rig.ask("research: does pink noise improve sleep?", decisions={"provider": "fallback"})
    assert out["status"] == "degraded"
    assert "NEAR AI direct, the next usable rung of the ladder" in out["relay"]
    assert [c[0] for c in rig.calls].count("pipeline__research_start") == 1


def test_no_usable_provider_fails_plainly(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.world.balance = 0.0
    rig.world.ollama_up = False
    for _ in range(3):
        rig.hub.breakers["provider:nearai"].failure("HTTP 503")
    out = rig.ask("research: does pink noise improve sleep?")
    assert_contract(out, "failed", "provider")
    assert "OpenRouter is exhausted" in out["error"]["what_failed"]
    assert "Ollama" in out["error"]["recommendation"]


# -- verifier ---------------------------------------------------------------------------------
def test_verifier_down_runs_at_explore_strictness_when_chosen(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.world.models.remove("bespoke-minicheck:latest")
    assert_contract(rig.ask("research: is pink noise safe?"), "needs_decision", "verifier")
    out = rig.ask("research: is pink noise safe?", decisions={"verifier": "fallback"})
    assert out["status"] == "degraded"
    start = [a for n, a in rig.calls if n == "pipeline__research_start"][-1]
    assert start["strictness"] == "explore"
    assert "claims marked unverified" in out["relay"]


def test_explore_strictness_does_not_need_the_verifier(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.world.models.remove("bespoke-minicheck:latest")
    assert rig.ask("research: is pink noise safe?", strictness="explore")["status"] == "working"


# -- an upstream process ----------------------------------------------------------------------
def test_pipeline_upstream_down_fails_with_the_restart_hint_without_asking(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline = False
    rig.errors = {"pipeline": "BrokenPipeError: gone"}
    out = rig.ask("research: is pink noise safe?", decisions={})
    assert_contract(out, "failed", "upstream")
    assert "BrokenPipeError" in out["error"]["what_failed"]
    assert RESTART_HINT in out["relay"]


def test_pipeline_dying_mid_call_and_a_full_pipeline_are_reported(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.run_reply = RuntimeError("closed resource")  # type: ignore[assignment]
    out = rig.ask("research: is pink noise safe?")
    assert_contract(out, "failed", "upstream")
    assert rig.hub.breakers["upstream"].count == 1

    rig.run_reply = {"error": "4 runs at once is the limit; wait for one to finish"}
    rig.hub.deadline = 0.2
    rig.hub.research_wait = 0.5
    full = rig.ask("research: is pink noise safe?")
    assert_contract(full, "queued", "queue")
    assert "The research pipeline is full (4 runs at once). Position 1 in line" in full["relay"]
    assert f"`hub__result` with id `{full['request_id']}`" in full["relay"]


# -- GitHub -----------------------------------------------------------------------------------
def test_github_down_keeps_code_work_in_the_local_checkout(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.github = {"authenticated": False, "detail": "run `gh auth login` on the host"}
    task = "Fix the failing test in tests/test_rag.py"
    assert_contract(rig.ask(task, repo="polarizetech/x"), "needs_decision", "github")
    rig.hub._github = None  # the cached check is re-read
    out = rig.ask(task, repo="polarizetech/x", decisions={"github": "fallback"})
    assert out["status"] == "degraded"
    assert "edits a local checkout only; no pull request" in out["relay"]


def test_code_work_on_the_same_repo_from_another_session_is_mentioned(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.ask("Fix the failing test in tests/test_rag.py", repo="polarizetech/x", session="a")
    out = rig.ask("Fix the flaky check in tests/test_rag.py", repo="polarizetech/x", session="b")
    assert "Another session (a)" in out["relay"]
    assert "own git worktree" in out["relay"]
    alone = rig.ask("Fix the docs in README.md", repo="polarizetech/x", session="b")
    assert "Overlap" not in alone["relay"]  # different files, and its own earlier request


# -- research corpus --------------------------------------------------------------------------
def test_corpus_missing_lets_research_continue_with_propagation_pending(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.corpus = None
    assert_contract(rig.ask("research: is pink noise safe?"), "needs_decision", "corpus")
    out = rig.ask("research: is pink noise safe?", decisions={"corpus": "fallback"})
    assert out["status"] == "degraded"
    assert "evidence propagation is left pending" in out["relay"]
    assert [c[0] for c in rig.calls].count("pipeline__research_start") == 1


# -- the whole gateway (the bridge reports it) ------------------------------------------------
def test_the_bridge_reports_a_dead_gateway_with_the_contract() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async def go() -> None:
        http = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
        result = await bridge.Upstream("http://127.0.0.1:9/", "t", http).call_tool("ask", {})
        assert result.isError
        text = result.content[0].text  # type: ignore[union-attr]
        assert text.startswith("**Failed: the gateway.**")
        assert "files can still be read; no grounded claims" in text
        assert RESTART_HINT in text

    asyncio.run(go())


def test_the_bridge_names_its_session() -> None:
    headers = bridge.Upstream("http://127.0.0.1:9/", "t").headers
    assert headers["X-Hub-Session"].startswith("bridge-")
    assert headers["X-Hub-Project"]


# -- health report ----------------------------------------------------------------------------
def test_the_health_report_covers_every_component(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    report = asyncio.run(rig.hub.report())
    assert set(report) == {
        "reranker", "embeddings", "library", "provider", "verifier", "upstream", "github",
        "corpus",
    }  # fmt: skip
    assert report["embeddings"]["state"] == "up"
    assert report["provider"]["detail"] == "openrouter"
    rig.world.ollama_up = False
    rig.hub._ollama = None
    rig.errors = {"datasets": "FileNotFoundError"}
    down = asyncio.run(rig.hub.report())
    assert down["embeddings"]["state"] == "down"
    assert down["verifier"]["detail"] == "Ollama does not answer"
    assert down["upstream"]["state"] == "down"
    assert set(COMPONENTS) >= set(report)


# -- usage meters -----------------------------------------------------------------------------
def test_openrouter_meter_reports_balance_burn_and_run_out(tmp_path: Path) -> None:
    world = World()
    spec = hub_config(tmp_path)["providers"]["openrouter"]
    reading = read_openrouter("openrouter", spec, world.fetch)
    assert reading.balance_usd == 20.0
    assert reading.usage_usd == {"day": 0.5, "week": 3.5, "month": 9.0}
    assert reading.burn_per_day == 0.5
    assert reading.days_left == 40.0
    assert reading.state == "ok"
    assert reading.short() == "OpenRouter: $20.00 left, $0.50/day, ~40 days"
    world.balance = 0.8
    assert read_openrouter("openrouter", spec, world.fetch).state == "low"
    world.balance = 0
    assert read_openrouter("openrouter", spec, world.fetch).state == "exhausted"


def test_a_key_limit_counts_as_used_fraction(tmp_path: Path) -> None:
    spec = hub_config(tmp_path)["providers"]["openrouter"]

    def fetch(url: str, headers: dict[str, str]) -> Any:
        if url.endswith("/key"):
            return {"data": {"limit": 10, "limit_remaining": 0.4, "usage_weekly": 7}}
        return {"data": {"total_credits": 100, "total_usage": 10}}

    reading = read_openrouter("openrouter", spec, fetch)
    assert reading.balance_usd == 0.4  # the key's limit is what is left to spend
    assert reading.used_fraction == 0.96
    assert reading.state == "low"


def test_meters_say_when_they_cannot_read(tmp_path: Path) -> None:
    spec = hub_config(tmp_path)["providers"]["nearai"]
    reading = read_nearai("nearai", spec, World().fetch)
    assert not reading.readable
    assert reading.short() == "NEAR AI: needs a reporting token"

    (tmp_path / "missing.env").write_text("NEARAI_ORG_ID=org1\nNEARAI_REPORTING_TOKEN=tok\n")
    with_token = read_nearai("nearai", spec, lambda url, h: {"balance": 0.5})
    assert with_token.state == "low"

    no_key = dict(hub_config(tmp_path)["providers"]["openrouter"], key_file=str(tmp_path / "x"))
    assert read_openrouter("openrouter", no_key, World().fetch).detail == "no key file"

    def unreachable(url: str, headers: dict[str, str]) -> Any:
        raise urllib.error.URLError("dns")

    spec = hub_config(tmp_path)["providers"]["openrouter"]
    assert read_openrouter("openrouter", spec, unreachable).state == "down"


def test_meter_readings_are_cached_and_make_one_relay_line(tmp_path: Path) -> None:
    world = World()
    now = [0.0]
    meters = Meters(hub_config(tmp_path), Settings(), fetch=world.fetch, clock=lambda: now[0])
    assert meters.line() == "Usage: not read yet (see hub__usage)"
    meters.all()
    reads = len(world.reads)
    meters.all()
    assert len(world.reads) == reads  # within the TTL
    now[0] = 301
    meters.read("openrouter")
    assert len(world.reads) > reads
    line = meters.line()
    assert line.startswith("Usage: OpenRouter: $20.00 left")
    assert "NEAR AI: needs a reporting token" in line
    assert "Ollama: ok" in line
    assert "test-key" not in line + json.dumps([r.as_dict() for r in meters.all()])


def test_the_relay_ends_with_the_request_id_and_usage_line(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.hub.meters.all()
    out = rig.ask("What is pink noise?")
    last = out["relay"].splitlines()[-1]
    assert last.startswith(f"_Request `{out['request_id']}` · Usage: OpenRouter: $20.00 left")


def test_the_hub_config_merges_the_operators_file_and_ships_the_agreed_ladders(
    tmp_path: Path,
) -> None:
    local = tmp_path / "hub.toml"
    local.write_text("[providers.openrouter]\nmin_usd = 5.0\n")
    config = load_config(ROOT / "config" / "hub.default.toml", local)
    assert config["providers"]["openrouter"]["min_usd"] == 5.0
    assert config["providers"]["openrouter"]["max_used_fraction"] == 0.95
    for step in ("planning", "extraction", "writing"):
        assert config["ladders"][step] == ["openrouter", "nearai", "local"]
    assert config["providers"]["local"]["model"].startswith("qwen3:4b-instruct-2507")


# -- provider ladders -------------------------------------------------------------------------
class Rung(Ollama):
    def __init__(self, settings: Settings, name: str, fail: Exception | None = None):
        super().__init__(settings)
        self.name, self.fail, self.asked = name, fail, 0

    def chat_json(self, task: str, system: str, user: str, schema: dict[str, Any], **_: Any):
        self.asked += 1
        if self.fail is not None:
            raise self.fail
        self.calls.append({"task": task, "model": self.name})
        return {"by": self.name}

    def chat_text(self, task: str, model: str, user: str, **_: Any) -> str:
        return self.chat_json(task, "", user, {})["by"]


def ladder(tmp_path: Path, world: World, **fails: Exception) -> tuple[Ladder, dict[str, Rung]]:
    config = hub_config(tmp_path)
    settings = replace(
        Settings(),
        endpoint_url="https://openrouter.ai/api/v1",
        endpoint_model="qwen/qwen3.8-27b",
        endpoint_key="k",
    )
    rungs = {n: Rung(settings, n, fails.get(n)) for n in ("openrouter", "nearai", "local")}
    meters = Meters(config, settings, fetch=world.fetch)
    return Ladder(settings, config, meters=meters, rungs=dict(rungs)), rungs


def test_a_failing_provider_moves_the_step_down_and_says_so(tmp_path: Path) -> None:
    model, _ = ladder(tmp_path, World(), openrouter=LLMError("HTTP 502"))
    assert model.chat_json("synthesise", "s", "u", {}) == {"by": "nearai"}
    assert model.ladder_log[0]["step"] == "writing"
    assert "writing ran on NEAR AI direct instead of OpenRouter Qwen" in model.notes[0]
    assert model.calls == [{"task": "synthesise", "model": "nearai"}]  # one shared call log
    assert model.breakers["provider:openrouter"].count == 1


def test_low_credit_skips_a_rung_without_calling_it(tmp_path: Path) -> None:
    world = World()
    world.balance = 0.5
    model, rungs = ladder(tmp_path, world)
    assert model.chat_json("plan", "s", "u", {}) == {"by": "nearai"}
    assert rungs["openrouter"].asked == 0
    assert "OpenRouter is low ($0.50 left)" in model.notes[0]


def test_the_local_model_is_the_last_rung_and_its_output_is_labelled(tmp_path: Path) -> None:
    model, _ = ladder(
        tmp_path, World(), openrouter=LLMError("402 out of credits"), nearai=LLMError("401")
    )
    assert model.chat_text("synthesise", "qwen/qwen3.8-27b", "u") == "local"
    assert "a weaker local model" in model.notes[0]


def test_every_rung_failing_names_each_reason(tmp_path: Path) -> None:
    model, _ = ladder(
        tmp_path,
        World(),
        openrouter=LLMError("502"),
        nearai=LLMError("401"),
        local=LLMError("ollama unreachable"),
    )
    with pytest.raises(LLMError, match=r"writing: every provider.*OpenRouter Qwen: failed: 502"):
        model.chat_json("synthesise", "s", "u", {})


def test_a_bad_answer_is_not_a_provider_failure(tmp_path: Path) -> None:
    model, rungs = ladder(tmp_path, World(), openrouter=ModelOutputError("no JSON"))
    with pytest.raises(ModelOutputError):
        model.chat_json("plan", "s", "u", {})
    assert rungs["nearai"].asked == 0


def test_an_open_breaker_skips_the_rung_until_the_cool_down(tmp_path: Path) -> None:
    model, rungs = ladder(tmp_path, World(), openrouter=LLMError("503"))
    for _ in range(3):
        model.chat_json("plan", "s", "u", {})
    assert rungs["openrouter"].asked == 3
    model.chat_json("plan", "s", "u", {})
    assert rungs["openrouter"].asked == 3  # open: not tried


def test_the_verifier_is_not_laddered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model, rungs = ladder(tmp_path, World())
    seen: list[str] = []
    monkeypatch.setattr(
        Ollama, "chat_text", lambda self, task, m, user, **_: seen.append(m) or "Yes"
    )
    assert model.chat_text("verify", "bespoke-minicheck:latest", "doc") == "Yes"
    assert seen == ["bespoke-minicheck:latest"]
    assert all(r.asked == 0 for r in rungs.values())


def test_a_ladder_is_used_only_for_an_endpoint_the_hub_knows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "hub.toml"
    config.write_text((ROOT / "config" / "hub.default.toml").read_text())
    monkeypatch.setenv("HUB_CONFIG", str(config))
    known = replace(
        Settings(), endpoint_url="https://openrouter.ai/api/v1", endpoint_model="qwen/qwen3.8-27b"
    )
    assert isinstance(make_ladder(known), Ladder)
    assert make_ladder(replace(known, endpoint_url="http://localhost:9/v1")) is None
    assert make_ladder(replace(known, ladder=False)) is None


# -- the gateway's hub tools ------------------------------------------------------------------
def test_the_gateway_lists_hub_tools_and_asks_through_the_contract(tmp_path: Path) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    from research_mcp import gateway

    config = tmp_path / "gateway.json"
    config.write_text(json.dumps({"upstreams": {}}))
    server, _ = gateway.build_server(config, "127.0.0.1", 0, tmp_path / "token", http=False)

    async def go() -> None:
        async with create_connected_server_and_client_session(server._mcp_server) as session:
            names = {t.name for t in (await session.list_tools()).tools}
            assert {"hub__status", "hub__board", "hub__result", "hub__usage"} <= names

            # No paper library connected: retrieval fails, and the reply asks how to go on.
            asked = await session.call_tool("ask", {"question": "What is pink noise?"})
            out = asked.structuredContent or {}
            assert out["status"] == "needs_decision"
            assert out["error"]["component"] == "library"
            assert out["relay"].startswith("**Your choice is needed: the paper library failed.**")

            status = (await session.call_tool("hub__status", {})).structuredContent or {}
            assert [r["id"] for r in status["active"]] == [out["request_id"]]
            assert status["active"][0]["session"].startswith("stdio-")
            assert status["active"][0]["session_inferred"] is True
            assert set(status["health"]) >= {"library", "provider", "github", "corpus"}
            assert status["queues"]["retrieval"]["limit"] == 2

            got = await session.call_tool("hub__result", {"id": out["request_id"]})
            assert (got.structuredContent or {})["status"] == "needs_decision"
            missing = await session.call_tool("hub__result", {"id": "nope"})
            assert missing.isError
            assert '"status": "refused"' in missing.content[0].text  # type: ignore[union-attr]

            board = (await session.call_tool("hub__board", {})).structuredContent or {}
            assert isinstance(board["processes"], list)
            assert sum(p["counts"].get("needs_decision", 0) for p in board["projects"].values())

            usage = (await session.call_tool("hub__usage", {})).structuredContent or {}
            assert usage["plans"]
            assert usage["line"].startswith("Usage:")

            bad = await session.call_tool(
                "ask", {"question": "x", "decisions": {"gateway": "fallback"}}
            )
            assert bad.isError
            assert "**Refused: the request.**" in bad.content[0].text  # type: ignore[union-attr]

    asyncio.run(go())
