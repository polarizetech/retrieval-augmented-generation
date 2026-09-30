"""The MCP client as the pipeline's model: hand-out, validation, batching and failure."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from research_pipeline.client_llm import ClientLLM
from research_pipeline.config import Settings
from research_pipeline.llm import LLMError, Ollama

SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"enum": ["YES", "NO"]}},
    "required": ["verdict"],
}


class LocalModel(Ollama):
    def __init__(self) -> None:
        super().__init__(Settings(text_model="local"))
        self.asked: list[str] = []

    def chat_json(
        self,
        task: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        model: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        self.asked.append(model or "")
        return {"verdict": "NO"}

    def embed(self, texts: list[str], **_: Any) -> list[list[float]]:
        return [[1.0] for _ in texts]


@pytest.fixture
def client() -> ClientLLM:
    return ClientLLM(LocalModel(), "TestClient/1.0", timeout=5)


def ask(client: ClientLLM, task: str = "verify") -> Any:
    return client.chat_json(task, "system prompt", "user input", SCHEMA)


def test_calls_wait_for_the_client_and_are_logged(client: ClientLLM) -> None:
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(ask, client)
        [request] = client.take(wait=2)
        assert (request.task, request.system, request.user) == (
            "verify",
            "system prompt",
            "user input",
        )
        assert client.answer(request.id, {"verdict": "YES"}) is None
        assert pending.result(timeout=2) == {"verdict": "YES"}
    assert client.calls[0]["model"] == "mcp-client:TestClient/1.0"


def test_concurrent_calls_are_handed_out_as_one_batch(client: ClientLLM) -> None:
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(ask, client) for _ in range(5)]
        batch = client.take(wait=2)
        assert len(batch) == 5
        for request in batch:
            client.answer(request.id, {"verdict": "NO"})
        assert [f.result(timeout=2) for f in futures] == [{"verdict": "NO"}] * 5


def test_a_turn_hands_out_at_most_limit_tasks(client: ClientLLM) -> None:
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(ask, client) for _ in range(5)]
        first = client.take(wait=2, limit=3)
        second = client.take(wait=2, limit=3)
        assert (len(first), len(second)) == (3, 2)
        for request in first + second:
            client.answer(request.id, {"verdict": "YES"})
        assert all(f.result(timeout=2) == {"verdict": "YES"} for f in futures)


def test_an_answer_that_breaks_the_schema_is_refused_and_still_owed(client: ClientLLM) -> None:
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(ask, client)
        [request] = client.take(wait=2)
        problem = client.answer(request.id, {"verdict": "MAYBE"})
        assert problem is not None
        assert "does not match output_schema" in problem
        assert client.outstanding() == [request]
        assert client.answer(request.id, {"verdict": "YES"}) is None
        assert pending.result(timeout=2) == {"verdict": "YES"}
    assert "unknown or already answered" in (client.answer(request.id, {"verdict": "NO"}) or "")


def test_a_client_that_keeps_breaking_the_schema_ends_the_call(client: ClientLLM) -> None:
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(ask, client)
        [request] = client.take(wait=2)
        replies = [client.answer(request.id, {"verdict": "MAYBE"}) for _ in range(3)]
        assert "cannot continue" in (replies[-1] or "")
        with pytest.raises(LLMError, match="3 answers broke the schema"):
            pending.result(timeout=2)
    assert client.outstanding() == []


def test_take_returns_nothing_when_nothing_is_asked(client: ClientLLM) -> None:
    assert client.take(wait=0.1) == []


def test_calls_naming_a_local_model_go_to_ollama(client: ClientLLM) -> None:
    assert client.chat_json("verify", "s", "u", SCHEMA, model="bespoke-minicheck") == {
        "verdict": "NO"
    }
    assert client.ollama.asked == ["bespoke-minicheck"]  # type: ignore[attr-defined]
    assert client.take(wait=0.1) == []
    assert client.resolve("mcp-client:x") == ("mcp-client:TestClient/1.0", None)


def test_an_abandoned_run_fails_instead_of_hanging(client: ClientLLM) -> None:
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(ask, client)
        client.take(wait=2)
        client.close("client went away")
        with pytest.raises(LLMError, match="client went away"):
            pending.result(timeout=2)
    with pytest.raises(LLMError, match="stopped answering"):
        ask(client)


def test_a_client_that_never_answers_times_out() -> None:
    client = ClientLLM(LocalModel(), timeout=0.2)
    with pytest.raises(LLMError, match="no answer from the MCP client"):
        ask(client)


def test_take_wakes_when_the_run_closes(client: ClientLLM) -> None:
    timer = threading.Timer(0.2, client.close)
    timer.start()
    assert client.take(wait=5) == []
    timer.join()
