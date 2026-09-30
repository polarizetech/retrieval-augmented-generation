"""The Ollama client's request handling, with HTTP replaced by canned responses."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any
from unittest import mock

import pytest

from research_pipeline.config import Settings
from research_pipeline.llm import LLMError, Ollama


class Stream:
    def __init__(self, lines: list[dict[str, Any]]):
        self.lines = [json.dumps(line).encode() + b"\n" for line in lines]

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def __iter__(self) -> Any:
        return iter(self.lines)


def body(payload: dict[str, Any]) -> mock.MagicMock:
    response = mock.MagicMock()
    response.__enter__.return_value = io.BytesIO(json.dumps(payload).encode())
    return response


@pytest.fixture
def ollama() -> Ollama:
    return Ollama(Settings(ollama_url="http://ollama.test", text_model="text"))


def test_chat_json_returns_the_value_and_logs_the_call(ollama: Ollama) -> None:
    lines = [{"message": {"content": '{"verdict": '}}, {"message": {"content": '"SUPPORTED"}'}}]
    with mock.patch("urllib.request.urlopen", return_value=Stream(lines)):
        assert ollama.chat_json("verify", "sys", "user", {}) == {"verdict": "SUPPORTED"}
    assert ollama.calls[0]["task"] == "verify"
    assert ollama.calls[0]["model"] == "text"


def test_models_without_thinking_are_retried_without_the_flag(ollama: Ollama) -> None:
    sent: list[dict[str, Any]] = []

    def respond(request: Any, timeout: float) -> Stream:
        sent.append(json.loads(request.data))
        if "think" in sent[-1]:
            return Stream([{"error": "model does not support think"}])
        return Stream([{"message": {"content": "{}"}, "done": True}])

    with mock.patch("urllib.request.urlopen", side_effect=respond):
        assert ollama.chat_json("plan", "s", "u", {}) == {}
    assert "think" in sent[0]
    assert "think" not in sent[1]


def test_a_non_json_reply_is_an_error(ollama: Ollama) -> None:
    with (
        mock.patch(
            "urllib.request.urlopen",
            return_value=Stream([{"message": {"content": "no"}, "done": True}]),
        ),
        pytest.raises(LLMError, match="non-JSON"),
    ):
        ollama.chat_json("plan", "s", "u", {})


def test_runaway_whitespace_is_retried_once_with_another_seed(ollama: Ollama) -> None:
    seeds: list[int] = []

    def respond(request: Any, timeout: float) -> Stream:
        seeds.append(json.loads(request.data)["options"]["seed"])
        if len(seeds) == 1:
            return Stream([{"message": {"content": "{" + " " * 250}}])
        return Stream([{"message": {"content": '{"ok": true}'}, "done": True}])

    with mock.patch("urllib.request.urlopen", side_effect=respond):
        assert ollama.chat_json("plan", "s", "u", {}) == {"ok": True}
    assert seeds == [7, 8]


def test_http_errors_and_outages_become_llm_errors(ollama: Ollama) -> None:
    error = urllib.error.HTTPError("u", 500, "boom", {}, io.BytesIO(b"trace"))  # type: ignore[arg-type]
    outage = urllib.error.URLError("refused")
    with (
        mock.patch("urllib.request.urlopen", side_effect=error),
        pytest.raises(LLMError, match="HTTP 500"),
    ):
        ollama.chat_text("verify", "minicheck", "claim")
    with (
        mock.patch("urllib.request.urlopen", side_effect=outage),
        pytest.raises(LLMError, match="unreachable"),
    ):
        ollama.chat_text("verify", "minicheck", "claim")


@pytest.mark.parametrize(
    ("installed", "asked", "resolved"),
    [
        (["qwen3:4b"], "qwen3:4b", "qwen3:4b"),
        (["bge-m3:latest"], "bge-m3", "bge-m3:latest"),
        (
            ["qwen3:4b-instruct-2507-q4_K_M"],
            "qwen3:4b-instruct-2507",
            "qwen3:4b-instruct-2507-q4_K_M",
        ),
    ],
)
def test_resolve_maps_aliases_to_one_installed_tag(
    ollama: Ollama, installed: list[str], asked: str, resolved: str
) -> None:
    tags = {"models": [{"name": n, "digest": f"sha-{n}"} for n in installed]}
    with mock.patch("urllib.request.urlopen", return_value=body(tags)):
        assert ollama.resolve(asked) == (resolved, f"sha-{resolved}")


def test_resolve_refuses_to_guess(ollama: Ollama) -> None:
    tags = {"models": [{"name": "m:a-q4"}, {"name": "m:a-q8"}]}
    with mock.patch("urllib.request.urlopen", side_effect=lambda *_, **__: body(tags)):
        with pytest.raises(LLMError, match="candidates"):
            ollama.resolve("m:a")
        with pytest.raises(LLMError, match="not installed"):
            ollama.resolve("absent")


def test_classifier_replies_are_short_text(ollama: Ollama) -> None:
    with mock.patch(
        "urllib.request.urlopen", return_value=body({"message": {"content": " Yes \n"}})
    ):
        assert ollama.chat_text("verify", "minicheck", "doc") == "Yes"
