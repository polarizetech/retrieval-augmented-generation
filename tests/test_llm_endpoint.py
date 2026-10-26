"""The OpenAI-compatible endpoint backend, with HTTP replaced by canned responses."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any
from unittest import mock

import pytest

from research_pipeline.config import Settings
from research_pipeline.llm import Endpoint, LLMError, ModelOutputError, make_llm, parse_json

SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"enum": ["SUPPORTED", "NOT_SUPPORTED"]}},
    "required": ["verdict"],
}


def settings(**kw: Any) -> Settings:
    base: dict[str, Any] = {
        "endpoint_url": "https://openrouter.ai/api/v1",
        "endpoint_model": "deepseek/deepseek-v4-pro",
        "endpoint_key": "sk-test",
        "text_model": "local-text",
    }
    return Settings(**(base | kw))


def reply(content: str, finish: str = "stop", **extra: Any) -> mock.MagicMock:
    payload = {
        "model": "deepseek/deepseek-v4-pro-0813",
        "provider": "Fireworks",
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "cost": 0.0001,
        },
    } | extra
    response = mock.MagicMock()
    response.__enter__.return_value = io.BytesIO(json.dumps(payload).encode())
    return response


def http_error(code: int, text: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(text.encode()))  # type: ignore[arg-type]


class Recorder:
    def __init__(self, *responses: Any):
        self.responses = list(responses)
        self.sent: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []

    def __call__(self, request: Any, timeout: float) -> Any:
        if not isinstance(request, str):  # a plain URL is a GET, e.g. a price catalogue
            self.sent.append(json.loads(request.data))
            self.headers.append(dict(request.header_items()))
        out = self.responses.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_request_asks_for_schema_zero_retention_and_no_reasoning() -> None:
    rec = Recorder(reply('{"verdict": "SUPPORTED"}'))
    ep = Endpoint(settings())
    with mock.patch("urllib.request.urlopen", side_effect=rec):
        assert ep.chat_json("verify", "sys", "user", SCHEMA) == {"verdict": "SUPPORTED"}
    body = rec.sent[0]
    assert body["model"] == "deepseek/deepseek-v4-pro"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["reasoning"] == {"effort": "none"}
    assert body["provider"] == {"zdr": True, "sort": "price"}
    assert rec.headers[0]["Authorization"] == "Bearer sk-test"
    call = ep.calls[0]
    assert call["served_model"] == "deepseek/deepseek-v4-pro-0813"
    assert call["provider"] == "Fireworks"
    assert (call["input_tokens"], call["output_tokens"], call["cost"]) == (100, 10, 0.0001)


def test_other_hosts_get_no_openrouter_fields() -> None:
    rec = Recorder(reply('{"verdict": "SUPPORTED"}'))
    ep = Endpoint(settings(endpoint_url="http://gpu.test:8000/v1", endpoint_reasoning="default"))
    with mock.patch("urllib.request.urlopen", side_effect=rec):
        ep.chat_json("verify", "sys", "user", SCHEMA)
    assert "provider" not in rec.sent[0]
    assert "reasoning" not in rec.sent[0]


def test_a_model_that_must_reason_is_retried_without_reasoning_off() -> None:
    rec = Recorder(
        http_error(400, '{"error":{"message":"reasoning effort none is not supported"}}'),
        reply('{"verdict": "SUPPORTED"}'),
        reply('{"verdict": "SUPPORTED"}'),
    )
    ep = Endpoint(settings())
    with mock.patch("urllib.request.urlopen", side_effect=rec):
        ep.chat_json("verify", "s", "u", SCHEMA)
        ep.chat_json("verify", "s", "u", SCHEMA)
    assert "reasoning" in rec.sent[0]
    assert "reasoning" not in rec.sent[1]
    assert "reasoning" not in rec.sent[2]
    assert ep.notes
    assert "reasoning" in ep.notes[0]


def test_no_json_schema_support_falls_back_to_json_object() -> None:
    rec = Recorder(
        http_error(400, "response_format json_schema is not supported"),
        reply('```json\n{"verdict": "NOT_SUPPORTED"}\n```'),
    )
    ep = Endpoint(settings())
    with mock.patch("urllib.request.urlopen", side_effect=rec):
        assert ep.chat_json("verify", "s", "u", SCHEMA) == {"verdict": "NOT_SUPPORTED"}
    assert rec.sent[1]["response_format"] == {"type": "json_object"}


def test_an_unrelated_400_is_an_error_not_a_relaxation() -> None:
    rec = Recorder(http_error(400, "model not found"))
    with mock.patch("urllib.request.urlopen", side_effect=rec), pytest.raises(LLMError):
        Endpoint(settings()).chat_json("verify", "s", "u", SCHEMA)


def test_an_answer_that_breaks_the_schema_gets_one_corrected_retry() -> None:
    rec = Recorder(reply('{"verdict": "MAYBE"}'), reply('{"verdict": "SUPPORTED"}'))
    with mock.patch("urllib.request.urlopen", side_effect=rec):
        assert Endpoint(settings()).chat_json("verify", "s", "u", SCHEMA) == {
            "verdict": "SUPPORTED"
        }
    retry = rec.sent[1]["messages"]
    assert retry[-2]["role"] == "assistant"
    assert "rejected" in retry[-1]["content"]


def test_two_bad_answers_are_a_model_output_error() -> None:
    rec = Recorder(reply("not json"), reply("still not"))
    with mock.patch("urllib.request.urlopen", side_effect=rec), pytest.raises(ModelOutputError):
        Endpoint(settings()).chat_json("verify", "s", "u", SCHEMA)


def test_cut_off_answers_are_a_model_output_error() -> None:
    rec = Recorder(reply('{"verdict": "SUP', finish="length"))
    with mock.patch("urllib.request.urlopen", side_effect=rec), pytest.raises(ModelOutputError):
        Endpoint(settings()).chat_json("verify", "s", "u", SCHEMA)


def test_rate_limits_are_retried() -> None:
    rec = Recorder(http_error(429, "slow down"), reply('{"verdict": "SUPPORTED"}'))
    with (
        mock.patch("urllib.request.urlopen", side_effect=rec),
        mock.patch("time.sleep") as sleep,
    ):
        Endpoint(settings()).chat_json("verify", "s", "u", SCHEMA)
    assert sleep.called
    assert len(rec.sent) == 2


def test_errors_inside_a_200_are_raised() -> None:
    rec = Recorder(reply("", error={"code": 403, "message": "no ZDR provider"}))
    with (
        mock.patch("urllib.request.urlopen", side_effect=rec),
        pytest.raises(LLMError, match="ZDR"),
    ):
        Endpoint(settings()).chat_json("verify", "s", "u", SCHEMA)


def test_other_models_go_to_the_local_ollama() -> None:
    local = mock.MagicMock()
    local.chat_json.return_value = {"verdict": "SUPPORTED"}
    local.chat_text.return_value = "Yes"
    local.resolve.return_value = ("bespoke-minicheck:latest", "sha")
    ep = Endpoint(settings(), local=local)
    assert ep.chat_json("verify", "s", "u", SCHEMA, model="qwen3:4b") == {"verdict": "SUPPORTED"}
    assert ep.chat_text("verify", "bespoke-minicheck:latest", "u") == "Yes"
    assert ep.resolve("bespoke-minicheck") == ("bespoke-minicheck:latest", "sha")
    assert ep.resolve("deepseek/deepseek-v4-pro") == ("deepseek/deepseek-v4-pro", None)


def test_the_key_is_not_in_the_settings_repr() -> None:
    assert "sk-test" not in repr(settings())


def test_make_llm_needs_url_and_model() -> None:
    with pytest.raises(LLMError, match="PIPELINE_ENDPOINT_URL"):
        make_llm(Settings(endpoint_url="", endpoint_model=""), "endpoint")
    assert isinstance(make_llm(settings(), "endpoint"), Endpoint)
    with pytest.raises(LLMError, match="unknown"):
        make_llm(settings(), "nope")


def test_parse_json_finds_the_object_and_checks_it() -> None:
    assert parse_json('Sure: {"verdict": "SUPPORTED"} done', SCHEMA)[0] == {"verdict": "SUPPORTED"}
    assert parse_json("[]", SCHEMA)[0] is None
    value, problem = parse_json('{"verdict": 3}', SCHEMA)
    assert value is None
    assert "schema" in problem


def test_local_verifier_calls_land_in_the_same_call_log() -> None:
    ep = Endpoint(settings())
    ep.local.calls.append({"task": "verify", "model": "bespoke-minicheck:latest"})
    assert ep.calls == [{"task": "verify", "model": "bespoke-minicheck:latest"}]


def test_provider_sort_can_be_turned_off_without_losing_zero_retention() -> None:
    rec = Recorder(reply('{"verdict": "SUPPORTED"}'))
    with mock.patch("urllib.request.urlopen", side_effect=rec):
        Endpoint(settings(endpoint_sort="")).chat_json("verify", "s", "u", SCHEMA)
    assert rec.sent[0]["provider"] == {"zdr": True}


NEARAI_CATALOGUE = [
    {
        "modelId": "Qwen/Qwen3.8-27B",
        "inputCostPerToken": {"amount": 40, "scale": 9, "currency": "USD"},
        "outputCostPerToken": {"amount": 1350, "scale": 9, "currency": "USD"},
        "cacheReadCostPerToken": {"amount": 18, "scale": 9, "currency": "USD"},
    }
]


def nearai_settings(**kw: Any) -> Settings:
    base: dict[str, Any] = {
        "endpoint_url": "https://cloud-api.near.ai/v1",
        "endpoint_model": "Qwen/Qwen3.8-27B",
        "endpoint_key": "sk-test",
    }
    return settings(**(base | kw))


def nearai_reply(content: str) -> mock.MagicMock:
    payload = {
        "model": "Qwen/Qwen3.8-27B",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_tokens_details": None},
    }
    response = mock.MagicMock()
    response.__enter__.return_value = io.BytesIO(json.dumps(payload).encode())
    return response


def catalogue() -> mock.MagicMock:
    response = mock.MagicMock()
    response.__enter__.return_value = io.BytesIO(json.dumps(NEARAI_CATALOGUE).encode())
    return response


def test_nearai_calls_are_priced_from_its_catalogue_and_named() -> None:
    rec = Recorder(nearai_reply('{"verdict": "SUPPORTED"}'), catalogue())
    ep = Endpoint(nearai_settings())
    with mock.patch("urllib.request.urlopen", side_effect=rec):
        ep.chat_json("verify", "s", "u", SCHEMA)
    body = rec.sent[0]
    assert "provider" not in body  # OpenRouter-only fields stay off
    assert body["reasoning"] == {"effort": "none"}  # Near AI honours it
    call = ep.calls[0]
    assert call["provider"] == "Near AI"
    assert call["cost"] == pytest.approx(1000 * 40e-9 + 100 * 1350e-9)


def test_a_configured_price_is_used_for_any_host() -> None:
    rec = Recorder(nearai_reply('{"verdict": "SUPPORTED"}'))
    ep = Endpoint(settings(endpoint_url="http://gpu.test/v1", endpoint_price="0.5,2"))
    with mock.patch("urllib.request.urlopen", side_effect=rec):
        ep.chat_json("verify", "s", "u", SCHEMA)
    assert ep.calls[0]["provider"] == "gpu.test"
    assert ep.calls[0]["cost"] == pytest.approx(1000 * 0.5e-6 + 100 * 2e-6)


def test_an_unknown_price_is_none_not_zero() -> None:
    rec = Recorder(nearai_reply('{"verdict": "SUPPORTED"}'))
    ep = Endpoint(settings(endpoint_url="http://gpu.test/v1"))
    with mock.patch("urllib.request.urlopen", side_effect=rec):
        ep.chat_json("verify", "s", "u", SCHEMA)
    assert ep.calls[0]["cost"] is None


def test_running_out_of_credits_says_so() -> None:
    rec = Recorder(http_error(402, '{"error":"insufficient balance"}'))
    with (
        mock.patch("urllib.request.urlopen", side_effect=rec),
        pytest.raises(LLMError, match="out of credits at Near AI"),
    ):
        Endpoint(nearai_settings()).chat_json("verify", "s", "u", SCHEMA)


def test_dropped_connections_are_retried_until_the_limit() -> None:
    import ssl

    drop = urllib.error.URLError(ssl.SSLError("UNEXPECTED_EOF_WHILE_READING"))
    rec = Recorder(drop, ConnectionResetError(), reply('{"verdict": "SUPPORTED"}'))
    with mock.patch("urllib.request.urlopen", side_effect=rec), mock.patch("time.sleep"):
        assert Endpoint(settings()).chat_json("verify", "s", "u", SCHEMA) == {
            "verdict": "SUPPORTED"
        }
    rec = Recorder(*([drop] * 3))
    with (
        mock.patch("urllib.request.urlopen", side_effect=rec),
        mock.patch("time.sleep"),
        pytest.raises(LLMError, match="after 3 tries"),
    ):
        Endpoint(settings(endpoint_retries=3)).chat_json("verify", "s", "u", SCHEMA)


def test_auto_means_the_configured_endpoint_when_there_is_one() -> None:
    from research_pipeline.llm import resolve_backend

    assert resolve_backend(settings(), "auto", otherwise="client") == "endpoint"
    assert resolve_backend(settings(), "client", otherwise="ollama") == "client"
    assert isinstance(make_llm(settings(llm_backend="auto")), Endpoint)
    with pytest.raises(LLMError, match="needs an MCP client"):
        make_llm(settings(), "client")
