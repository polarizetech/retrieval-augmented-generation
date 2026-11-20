"""An Ollama behind a key: the pipeline, the hub's duplicate check and its health probe send it."""

from __future__ import annotations

import asyncio
import io
import json
import threading
import urllib.request
from typing import Any

import pytest

from research_mcp.hub import overlap
from research_pipeline.config import Settings
from research_pipeline.llm import Ollama


def test_no_key_means_no_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIPELINE_OLLAMA_KEY", raising=False)
    monkeypatch.delenv("PIPELINE_OLLAMA_KEY_FILE", raising=False)
    assert Settings().ollama_headers == {}


def test_key_from_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.delenv("PIPELINE_OLLAMA_KEY", raising=False)
    key = tmp_path / "models.key"
    key.write_text("abc\n")
    monkeypatch.setenv("PIPELINE_OLLAMA_KEY_FILE", str(key))
    settings = Settings()
    assert settings.ollama_headers == {"Authorization": "Bearer abc"}
    assert "abc" not in repr(settings)


def test_pipeline_sends_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_OLLAMA_KEY", "abc")
    seen: list[urllib.request.Request] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float = 0) -> io.BytesIO:
        seen.append(req)
        return io.BytesIO(json.dumps({"models": []}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = Ollama(Settings())
    client._post("/api/embed", {"model": "bge-m3", "input": ["x"]}, timeout=1)
    client._post_get("/api/tags")
    assert [r.get_header("Authorization") for r in seen] == ["Bearer abc", "Bearer abc"]


def test_hub_post_sends_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[urllib.request.Request] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float = 0) -> io.BytesIO:
        seen.append(req)
        return io.BytesIO(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    overlap.http_post("http://x/api/embed", {}, headers={"Authorization": "Bearer abc"})
    assert seen[0].get_header("Authorization") == "Bearer abc"


def test_gateway_wakes_hosted_models_once_a_minute(monkeypatch):
    from research_mcp import gateway

    calls = []
    sent = threading.Event()

    def opener(request, timeout):
        calls.append((request.full_url, request.get_header("Authorization"), timeout))
        sent.set()
        return io.BytesIO(b"")

    hosted = Settings(ollama_url="https://models.example", ollama_key="k")
    monkeypatch.setattr(gateway, "_last_wake", [float("-inf")])
    now = [1000.0]
    assert gateway.wake_hosted_models(hosted, opener, lambda: now[0])
    assert sent.wait(5)
    assert calls == [("https://models.example/api/version", "Bearer k", 300)]
    now[0] += 30
    assert not gateway.wake_hosted_models(hosted, opener, lambda: now[0])
    now[0] += 31
    assert gateway.wake_hosted_models(hosted, opener, lambda: now[0])
    local = Settings(ollama_url="http://127.0.0.1:11434", ollama_key="")
    monkeypatch.setattr(gateway, "_last_wake", [float("-inf")])
    assert not gateway.wake_hosted_models(local, opener, lambda: now[0])


def test_ask_waits_for_a_wake_up_in_flight(monkeypatch):
    from research_mcp import gateway

    release = threading.Event()

    def opener(request, timeout):
        release.wait(5)
        return io.BytesIO(b"")

    hosted = Settings(ollama_url="https://models.example", ollama_key="k")
    monkeypatch.setattr(gateway, "_last_wake", [float("-inf")])
    idle = threading.Event()
    idle.set()
    monkeypatch.setattr(gateway, "_wake_done", [idle])
    asyncio.run(asyncio.wait_for(gateway.models_ready(), 5))  # nothing in flight: no wait
    assert gateway.wake_hosted_models(hosted, opener)
    assert not gateway._wake_done[0].is_set()
    release.set()
    asyncio.run(asyncio.wait_for(gateway.models_ready(), 5))
    assert gateway._wake_done[0].is_set()


def test_only_asks_that_use_the_models_wake_them() -> None:
    from pathlib import Path

    from research_mcp import gateway
    from research_mcp.routing import classify, load_rules

    rules = load_rules(Path(__file__).parent.parent / "config" / "routing.default.toml")
    assert gateway.needs_models(classify("What is pink noise?", rules).mode)
    assert not gateway.needs_models(classify("code: rename the helper in ask.py", rules).mode)
    assert gateway.needs_models("fetch")
