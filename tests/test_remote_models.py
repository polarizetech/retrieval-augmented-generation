"""An Ollama behind a key: the pipeline, the hub's duplicate check and its health probe send it."""

from __future__ import annotations

import io
import json
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
