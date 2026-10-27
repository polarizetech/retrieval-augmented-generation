"""The stdio bridge: local clients reach the one shared HTTP gateway with the local token."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn

from research_mcp import bridge, gateway

KEY = "correct horse battery staple access key"
LOCAL = "a-local-bridge-token-that-is-long-enough-123"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def shared_gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """The public-mode HTTP gateway, served for real on a loopback port."""
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MCP_PUBLIC_URL", "https://mcp.example.org/research")
    monkeypatch.setenv("MCP_RESOURCE_PATH", "")
    monkeypatch.setenv("MCP_OAUTH_ACCESS_KEY", KEY)
    monkeypatch.setenv("MCP_OAUTH_STATE", str(tmp_path / "oauth.sqlite"))
    monkeypatch.setenv("MCP_TOKEN", LOCAL)
    config = tmp_path / "gateway.json"
    config.write_text(json.dumps({"upstreams": {}}))
    port = _free_port()
    server, _ = gateway.build_server(config, "127.0.0.1", port, tmp_path / "token", http=True)
    app = gateway.LocalOnly(server.streamable_http_app(), LOCAL)
    running = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    for _ in range(100):
        if running.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/"
    running.should_exit = True
    thread.join(timeout=5)


def test_the_bridge_passes_tools_and_instructions_through(shared_gateway: str) -> None:
    async def go() -> None:
        upstream = bridge.Upstream(shared_gateway, LOCAL)
        info = await upstream.initialize()
        assert "Evidence-first" in info["instructions"]
        names = {t.name for t in await upstream.list_tools()}
        assert {"gateway_status", "rag__retrieve_evidence"} <= names
        status = await upstream.call_tool("gateway_status", {})
        assert not status.isError

    asyncio.run(go())


def test_a_wrong_token_is_reported_not_crashed(shared_gateway: str) -> None:
    async def go() -> None:
        upstream = bridge.Upstream(shared_gateway, LOCAL + "x")
        result = await upstream.call_tool("gateway_status", {})
        assert result.isError
        assert "HTTP 401" in result.content[0].text  # type: ignore[union-attr]

    asyncio.run(go())


def test_a_gateway_that_is_down_fails_one_call_only() -> None:
    async def go() -> None:
        upstream = bridge.Upstream(f"http://127.0.0.1:{_free_port()}/", LOCAL)
        result = await upstream.call_tool("gateway_status", {})
        assert result.isError
        assert "not reachable" in result.content[0].text  # type: ignore[union-attr]

    asyncio.run(go())
