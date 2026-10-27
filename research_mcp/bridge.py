"""A stdio front for the shared HTTP gateway, so local clients don't each start their own.

Every stdio gateway spawns its own paper library (passage vectors and a reranker, about 1.7 GB),
so ten open Claude sessions hold ten copies. Pointing their config at this bridge instead makes
each session a small proxy to the one always-on HTTP gateway:

    research-rag-gateway --connect http://127.0.0.1:8891/

Tool names, descriptions, schemas and the server's instructions are passed through unchanged, so
clients see exactly what a stdio gateway would show. The bridge authenticates with the local token
(config/.mcp-token), which the HTTP gateway accepts only from this machine.

The gateway serves stateless JSON responses, so each request is one plain POST. That keeps the
bridge alive across a gateway restart: a call made while it is down fails on its own, and the
next one works.
"""

from __future__ import annotations

import asyncio
import itertools
import sys
from typing import Any

import httpx
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import LATEST_PROTOCOL_VERSION, CallToolResult, TextContent, Tool

# A research run or a large fetch can hold a call open for minutes; the client's own MCP
# timeout is what bounds a call, not the bridge.
CALL_SECONDS = 1800.0
STARTUP_ATTEMPTS = 15
RESTART_HINT = "launchctl kickstart -k gui/$(id -u)/com.polarizetech.research-rag"


def _log(message: str) -> None:
    print(f"research-rag bridge: {message}", file=sys.stderr, flush=True)


class GatewayError(RuntimeError):
    pass


class Upstream:
    def __init__(self, url: str, token: str, http: httpx.AsyncClient | None = None):
        self.url = url
        self.http = http or httpx.AsyncClient(timeout=httpx.Timeout(CALL_SECONDS, connect=10.0))
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": LATEST_PROTOCOL_VERSION,
        }
        self.ids = itertools.count(1)

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        body = {"jsonrpc": "2.0", "id": next(self.ids), "method": method, "params": params or {}}
        try:
            reply = await self.http.post(self.url, json=body, headers=self.headers)
        except httpx.HTTPError as exc:
            raise GatewayError(
                f"the shared gateway at {self.url} is not reachable ({exc})"
            ) from exc
        if reply.status_code != 200:
            raise GatewayError(
                f"the shared gateway at {self.url} answered HTTP {reply.status_code}: "
                f"{reply.text[:200]}"
            )
        data = reply.json()
        if "error" in data:
            raise GatewayError(f"{method}: {data['error'].get('message', data['error'])}")
        return data["result"]

    async def initialize(self) -> dict[str, Any]:
        params = {
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "research-rag-bridge", "version": "1"},
        }
        for attempt in range(STARTUP_ATTEMPTS):
            try:
                return await self.request("initialize", params)
            except GatewayError as exc:
                if attempt == STARTUP_ATTEMPTS - 1:
                    raise GatewayError(f"{exc}. Is it running? {RESTART_HINT}") from exc
                await asyncio.sleep(2)
        raise AssertionError("unreachable")

    async def list_tools(self) -> list[Tool]:
        return [Tool.model_validate(t) for t in (await self.request("tools/list"))["tools"]]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        try:
            result = await self.request("tools/call", {"name": name, "arguments": arguments})
        except GatewayError as exc:
            return CallToolResult(isError=True, content=[TextContent(type="text", text=str(exc))])
        return CallToolResult.model_validate(result)


def build(upstream: Upstream, info: dict[str, Any]) -> Server[Any, Any]:
    server: Server[Any, Any] = Server(
        str(info.get("serverInfo", {}).get("name") or "research-rag"),
        instructions=info.get("instructions"),
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return await upstream.list_tools()

    # The gateway validates arguments itself; validating here too would only add a second,
    # possibly stale, copy of every schema.
    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
        return await upstream.call_tool(name, arguments)

    return server


async def serve(url: str, token: str) -> None:
    upstream = Upstream(url, token)
    server = build(upstream, await upstream.initialize())
    _log(f"connected to {url}")
    async with stdio_server() as (stdin, stdout):
        await server.run(stdin, stdout, server.create_initialization_options())


def run(url: str, token: str) -> int:
    try:
        asyncio.run(serve(url, token))
    except GatewayError as exc:
        _log(str(exc))
        return 1
    return 0
