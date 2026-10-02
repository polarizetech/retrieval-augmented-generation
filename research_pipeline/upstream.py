"""A client for one MCP server named in the gateway config, spoken to over stdio.

The pipeline reaches other tools this way: the paper library (`upstreams.papers`) and companion
tools (`companions.<key>`, see companions.py). Each is a separate program with its own repository;
this module only spawns it and calls its tools. Every such server answers with one envelope:

    {"ok": true, "data": ...}
    {"ok": false, "code": "not_found" | "unavailable" | "tool_error", "error": "..."}
"""

from __future__ import annotations

import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Self

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent


class UpstreamError(RuntimeError):
    def __init__(self, message: str, code: str = "tool_error"):
        super().__init__(message)
        self.code = code  # the server's own code, e.g. not_found, unavailable


def spec_for(gateway_config: Path, name: str, section: str = "upstreams") -> dict[str, Any] | None:
    """The server entry `section.name` of the gateway config, or None if absent or disabled."""
    if not gateway_config.exists():
        return None
    spec = json.loads(gateway_config.read_text()).get(section, {}).get(name)
    return spec if spec and spec.get("enabled", True) else None


class Upstream:
    error: type[UpstreamError] = UpstreamError

    def __init__(self, gateway_config: Path, name: str, section: str = "upstreams"):
        if not gateway_config.exists():
            raise self.error(f"missing {gateway_config}; copy config/mcp-gateway.example.json")
        spec = spec_for(gateway_config, name, section)
        if not spec:
            raise self.error(f"no '{name}' {section.rstrip('s')} in {gateway_config}")
        self.name = name
        self.spec = spec
        self.stack = AsyncExitStack()
        self.session: ClientSession | None = None

    @classmethod
    def over(cls, session: ClientSession) -> Self:
        """Speak the contract over a session someone else already holds (the gateway)."""
        client = cls.__new__(cls)
        client.name, client.spec = "", {}
        client.stack, client.session = AsyncExitStack(), session
        return client

    async def __aenter__(self) -> Self:
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in self.spec.get("env", {}).items()})
        params = StdioServerParameters(
            command=self.spec["command"],
            args=[str(v) for v in self.spec.get("args", [])],
            cwd=self.spec.get("cwd"),
            env=env,
        )
        read, write = await self.stack.enter_async_context(stdio_client(params))
        self.session = await self.stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stack.aclose()

    async def _call(self, tool: str, **arguments: Any) -> dict[str, Any]:
        """Return the tool's envelope. Raises the client's error on a refused or failed call."""
        assert self.session is not None
        result = await self.session.call_tool(tool, arguments)
        text = next((c.text for c in result.content if isinstance(c, TextContent)), "")
        try:
            envelope = json.loads(text)
        except json.JSONDecodeError as exc:
            raise self.error(f"{tool}: unparseable result: {text[:200]!r}") from exc
        if result.isError or not envelope.get("ok", False):
            raise self.error(
                f"{tool}: {str(envelope.get('error') or envelope)[:300]}",
                code=str(envelope.get("code") or "tool_error"),
            )
        return envelope

    async def call(self, tool: str, **arguments: Any) -> Any:
        """The `data` of a successful call."""
        return (await self._call(tool, **arguments))["data"]
