"""Client for the paper library, over MCP.

The library lives in its own repository and owns discovery, open-access rules, full-text storage
and provenance. This module does not reimplement any of that: it spawns the same stdio server the
gateway federates (the `papers` upstream in config/mcp-gateway.json) and calls its tools.
"""

from __future__ import annotations

import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent


class PapersError(RuntimeError):
    def __init__(self, message: str, code: str = "tool_error"):
        super().__init__(message)
        self.code = code  # the library's own code, e.g. not_found, unavailable, no_store


class PaperLibrary:
    def __init__(self, gateway_config: Path, upstream: str = "papers"):
        if not gateway_config.exists():
            raise PapersError(f"missing {gateway_config}; copy config/mcp-gateway.example.json")
        spec = json.loads(gateway_config.read_text()).get("upstreams", {}).get(upstream)
        if not spec:
            raise PapersError(f"no '{upstream}' upstream in {gateway_config}")
        self.spec = spec
        self.stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def __aenter__(self) -> PaperLibrary:
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

    async def __aexit__(self, *exc) -> None:
        await self.stack.aclose()

    async def _call(self, tool: str, **arguments: Any) -> dict[str, Any]:
        """Return the tool's envelope. Raises PapersError on a refused or failed call."""
        assert self.session is not None
        result = await self.session.call_tool(tool, arguments)
        text = next((c.text for c in result.content if isinstance(c, TextContent)), "")
        try:
            envelope = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PapersError(f"{tool}: unparseable result: {text[:200]!r}") from exc
        if result.isError or not envelope.get("ok", False):
            raise PapersError(
                f"{tool}: {str(envelope.get('error') or envelope)[:300]}",
                code=str(envelope.get("code") or "tool_error"),
            )
        return envelope

    async def search(
        self, query: str, limit: int = 10, include_closed: bool = True
    ) -> dict[str, Any]:
        """Merged hits plus per-provider status. A provider that did not answer is NOT zero hits."""
        return (
            await self._call("search", query=query, limit=limit, include_closed=include_closed)
        )["data"]

    async def fetch(self, identifier: str) -> dict[str, Any]:
        return (await self._call("fetch", identifier=identifier))["data"]

    async def held(self, limit: int = 100000) -> list[dict[str, Any]]:
        return (await self._call("library", limit=limit))["data"]["works"]

    async def full_text(self, identifier: str) -> str:
        parts: list[str] = []
        offset = 0
        while True:
            data = (
                await self._call("text", identifier=identifier, offset=offset, max_chars=100000)
            )["data"]
            parts.append(data["text"])
            offset = data["end"]
            if offset >= data["total_chars"] or not data["text"]:
                return "".join(parts)

    async def provenance(self, identifier: str) -> dict[str, Any]:
        return (await self._call("provenance", identifier=identifier))["data"]

    async def citations(
        self, identifier: str, direction: str = "citations", limit: int = 25
    ) -> dict[str, Any]:
        return (
            await self._call("citations", identifier=identifier, direction=direction, limit=limit)
        )["data"]
