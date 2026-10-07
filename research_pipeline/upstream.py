"""Clients for the other programs named in the gateway config.

The pipeline reaches other tools this way: the paper library (`upstreams.papers`) and companion
tools (`companions.<key>`, see companions.py). Each is a separate program with its own repository;
this module only spawns it and calls it.

`Upstream` speaks to an MCP server over stdio. Every such server answers with one envelope:

    {"ok": true, "data": ...}
    {"ok": false, "code": "not_found" | "unavailable" | "tool_error", "error": "..."}

`Command` runs a tool that ships a command line and no MCP server: it is started once per call,
with fixed arguments (never through a shell), and its standard output is read as JSON.
"""

from __future__ import annotations

import asyncio
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


class Command:
    """A companion that is a command line: `await tool.run("search", query, "-n", "5")`.

    The gateway config gives the command and any leading arguments (`"args": ["--json"]`). Each
    call starts it afresh with those plus the call's own arguments, as an argument list: nothing
    a model wrote is ever interpreted by a shell. Standard output must be one JSON value.
    """

    def __init__(self, gateway_config: Path, name: str, section: str = "companions"):
        spec = spec_for(gateway_config, name, section)
        if not spec:
            raise UpstreamError(f"no '{name}' {section.rstrip('s')} in {gateway_config}")
        self.name, self.spec = name, spec

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def run(self, *arguments: str, timeout: float = 120.0) -> Any:
        argv = [self.spec["command"], *(str(v) for v in self.spec.get("args", [])), *arguments]
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in self.spec.get("env", {}).items()})
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.spec.get("cwd"),
                env=env,
            )
        except OSError as exc:
            raise UpstreamError(
                f"{self.name}: cannot start {argv[0]!r}: {exc}", "unavailable"
            ) from exc
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise UpstreamError(
                f"{self.name}: no answer in {timeout:.0f} s", "unavailable"
            ) from exc
        if proc.returncode != 0:
            detail = err.decode(errors="replace").strip().splitlines()[-1:] or ["no message"]
            raise UpstreamError(f"{self.name}: exit {proc.returncode}: {detail[0][:300]}")
        try:
            return json.loads(out)
        except json.JSONDecodeError as exc:
            raise UpstreamError(f"{self.name}: output is not JSON: {out[:200]!r}") from exc
