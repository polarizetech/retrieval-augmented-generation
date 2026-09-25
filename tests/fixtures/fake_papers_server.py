"""A stand-in paper library: a stdio MCP server with the library's tool names and envelope."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fake-papers")


@mcp.tool()
def search(query: str, include_closed: bool = False, limit: int = 10) -> dict[str, Any]:
    """Search the fake library."""
    hit = {"title": f"A paper about {query}", "ids": {"doi": "10.1000/fake"}, "work": "W-fake"}
    return {"ok": True, "data": {"hits": [hit][:limit], "providers": {"fake": {"status": "ok"}}}}


@mcp.tool()
def fetch(identifier: str) -> dict[str, Any]:
    """Refuse every fetch, as a library does for a closed-access paper."""
    return {"ok": False, "code": "not_found", "error": f"no open copy of {identifier}"}


if __name__ == "__main__":
    mcp.run()
