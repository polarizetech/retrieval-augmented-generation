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


OPEN_TEXT = (
    "Heart rate variability was recorded in 30 adults during paced breathing. " * 12
    + "\n\n"
    + "Paced breathing at six breaths per minute raised heart rate variability by 40 percent. " * 6
)
RECORDS = {
    "10.1000/open": {
        "work": "W-open",
        "doi": "10.1000/open",
        "title": "Paced breathing and HRV",
        "full_text": True,
        "is_retracted": False,
    },
    "10.1000/closed": {
        "work": "W-closed",
        "doi": "10.1000/closed",
        "title": "A closed paper",
        "full_text": False,
        "is_retracted": False,
    },
}


@mcp.tool()
def fetch(identifier: str) -> dict[str, Any]:
    """Two known papers, one open and one closed; refuse anything else as not found."""
    if identifier in RECORDS:
        return {"ok": True, "data": RECORDS[identifier]}
    return {"ok": False, "code": "not_found", "error": f"no open copy of {identifier}"}


@mcp.tool()
def text(identifier: str, offset: int = 0, max_chars: int = 100000) -> dict[str, Any]:
    """Page through the open paper's text, as the library does."""
    if identifier != "W-open":
        return {"ok": False, "code": "not_found", "error": f"{identifier} has no text"}
    chunk = OPEN_TEXT[offset : offset + max_chars]
    return {
        "ok": True,
        "data": {
            "text": chunk,
            "offset": offset,
            "end": offset + len(chunk),
            "total_chars": len(OPEN_TEXT),
        },
    }


if __name__ == "__main__":
    mcp.run()
