"""Expose patent-fetch's complete search/fetch/read lifecycle as MCP tools.

patent-fetch intentionally remains the owner of provider credentials, storage, family folding and
provenance.  This server is only a typed stdio adapter so the shared gateway can offer those
operations to every client.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

COMMAND = (
    os.environ.get("PATENT_FETCH_COMMAND")
    or shutil.which("patent-fetch")
    or str(Path.home() / ".local/bin/patent-fetch")
)
READ = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
SEARCH = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True)
FETCH = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=True
)

mcp = FastMCP(
    "patent-fetch",
    instructions=(
        "Search for patents, fetch selected publications into the operator's private patent "
        "library, then read their description or claims. Search results are prior-art pointers, "
        "not legal opinions. No result never proves that no patent exists."
    ),
)


def _run(*args: str, json_output: bool = True) -> Any:
    argv = [COMMAND]
    if json_output:
        argv.append("--json")
    argv.extend(args)
    result = subprocess.run(  # noqa: S603 - fixed executable and separated arguments
        argv, capture_output=True, text=True, timeout=180, check=False
    )
    if result.returncode:
        raise ValueError((result.stderr or result.stdout or "patent-fetch failed")[:1000])
    if not json_output:
        return result.stdout
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"patent-fetch returned invalid JSON: {result.stdout[:300]}") from exc


@mcp.tool(annotations=SEARCH)
def search(query: str, limit: int = 10, since: int | None = None) -> dict[str, Any]:
    """Search configured patent providers by title/abstract/full-text provider capability.

    Each provider's status is returned explicitly. No hits is not evidence of no patent.
    """
    args = ["search", query, "-n", str(max(1, min(limit, 50)))]
    if since is not None:
        args += ["--since", str(since)]
    return _run(*args)


@mcp.tool(annotations=FETCH)
def fetch(identifiers: list[str], collection: str = "") -> list[dict[str, Any]]:
    """Fetch up to ten patents, including description and claims when available."""
    if not identifiers or len(identifiers) > 10:
        raise ValueError("identifiers must contain between one and ten patent publication numbers")
    args = ["fetch", *identifiers]
    if collection:
        args += ["--collection", collection]
    return _run(*args)


@mcp.tool(annotations=READ)
def record(identifier: str) -> dict[str, Any]:
    """Return a held patent's normalized record and retrieval provenance."""
    return _run("record", identifier)


@mcp.tool(annotations=READ)
def text(identifier: str, max_chars: int = 40_000) -> dict[str, Any]:
    """Read a held patent's abstract, description and claims; fetch it first if it is not held."""
    body = str(_run("text", identifier, json_output=False))
    limit = max(1000, min(max_chars, 100_000))
    return {"identifier": identifier, "text": body[:limit], "truncated": len(body) > limit}


@mcp.tool(annotations=READ)
def claims(identifier: str, max_chars: int = 40_000) -> dict[str, Any]:
    """Read only the claims of a held patent. This is source text, not a legal interpretation."""
    body = str(_run("claims", identifier, json_output=False))
    limit = max(1000, min(max_chars, 100_000))
    return {"identifier": identifier, "claims": body[:limit], "truncated": len(body) > limit}


@mcp.tool(annotations=READ)
def library(query: str = "", full_text: bool = False) -> list[dict[str, Any]]:
    """Search only patents already held in the private local library; performs no network access."""
    args = ["library", query]
    if full_text:
        args.append("--full-text")
    return _run(*args)


def main() -> int:
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
