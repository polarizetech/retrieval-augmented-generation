"""Web search through a SearXNG instance, shared by the gateway's `web__search`, the quick route
of `ask` and the research pipeline.

The instance is SEARXNG_URL, else `web_search.searxng_url` in the gateway config. It runs on this
Mac (see docs/mcp-clients.md); nothing here reaches a commercial search API.

A web result is a lead, never evidence: its title and snippet are what a search engine showed, not
text anyone has read. The pipeline turns result URLs into paper identifiers with the library's
`leads` tool, and only papers the library then fetches and indexes can ground a claim.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

USER_AGENT = "research-rag"
TIMEOUT = 20.0


def searxng_url(config: dict[str, Any] | Path | None = None) -> str:
    """Base URL of the SearXNG instance, or "" when none is configured."""
    direct = os.environ.get("SEARXNG_URL", "").strip()
    if direct:
        return direct.rstrip("/")
    if isinstance(config, Path):
        try:
            config = json.loads(config.read_text())
        except (OSError, ValueError):
            config = None
    if not isinstance(config, dict):
        return ""
    section = config.get("web_search") or {}
    return str(section.get("searxng_url", "")).strip().rstrip("/")


def _get(base: str, query: str, categories: str, timeout: float) -> dict[str, Any]:
    url = (
        base
        + "/search?"
        + urllib.parse.urlencode({"q": query, "format": "json", "categories": categories})
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


async def search(
    base: str,
    query: str,
    limit: int = 8,
    *,
    categories: str = "general",
    timeout: float = TIMEOUT,
) -> list[dict[str, str]]:
    """Up to `limit` results as {title, url, snippet}. Raises on a failed request."""
    body = await asyncio.to_thread(_get, base, query, categories, timeout)
    rows = []
    for row in body.get("results", []):
        url = str(row.get("url") or "")
        if not url:
            continue
        rows.append(
            {
                "title": " ".join(str(row.get("title") or "").split()),
                "url": url,
                "snippet": " ".join(str(row.get("content") or "").split())[:400],
            }
        )
        if len(rows) >= limit:
            break
    return rows
