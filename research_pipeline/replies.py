"""How replies are laid out: plain text built in code, with no templates.

A short reply is up to three lists, one short sentence per item: what came back, what the
assistant does next, and what the person needs to do (left out when there is nothing).
"""

from __future__ import annotations

from collections.abc import Iterable


def bullets(items: Iterable[str]) -> str:
    """One `- item` line per non-empty item."""
    return "\n".join(f"- {i.strip()}" for i in items if i and i.strip())


def short(results: Iterable[str], nxt: Iterable[str], for_you: Iterable[str] = ()) -> str:
    """The short reply: **Results**, **Next**, and **For you** only when it has something."""
    parts = [f"**Results**\n{bullets(results) or '- Nothing came back.'}"]
    if body := bullets(nxt):
        parts.append(f"**Next**\n{body}")
    if body := bullets(for_you):
        parts.append(f"**For you**\n{body}")
    return "\n\n".join(parts)


def join(*parts: str) -> str:
    """The non-empty parts, a blank line between each."""
    return "\n\n".join(p.strip("\n") for p in parts if p and p.strip())
