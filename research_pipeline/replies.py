"""Reply templates: how the gateway's relay and the pipeline's answer are laid out.

The layout lives in `config/replies.default.md`; the operator's own copy, `config/replies.md`
(gitignored), replaces it template by template. Code still writes every block (passages, claims,
sources, limits); a template only says which blocks appear, in what order, under what headings.

Syntax, deliberately small:
- `<!-- template: name -->` starts a template; text before the first marker is ignored.
- `{{name}}` is replaced by that block (empty when there is nothing to show).
- `{{#name}}...{{/name}}` shows its inside only when the block has something in it.
- `{{^name}}...{{/name}}` shows its inside only when the block is empty.
- Other `<!-- ... -->` comments are notes for the person editing and are dropped.

The files are read on every reply, so an edit shows on the next one without a restart.
REPLY_TEMPLATES names one file to use instead of both (the tests use it).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

from .config import ROOT

DEFAULT_TEMPLATES = ROOT / "config" / "replies.default.md"
LOCAL_TEMPLATES = ROOT / "config" / "replies.md"

_MARKER = re.compile(r"^<!--\s*template:\s*([\w-]+)\s*-->[ \t]*$", re.MULTILINE)
_COMMENT = re.compile(r"<!--.*?-->[ \t]*\n?", re.DOTALL)
_SECTION = re.compile(r"\{\{([#^])\s*([\w-]+)\s*\}\}(.*?)\{\{/\s*\2\s*\}\}", re.DOTALL)
_FIELD = re.compile(r"\{\{\s*([\w-]+)\s*\}\}")
_BLANKS = re.compile(r"\n{3,}")


def parse(text: str) -> dict[str, str]:
    """Templates by name from one file."""
    parts = _MARKER.split(text)
    return {
        parts[i]: _COMMENT.sub("", parts[i + 1]).strip("\n") for i in range(1, len(parts) - 1, 2)
    }


def load(default: Path | None = None, local: Path | None = LOCAL_TEMPLATES) -> dict[str, str]:
    if default is None:
        override = os.environ.get("REPLY_TEMPLATES", "").strip()
        default = Path(override) if override else DEFAULT_TEMPLATES
        local = None if override else local
    templates = parse(default.read_text()) if default.is_file() else {}
    if local is not None and local.is_file():
        templates.update(parse(local.read_text()))
    return templates


def fill(template: str, blocks: Mapping[str, str]) -> str:
    """Fill `template` from `blocks`. An unknown name is left as written, so a typo shows."""

    def section(m: re.Match[str]) -> str:
        has = bool(str(blocks.get(m.group(2), "")).strip())
        show = has if m.group(1) == "#" else not has
        return m.group(3) if show else ""

    def field(m: re.Match[str]) -> str:
        name = m.group(1)
        return str(blocks[name]) if name in blocks else m.group(0)

    previous = None
    text = template
    while previous != text:  # sections may nest
        previous, text = text, _SECTION.sub(section, text)
    text = _FIELD.sub(field, text)
    return _BLANKS.sub("\n\n", text).strip()


def bullets(items: list[str]) -> str:
    """One `- item` line per non-empty item, for the short replies' lists."""
    return "\n".join(f"- {i}" for i in items if i and i.strip())


def render(name: str, blocks: Mapping[str, str]) -> str:
    """Fill the named template. With no template of that name, the blocks in the order given."""
    template = load().get(name)
    if template is None:
        return _BLANKS.sub("\n\n", "\n\n".join(v for v in blocks.values() if str(v).strip()))
    return fill(template, blocks)
