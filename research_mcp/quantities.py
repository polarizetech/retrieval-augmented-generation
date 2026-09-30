"""Deterministic lookups for quantities: calculator records in the corpus, and BioNumbers entries.

No model is involved. A value comes back as its source states it, with where it came from, so a
client model cites a number instead of recalling one. BioNumbers entries are cached on first fetch,
so a lookup that has succeeded once gives the same answer offline.
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

BIONUMBERS_URL = "https://bionumbers.hms.harvard.edu/bionumber.aspx?id={bnid}"
USER_AGENT = "retrieval-augmented-generation (math__bionumber; https://github.com/polarizetech)"
#: The labelled fields of a BioNumbers entry page, in the order the page shows them.
BIONUMBER_FIELDS = (
    "Value",
    "Range",
    "Organism",
    "Reference",
    "PubMed ID",
    "Primary Source",
    "Method",
    "Comments",
    "Entered by",
    "ID",
)
PAGE_END = "Related BioNumbers"
WORD = re.compile(r"[a-z0-9]+")


def corpus_root() -> Path | None:
    """The research corpus checkout: $RESEARCH_CORPUS, else a sibling `research` folder."""
    configured = os.environ.get("RESEARCH_CORPUS", "").strip()
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.append(Path(__file__).resolve().parents[2] / "research")
    return next((c for c in candidates if (c / "projects").is_dir()), None)


def _page_lines(page: str) -> list[str]:
    body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", "", page)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", body))
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_bionumber(page: str) -> dict[str, Any]:
    """The fields of a BioNumbers entry page, verbatim. Raises ValueError if the page isn't one."""
    lines = _page_lines(page)
    if "Value" not in lines or "ID" not in lines:
        raise ValueError("not a BioNumbers entry page")
    end = lines.index(PAGE_END) if PAGE_END in lines else len(lines)
    start = lines.index("Value")
    title = lines[start - 1]
    fields: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[start:end]:
        if line in BIONUMBER_FIELDS and line not in fields:
            current = line
            fields[current] = []
        elif current is not None:
            fields[current].append(line)
    value = fields.get("Value", [])
    record: dict[str, Any] = {
        "title": title,
        "value": value[0] if value else "",
        "units": " ".join(value[1:]),
    }
    for name in BIONUMBER_FIELDS[1:]:
        if name in fields:
            record[name.lower().replace(" ", "_")] = " ".join(fields[name])
    return record


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _sections(text: str) -> dict[str, str]:
    """A calculator record's `## ` sections, verbatim."""
    parts = re.split(r"^## ", text, flags=re.M)
    pairs = (p.split("\n", 1) for p in parts[1:] if "\n" in p)
    return {head.strip(): body.strip() for head, body in pairs}


def _commit(root: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() or None


class QuantityStore:
    """The math__ tools: calculator records from the corpus, and BioNumbers entries by BNID."""

    def __init__(
        self,
        corpus: Path | None,
        cache_dir: Path,
        fetch: Callable[[str], str] = _fetch,
    ) -> None:
        self.corpus = corpus
        self.cache_dir = cache_dir
        self.fetch = fetch

    def lookup(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Calculator records (projects/*/calculators/*.md) best matching `query`, with sections."""
        if self.corpus is None:
            raise ValueError("no research corpus found: set RESEARCH_CORPUS to a checkout of it")
        terms = set(WORD.findall(query.lower()))
        if not terms:
            raise ValueError("the query has no words to match")
        scored = []
        for path in sorted(self.corpus.glob("projects/*/calculators/*.md")):
            text = path.read_text(encoding="utf-8")
            words = WORD.findall(text.lower())
            hits = sum(1 for w in words if w in terms)
            covered = len(terms & set(words))
            if covered:
                scored.append((covered, hits, path, text))
        scored.sort(key=lambda s: (-s[0], -s[1], str(s[2])))
        records = []
        for covered, _, path, text in scored[: max(1, min(limit, 20))]:
            heads = (ln[2:].strip() for ln in text.splitlines() if ln.startswith("# "))
            title = next(heads, path.stem)
            records.append(
                {
                    "path": path.relative_to(self.corpus).as_posix(),
                    "project": path.parts[-3],
                    "title": title,
                    "matched_terms": covered,
                    "sections": _sections(text),
                }
            )
        return {
            "records": records,
            "searched": len(list(self.corpus.glob("projects/*/calculators/*.md"))),
            "corpus_commit": _commit(self.corpus),
            "note": "Values are as the record states them; cite the record path and commit.",
        }

    def bionumber(self, bnid: str | int) -> dict[str, Any]:
        """One BioNumbers entry by its ID, verbatim, from the cache or fetched once and cached."""
        ident = str(bnid).strip().upper().removeprefix("BNID").strip()
        if not ident.isdigit():
            raise ValueError(f"a BNID is a number, not {bnid!r}")
        cached = self.cache_dir / "bionumbers" / f"{ident}.json"
        if cached.exists():
            return {**json.loads(cached.read_text(encoding="utf-8")), "cached": True}
        url = BIONUMBERS_URL.format(bnid=ident)
        record = parse_bionumber(self.fetch(url))
        if record.get("id") != ident:
            raise ValueError(f"BioNumbers returned entry {record.get('id')!r} for BNID {ident}")
        record.update(
            {
                "bnid": ident,
                "url": url,
                "retrieved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "cite_as": f"BioNumbers BNID {ident} (Milo et al. 2010, doi:10.1093/nar/gkp889)",
            }
        )
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return {**record, "cached": False}
