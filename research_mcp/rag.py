"""Deterministic evidence tools for client-mediated research: retrieve, check, save.

Here the calling model (Claude, ChatGPT, a local chat model) writes the answer, and these tools hold
it to the same deterministic rules the pipeline applies to its own drafts:

- evidence comes only from the paper library's passage index (paper-fetch), under ids that name
  an exact passage;
- passages flagged by the safety scan, and passages from retracted papers, are never returned;
- a claim is valid only if each quote it relies on occurs in the cited passage, it relies on at
  least one such quote, and every number it states occurs in the passages it cites.

What these tools cannot do is judge entailment: whether the passage, read in full, supports the
claim. That needs a verifier model, which is what `python -m research_pipeline ask` adds. A client
that uses only these tools gets existence, quote and number checks, and nothing more.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_pipeline import safety
from research_pipeline.papers import PaperLibrary, PapersError, Passage
from research_pipeline.verify import anchor_quote, unsupported_numbers

EVIDENCE_ID = re.compile(r"^(?P<work>.+)#p(?P<ord>\d+)\.(?P<digest>[0-9a-f]{8})$")


def evidence_id(passage: Passage) -> str:
    """`<work>#p<position>.<text hash>`: stable across re-indexing; stale if the text changes."""
    return f"{passage.work}#p{passage.ord}.{_digest(passage.text)}"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


class EvidenceStore:
    """The rag__ tools' logic. `library` returns a client for the paper library's MCP session."""

    def __init__(self, library: Callable[[], PaperLibrary], runs_dir: Path):
        self.library = library
        self.runs_dir = runs_dir

    # -- retrieval ---------------------------------------------------------------------------
    async def retrieve(self, query: str, limit: int = 8, max_chars: int = 1800) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("query is required")
        limit = max(1, min(int(limit), 20))
        max_chars = max(400, min(int(max_chars), 4000))
        # Over-fetch so that excluded passages do not shrink the result below `limit`.
        res = await self.library().retrieve([query], limit=limit * 3)
        results, excluded = [], []
        for passage in (Passage.from_json(p) for p in res["results"][0]["passages"]):
            paper = passage.paper
            flags = safety.scan(passage.text) + (["retracted"] if paper.get("is_retracted") else [])
            if flags:
                excluded.append({"evidence_id": evidence_id(passage), "flags": flags})
                continue
            if len(results) == limit:
                continue
            results.append(
                {
                    "evidence_id": evidence_id(passage),
                    "work": passage.work,
                    "start": passage.start,
                    "end": passage.end,
                    "text": passage.text[:max_chars],
                    "truncated": len(passage.text) > max_chars,
                    "title": paper.get("title"),
                    "doi": paper.get("doi"),
                    "pmid": paper.get("pmid"),
                    "year": paper.get("year"),
                    "authors": paper.get("authors") or [],
                    "bm25_rank": passage.bm25_rank,
                    "dense_rank": passage.dense_rank,
                    "score": passage.fused,
                }
            )
        index = res.get("index") or {}
        return {
            "query": query,
            "retrieval": "hybrid" if index.get("embedding_model") else "lexical",
            "n": len(results),
            "results": results,
            "excluded": excluded,
            "index": index,
        }

    # -- checking ----------------------------------------------------------------------------
    async def check_citations(self, claims: list[dict[str, Any]]) -> dict[str, Any]:
        checked = [await self._check_claim(number, claim) for number, claim in enumerate(claims, 1)]
        return {
            "n": len(checked),
            "valid": bool(checked) and all(c["valid"] for c in checked),
            "claims": checked,
        }

    async def _check_claim(self, number: int, claim: dict[str, Any]) -> dict[str, Any]:
        text = str(claim.get("text", "")).strip()
        evidence_ids = [str(e) for e in claim.get("evidence_ids") or []]
        quotes = claim.get("quotes") or {}
        errors: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        passages: list[str] = []
        anchored_quotes = 0

        if not evidence_ids:
            errors.append({"error": "a claim must cite at least one evidence_id"})
        for eid in evidence_ids:
            passage, problem = await self._resolve(eid)
            if passage is None:
                errors.append({"evidence_id": eid, "error": problem})
                continue
            passages.append(passage.text)
            item: dict[str, Any] = {"evidence_id": eid, "valid": True}
            quote = str(quotes.get(eid, "")).strip()
            if quote:
                anchored, ratio = anchor_quote(quote, passage.text)
                item.update({"quote": anchored, "match": round(ratio, 3)})
                if anchored is None:
                    item["valid"] = False
                    errors.append({"evidence_id": eid, "error": "quote not found in passage"})
                else:
                    anchored_quotes += 1
            evidence.append(item)

        if evidence_ids and not anchored_quotes and not errors:
            errors.append({"error": "a claim must quote at least one cited passage verbatim"})
        missing = unsupported_numbers(text, passages) if passages else []
        if missing:
            errors.append({"error": "numbers not in cited evidence", "numbers": missing})
        return {
            "claim_index": number,
            "claim": text,
            "evidence_ids": evidence_ids,
            "valid": not errors,
            "evidence": evidence,
            "errors": errors,
        }

    async def _resolve(self, eid: str) -> tuple[Passage | None, str]:
        match = EVIDENCE_ID.match(eid)
        if not match:
            return (
                None,
                "malformed evidence_id; expected <work>#p<n>.<hash> from rag__retrieve_evidence",
            )
        ord_ = int(match["ord"])
        try:
            found = await self.library().passages(match["work"], ord_, 1)
        except PapersError as exc:
            if exc.code != "not_found":
                raise ValueError(f"the paper library did not answer: {exc}") from exc
            found = []
        passage = next((p for p in found if p.ord == ord_), None)
        if passage is None:
            return None, "unknown evidence_id"
        if _digest(passage.text) != match["digest"]:
            return None, "stale evidence_id: the passage text changed after it was retrieved"
        if passage.paper.get("is_retracted"):
            return None, "the cited paper is retracted"
        if flags := safety.scan(passage.text):
            return None, f"the cited passage is excluded ({', '.join(flags)})"
        return passage, ""

    # -- saving ------------------------------------------------------------------------------
    async def save_report(
        self,
        title: str,
        markdown: str,
        evidence_ids: list[str],
        claims: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Save a report with a manifest of the exact passages it cites.

        Every evidence id must resolve. When `claims` are given they must pass check_citations and
        the check is stored with the report; without them the manifest says the claims were not
        checked, so a reader can tell the difference.
        """
        title, markdown = title.strip(), markdown.strip()
        if not title or not markdown:
            raise ValueError("title and markdown are required")
        cited: dict[str, dict[str, Any]] = {}
        for eid in dict.fromkeys(evidence_ids):
            passage, problem = await self._resolve(eid)
            if passage is None:
                raise ValueError(f"{eid}: {problem}")
            cited[eid] = {
                "work": passage.work,
                "start": passage.start,
                "end": passage.end,
                "text": passage.text,
            }
        if not cited:
            raise ValueError("a report must cite at least one evidence_id")
        check = await self.check_citations(claims) if claims else None
        if check is not None and not check["valid"]:
            raise ValueError("claims failed check_citations; run it and fix the reported errors")

        created = datetime.now(UTC)
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80] or "report"
        run_dir = self.runs_dir / f"rag-{created:%Y%m%dT%H%M%SZ}-{slug}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "answer.md").write_text(markdown + "\n")
        manifest = {
            "title": title,
            "created_at": created.isoformat(timespec="seconds"),
            "evidence": cited,
            "claims_checked": check is not None,
            "check": check,
        }
        (run_dir / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        return {
            "run_dir": str(run_dir),
            "answer": str(run_dir / "answer.md"),
            "evidence_ids": list(cited),
            "claims_checked": check is not None,
        }

    async def stats(self) -> dict[str, Any]:
        return (await self.library().status()).get("passages") or {}

    # -- indexing ----------------------------------------------------------------------------
    async def index_papers(self, identifiers: list[str]) -> dict[str, Any]:
        """Fetch papers through the library and have it index their full text, so retrieve can
        return and cite them. Each result says what happened, and why not when it did not."""
        lib = self.library()
        results, works = [], []
        for ident in identifiers[:10]:
            try:
                rec = await lib.fetch(ident)
            except PapersError as exc:
                results.append({"identifier": ident, "status": exc.code, "error": str(exc)})
                continue
            if rec.get("is_retracted"):
                results.append({"identifier": ident, "status": "retracted"})
            elif not rec.get("full_text"):
                results.append(
                    {"identifier": ident, "status": "no_open_full_text", "title": rec.get("title")}
                )
            else:
                works.append(rec["work"])
                results.append(
                    {
                        "identifier": ident,
                        "work": rec["work"],
                        "doi": rec.get("doi"),
                        "title": rec.get("title"),
                    }
                )
        if not works:
            return {"results": results, "index": await self.stats()}
        try:
            report = await lib.index(works)
        except PapersError as exc:
            for r in results:
                if r.get("work"):
                    r.update(status="not_indexed", error=str(exc))
            return {"results": results, "index": await self.stats()}
        fresh = set(report.get("indexed") or [])
        for r in results:
            if r.get("work"):
                r["status"] = "indexed" if r["work"] in fresh else "already_indexed"
        return {"results": results, "index": report.get("index") or {}}
