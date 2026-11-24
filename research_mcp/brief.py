"""The compact reply `ask` sends to chat clients: one structured brief for the model to answer from.

`ask` builds a full reply (passages, a short relay, a long relay, progress, routing). Sending all of
it made chat replies long and repetitive: the passages appeared twice, and the model was told to
show a fixed block instead of answering. The gateway now records the full reply in the call log
and sends this brief instead. Each passage appears once, with a coverage level that says how well
the library answers the question, and `answer_format` tells the model how to write a short,
cited answer. `ask(..., detail="full")` still returns the full reply.
"""

from __future__ import annotations

from typing import Any

# Reranker (bge-reranker-v2-m3) logits. Relevant passages score above 0 and often above 1;
# near misses sit around -1; unrelated text scores -4 to -6 (see RELEVANCE_FLOOR in ask.py).
STRONG = 1.0
PARTIAL = -1.0

COVERAGE_NOTES = {
    "strong": "The library has passages that bear directly on the question.",
    "partial": "The library has related passages; some may answer only part of the question.",
    "weak": (
        "Only loosely related passages; none clearly answers the question. "
        "Note it in the footer as `Library: weak match`, not in the answer."
    ),
    "none": (
        "No passage in the library matched. Note it in the footer as `Library: no match`, "
        "not in the answer."
    ),
    "unscored": (
        "Passages were not reranked: judge each one's relevance to the question yourself, use "
        "only those that bear on it, and say how well they cover it."
    ),
}

# What the assistant does with the fresh leads: fetch and index them (delegate, the default), or
# suggest asking again with `research:` to fetch and read them (strict).
FRESH_LEADS_NEXT = {
    "delegate": "fetch and index up to five of the closest leads with rag__index_paper, then "
    "retrieve again and answer from what that returns; list any lead that could not be fetched. ",
    "strict": "suggest asking again with `research:` to fetch and read them. ",
}
_ANSWER_BEFORE_LEADS = (
    "Write the reply yourself from `brief`, as a conversation; do not paste the brief. "
    "(1) If `brief.for_you` has items, start with them: the person has to act. "
    "(2) Answer in 2 or 3 sentences when you can. When it needs more, use short lists that are "
    "easy to glance over. Cite each claim that rests on a passage with its evidence id in "
    "backticks, quoting only the few words that matter. "
    "(3) Don't describe what didn't match, what was skipped, left out or not relevant, or what "
    "the library lacks: that goes in the footer. "
    "(4) When `brief.specialists` lists more than one, add **Specialist views**: a sentence or "
    "two per specialist the passages serve, citing evidence ids. When "
    "`brief.tool_results.profiles` is there, list each profile's name and what it is for, and "
    "say a question recruits one by naming it. "
    "(5) When coverage is partial, weak or none and `brief.fresh_search` has papers or web "
    "results, add **Fresh leads**: up to five, each with its link (https://doi.org/ for a DOI), "
    "labelled found but not read; no claim may rest on them; "
)
_ANSWER_AFTER_LEADS = (
    "(6) When the answer cites papers, list them under **Sources** as title (year), doi. "
    "Use only the passages and tool results in `brief`, never facts from memory; anything of "
    "your own goes after, under a heading saying it is yours. "
    "End with a footer after a horizontal rule, at most three short lines: **Tools:** the steps "
    "in `brief.used.tools`, each in two or three words; **Services:** `brief.used.services` "
    "abbreviated (provider, model, balance), with `Library: no match` or `Library: weak match` "
    "when coverage is none or weak, and `brief.limits` in a few words; then the `links`. When "
    "the person asks for more, open the record link: it holds the full reply."
)

SKIPPED = ("skipped", "not run", "Left out")


def _used(reply: dict[str, Any]) -> dict[str, Any]:
    """What ran and what it used, for the reply's footer: the steps that ran (not the skipped
    ones) and the services' usage line, with the research run's model when there is one."""
    tools = [p for p in reply.get("progress") or [] if not any(s in p for s in SKIPPED)]
    services = str(reply.get("usage") or "")
    model = ((reply.get("results") or {}).get("research") or {}).get("model_backend")
    if model:
        services = (
            f"{services} · research model: {model}" if services else f"research model: {model}"
        )
    return {"tools": tools, "services": services}


def answer_format(autonomy: str = "strict") -> str:
    """The answer instructions for a turn: the fresh-leads sentence follows its autonomy."""
    return _ANSWER_BEFORE_LEADS + FRESH_LEADS_NEXT[autonomy] + _ANSWER_AFTER_LEADS


ANSWER_FORMAT = answer_format("strict")


def coverage(found: dict[str, Any] | None) -> dict[str, Any] | None:
    """How well the retrieved passages cover the question, from the best reranker score."""
    if found is None:
        return None
    rows = found.get("results", [])
    scores = [float(r["score"]) for r in rows if isinstance(r.get("score"), (int, float))]
    if not rows:
        level = "none"
    elif not found.get("reranker") or not scores:
        level = "unscored"
    elif max(scores) >= STRONG:
        level = "strong"
    elif max(scores) >= PARTIAL:
        level = "partial"
    else:
        level = "weak"
    out: dict[str, Any] = {"level": level, "note": COVERAGE_NOTES[level]}
    if scores:
        out["best_score"] = round(max(scores), 2)
    if low := found.get("below_floor"):
        out["left_out"] = (
            f"{low['n']} weaker passage(s) scored under the relevance floor "
            f"({low['floor']:g}; best {low['best_score']:.2f})."
        )
    if found.get("excluded"):
        out["excluded"] = f"{len(found['excluded'])} passage(s) left out by the safety check."
    return out


def _source(row: dict[str, Any]) -> str:
    authors = row.get("authors") or []
    lead = f"{authors[0]}{' et al.' if len(authors) > 1 else ''}" if authors else ""
    parts = [str(x) for x in (lead, row.get("title"), row.get("year")) if x]
    if row.get("doi"):
        parts.append(f"doi:{row['doi']}")
    return ", ".join(parts)


def _passage(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": row.get("evidence_id"),
        "source": _source(row),
        "text": " ".join(str(row.get("text", "")).split()),
    }
    if isinstance(row.get("score"), (int, float)):
        out["score"] = round(float(row["score"]), 2)
    return out


LIMITS = {
    "quick": "These are retrieved passages, not a checked answer; start with `research:` for one.",
    "research": "Nothing is concluded until the research run finishes.",
    "code": "Any passages are context only; the change itself is not made or tested yet.",
    "fetch": "The papers are indexed, not read: nothing is answered from them yet.",
}


def compact(reply: dict[str, Any]) -> dict[str, Any]:
    """The client-facing form of a full `ask` reply. Unknown shapes pass through unchanged."""
    if "relay" not in reply or "summary" not in reply:
        return reply
    results = dict(reply.get("results") or {})
    spec = reply.get("specialist") or {}
    summary = reply["summary"]
    mode = (reply.get("route") or {}).get("mode", "quick")
    brief: dict[str, Any] = {}
    notes = dict(reply.get("brief_notes") or {})
    for key in notes:
        if key in results:
            brief[key] = results.pop(key)
    if found := results.pop("evidence", None):
        brief["coverage"] = coverage(found)
        brief["passages"] = [_passage(r) for r in found.get("results", [])]
    if spec.get("name"):
        brief["specialist"] = {"name": spec["name"], "guidance": spec.get("guidance") or ""}
        several = spec.get("specialists") or []
        if len(several) > 1:
            brief["specialists"] = [
                {"name": one["name"], "guidance": one.get("guidance") or ""} for one in several
            ]
    if hits := results.pop("search", None):
        brief["titles_only"] = {
            "note": "Scholarly index titles: no verbatim passages, so not citable.",
            "papers": [_source(r) for r in hits.get("results", [])],
        }
    if (fresh := results.pop("fresh", None)) is not None:
        brief["fresh_search"] = {
            "note": "A fresh paper search (SearXNG's science engines) and web search run with "
            "this request: titles, links and snippets only, nothing read, so not citable.",
            "papers": fresh.get("papers", []),
            "web": fresh.get("web", []),
            "not_run": fresh.get("not_run", []),
        }
    if results:
        brief["tool_results"] = results
    brief["used"] = _used(reply)
    brief["found"] = summary.get("results", [])
    brief["next"] = summary.get("next", [])
    brief["for_you"] = summary.get("for_you", [])
    brief["limits"] = (
        "Nothing was concluded: the request is waiting for the person's choice."
        if reply.get("status") == "needs_decision"
        else "This is the list of installed profiles, not an answer: nothing was retrieved."
        if "profiles" in brief.get("tool_results", {})
        else LIMITS.get(mode, "")
    )
    out: dict[str, Any] = {
        "request_id": reply.get("request_id"),
        "status": reply.get("status"),
        "mode": mode,
        "brief": brief,
        "answer_format": "".join(notes.values())
        + answer_format("delegate" if reply.get("autonomy") == "delegate" else "strict"),
    }
    for key in ("error", "links"):
        if key in reply:
            out[key] = reply[key]
    return out
