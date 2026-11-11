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
        "The library has only loosely related passages: none clearly answers the question. "
        "Say so before anything else."
    ),
    "none": "No passage in the library matched. Say so before answering any other way.",
    "unscored": "Passages were not reranked, so their relevance is unjudged.",
}

ANSWER_FORMAT = (
    "Write the reply yourself from `brief`; do not paste the brief. Keep it short, in this order: "
    "(1) If `brief.for_you` has items, start with them: the person has to act. "
    "(2) **Bottom line**: one or two sentences answering the question as far as the passages "
    "allow; when coverage is weak or none, say first that the library doesn't cover it well. "
    "(3) **Key points**: at most five bullets, each resting on a passage and ending with its "
    "evidence id in backticks; quote only the few words that matter. "
    "(4) **Gaps**: one line on what the passages leave open, and `brief.limits`. "
    "(5) **Sources**: the papers you cited, as title (year), doi. "
    "When `brief.specialists` lists more than one, add **Specialist views** after the key "
    "points: one short paragraph (1-3 sentences) per specialist, reading the same passages "
    "through its guidance and citing evidence ids; a specialist the passages don't serve gets "
    "one line saying so. When `brief.tool_results.profiles` is there, list each profile's name "
    "and what it is for, and say a question recruits one by naming it. "
    "Then the `links`. Use only the passages and tool results in `brief`, never facts from "
    "memory; anything of your own goes after, under a heading saying it is yours. When the "
    "person asks for more, open the record link: it holds the full reply."
)


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
    if results:
        brief["tool_results"] = results
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
        "answer_format": ANSWER_FORMAT,
    }
    for key in ("error", "links"):
        if key in reply:
            out[key] = reply[key]
    return out
