"""Assemble the answer from verified records. No model writes here.

The bibliography is printed from the library's stored metadata, keyed by evidence id. The model
never types an author, a year, or a DOI, so it cannot invent one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .domains import GENERIC, DomainPolicy
from .schema import source_status
from .verify import VERDICT_NUMBERS, VERDICT_UNCHECKED

if TYPE_CHECKING:
    from .pipeline import State

LABEL_TEXT = {
    "strong": "Strong",
    "moderate": "Moderate",
    "weak": "Weak",
    "insufficient": "Insufficient",
}


def _cite(paper: dict[str, Any]) -> str:
    authors = paper.get("authors") or []
    lead = authors[0].split()[-1] if authors else "Unknown"
    lead += " et al." if len(authors) > 1 else ""
    return f"{lead} {paper.get('year') or 'n.d.'}"


def render(
    st: State,
    library: dict[str, dict[str, Any]],
    models: dict[str, Any],
    policy: DomainPolicy = GENERIC,
) -> str:
    """`library`: what the paper library said about each cited work (title, year, doi, authors)."""
    assert st.plan
    papers: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def ref(work: str) -> int:
        if work not in papers:
            papers[work] = library.get(work) or {"work": work}
            order.append(work)
        return order.index(work) + 1

    def claim_lines(claims) -> list[str]:
        lines = []
        for c in claims:
            used = c.supported_by or c.evidence_ids
            marks = "".join(f"[{n}]" for n in sorted({ref(st.evidence[e].work) for e in used}))
            disputed = " *(verifiers disagreed)*" if c.verdict == "DISPUTED" else ""
            lines.append(
                f"- {c.text} {marks}{disputed}  \n"
                f"  *Evidence: {LABEL_TEXT[c.label]} — {'; '.join(c.flags[-3:])}*"
            )
        return lines

    kept = [c for c in st.claims if c.verdict in ("SUPPORTED", "DISPUTED")]
    removed = [
        c
        for c in st.claims
        if c.verdict
        not in ("SUPPORTED", "DISPUTED", "RETRACTED_SOURCE", VERDICT_NUMBERS, VERDICT_UNCHECKED)
    ]
    unchecked = [c for c in st.claims if c.verdict == VERDICT_UNCHECKED]
    wrong_numbers = [c for c in st.claims if c.verdict == VERDICT_NUMBERS]
    out = [f"# {st.question}", ""]

    out += ["## Bottom line", ""]
    bottom = [s for s in st.summary if s["kept"]]
    if bottom:
        out.append(" ".join(s["text"] for s in bottom))
    elif kept:
        out.append(
            "The verified claims below could not be combined into a summary that itself passed "
            "verification. Read them individually."
        )
    else:
        out.append(
            "**The retrieved evidence is insufficient to answer this question.** No claim "
            "survived verification against a retrieved full-text passage. This reflects what "
            "could be found and read in this run, not what exists."
        )
    out.append("")

    out += ["## Findings", ""]
    for sq in st.plan.subquestions:
        if sq.kind == "falsification":
            continue
        out += [f"### {sq.text}", ""]
        mine = [c for c in kept if c.subquestion == sq.id]
        out += (
            claim_lines(mine)
            if mine
            else [
                "- *Insufficient evidence: no claim for this sub-question survived verification.*"
            ]
        )
        out.append("")

    # Disagreement is computed from the direction of each retrieved result, never asserted by a
    # model.
    out += ["## Conflicting or negative evidence", ""]
    f1 = next(sq for sq in st.plan.subquestions if sq.id == "F1")
    wrote = False
    for sq in st.plan.subquestions:
        # First-hand results only, and only when different papers are on each side: one paper
        # describing earlier positive reports and then its own null result is not two sources.
        rows = [e for e in st.evidence.values() if e.subquestion == sq.id and e.role == "E"]
        yes = sorted({ref(e.work) for e in rows if e.direction == "affirms"})
        no = sorted({ref(e.work) for e in rows if e.direction == "denies"})
        if yes and no and set(yes) != set(no):
            wrote = True
            out.append(
                f"- **Sources disagree** on “{sq.text}”: reporting the effect "
                + "".join(f"[{n}]" for n in yes)
                + "; reporting a null or opposite result "
                + "".join(f"[{n}]" for n in no)
                + "."
            )
    # A claim found by the search for null results is opposing evidence only if its own evidence
    # reports a null or mixed result; the search often finds supporting papers instead.
    from_falsification = [c for c in kept if c.subquestion == "F1"]
    opposing = [
        c
        for c in from_falsification
        if any(
            st.evidence[e].direction in ("denies", "mixed")
            for e in c.supported_by or c.evidence_ids
        )
    ]
    supporting = [c for c in from_falsification if c not in opposing]
    if opposing:
        wrote = True
        out.append("- Found by searching specifically for null results and failed replications:")
        out += ["  " + line for line in claim_lines(opposing)]
    if not wrote:
        out.append(
            "- No opposing result was retrieved. Searches run to find one: "
            + "; ".join(f"“{q}”" for q in f1.queries)
            + ". An empty result is not evidence that the finding is uncontested."
        )
    if supporting:
        out.append(
            "- The searches for null results and failed replications found supporting "
            "results instead:"
        )
        out += ["  " + line for line in claim_lines(supporting)]
    out.append("")

    out += ["## Limits of this answer", ""]
    if removed:
        out.append(
            f"- {len(removed)} drafted claim(s) were removed because their cited passage did "
            "not support them: "
            + "; ".join(
                f"“{c.text[:90]}” ({c.verdict.lower().replace('_', ' ')})" for c in removed[:4]
            )
        )
    if wrong_numbers:
        out.append(
            f"- {len(wrong_numbers)} drafted claim(s) were removed because they stated a number "
            "their cited passage does not contain: "
            + "; ".join(f"“{c.text[:90]}”" for c in wrong_numbers[:4])
        )
    if unchecked:
        out.append(
            f"- {len(unchecked)} drafted claim(s) were removed because the verifier returned no "
            "verdict on them (the call ran out of time or tokens): "
            + "; ".join(f"“{c.text[:90]}”" for c in unchecked[:4])
        )
    unread = [c for c in st.candidates.values() if c.outcome == "not_obtainable"]
    if unread:
        unread.sort(key=lambda c: -c.relevance)
        out.append(
            "- Relevant-looking papers that could not be read (no open-access copy), so they "
            "contributed nothing: "
            + "; ".join(
                f"{c.title} ({c.year}{', doi:' + c.ids['doi'] if c.ids.get('doi') else ''})"
                for c in unread[:6]
            )
        )
    skipped = [c for c in st.candidates.values() if c.outcome == "over_budget"]
    if skipped:
        out.append(f"- {len(skipped)} further candidate paper(s) were not fetched (fetch budget).")
    pulled = [c for c in st.claims if c.verdict == "RETRACTED_SOURCE"]
    if pulled:
        out.append(
            f"- {len(pulled)} claim(s) were removed because Crossref/Retraction Watch lists "
            "their source as retracted."
        )
    retracted = [c for c in st.candidates.values() if c.outcome == "retracted"]
    if retracted:
        out.append("- Excluded as retracted: " + "; ".join(str(c.title) for c in retracted))
    silent = sorted({p for s in st.searches for p in s.get("did_not_answer", [])})
    if silent:
        out.append(
            f"- Search providers that did not answer at least once: {', '.join(silent)}. "
            "Their silence is not zero results."
        )
    injected = [d for d in st.dropped_passages if "prompt_injection_pattern" in d["flags"]]
    if injected:
        out.append(
            f"- {len(injected)} passage(s) were excluded for containing text addressed to an AI "
            "reader."
        )
    hidden = [d for d in st.dropped_passages if "hidden_characters" in d["flags"]]
    if hidden:
        out.append(f"- {len(hidden)} passage(s) were excluded for containing hidden characters.")
    out.append(
        "- Evidence is text passages only. Results that appear only in figures, tables or "
        "supplements were not seen."
    )
    out.append(
        f"- Searched as {policy.label}: {policy.scope} If the question falls outside that "
        "scope, these rules for what counts as evidence may not fit it."
    )
    out.append(
        "- Every quoted passage was retrieved and machine-checked, not read by a person. The "
        "evidence labels are computed from the retrieved set and are not formal evidence grades."
    )
    out += [f"- {n}" for n in st.notes if not n.startswith("fetch ")]
    out.append("")

    out += ["## Sources", ""]
    for n, work in enumerate(order, 1):
        p = papers[work]
        status = source_status(p.get("doi"))
        tag = (
            "PREPRINT"
            if status == "preprint"
            else "published"
            if status == "published"
            else "status unknown"
        )
        record = getattr(st, "integrity", {}).get(p.get("doi") or "", {})
        editorial = {
            "no_notice_found": "no retraction or concern found in Crossref",
            "corrected": "CORRECTED",
            "partially_retracted": "PARTIALLY RETRACTED",
            "expression_of_concern": "EXPRESSION OF CONCERN",
            "retracted": "RETRACTED",
            "not_in_crossref": "not registered with Crossref; editorial status unknown",
        }.get(record.get("status"), "editorial status not checked")
        venue = f" {record['venue']}." if record.get("venue") else ""
        out.append(
            f"{n}. {_cite(p)}. {p.get('title') or work}.{venue} "
            f"{'doi:' + p['doi'] if p.get('doi') else work} — *{tag}; {editorial}; "
            f"full text via {p.get('route') or '?'}*"
        )
        for e in st.evidence.values():
            if e.work == work and any(e.id in (c.supported_by or c.evidence_ids) for c in kept):
                hand = "reporting other work" if e.secondhand else e.study_type.replace("_", " ")
                out.append(f"   - [{e.id}, {hand}, passage chars {e.start}–{e.end}] “{e.quote}”")
    if not order:
        out.append("*No source supported a verified claim.*")
    out.append("")

    verifiers = ", ".join(v["name"] for v in models.get("verifiers", []))
    out.append(
        f"*Generated by {models['text']['name']}; verified by {verifiers}; "
        f"{len(st.searches)} searches, {len(st.evidence)} evidence passages, "
        f"{len(kept)}/{len(st.claims)} claims kept.*"
    )
    return "\n".join(out) + "\n"
