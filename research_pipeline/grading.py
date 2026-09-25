"""Evidence labels are computed from the evidence set. The model gets no vote.

The label describes what was RETRIEVED AND MACHINE-CHECKED for one claim in one run. It is not a
formal evidence grade such as GRADE: nobody has read these papers, and retrieval is not reading.
Nothing a run produces is written back into the paper library or treated as a source by a later
run.
"""

from __future__ import annotations

from typing import Any

from .domains import GENERIC, DomainPolicy
from .schema import Claim, Evidence, source_status, title_key


def label(
    claim: Claim,
    evidence: dict[str, Evidence],
    papers: dict[str, dict[str, Any]],
    contested: bool,
    policy: DomainPolicy = GENERIC,
) -> tuple[str, list[str]]:
    """Return (label, reasons). Only evidence every verifier accepted is counted.

    Which designs count as first-hand primary, and how many independent ones a claim needs before
    the field is willing to call its support strong, are the field's judgements and arrive in
    `policy`. Everything else here is arithmetic over the evidence set.
    """
    primary_designs = policy.taxonomy.primary
    used = [evidence[e] for e in claim.supported_by if e in evidence]
    if claim.verdict not in ("SUPPORTED", "DISPUTED") or not used:
        return "insufficient", ["no cited passage passed verification"]

    # Independent sources: a preprint and its published version are one source, not two.
    sources = {title_key(papers[e.work].get("title")) or e.work for e in used}
    firsthand_primary = {
        title_key(papers[e.work].get("title")) or e.work
        for e in used
        if e.study_type in primary_designs and not e.secondhand
    }
    statuses = {source_status(papers[e.work].get("doi")) for e in used}
    reasons = [
        f"{len(sources)} independent source(s)",
        f"{len(firsthand_primary)} first-hand primary",
    ]
    enough = policy.min_primary_for_strong

    if claim.verdict == "DISPUTED":
        reasons.append("verifiers disagreed")
        return "weak", reasons
    if statuses == {"preprint"}:
        reasons.append("preprint only")
        return "weak", reasons
    if not firsthand_primary:
        reasons.append("rests on reviews, models, or second-hand descriptions of other work")
        return "weak", reasons
    if contested:
        reasons.append("contradicting evidence was retrieved for this sub-question")
        return "moderate" if len(firsthand_primary) >= enough else "weak", reasons
    if len(firsthand_primary) >= enough:
        return "strong", reasons
    reasons.append(
        f"fewer than {enough} independent primary source(s); not independently "
        f"replicated in the retrieved set"
    )
    return "moderate", reasons
