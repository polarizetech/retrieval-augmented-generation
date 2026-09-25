"""Records that flow between stages. Everything here is serialised into the run log."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .domains import GENERIC

MODES = [
    "factual_lookup",
    "single_paper_analysis",
    "multi_paper_synthesis",
    "state_of_field",
    "hypothesis_pressure_test",
    "contradiction_search",
    "methodological_comparison",
]
# Direction is relative to the effect or relationship the question asks about, not to a hoped-for
# answer: "affirms" = the passage reports it exists, "denies" = null result, absence, or failed
# replication. Disagreement between sources is then computed, not guessed.
DIRECTIONS = ["affirms", "denies", "mixed", "neutral"]
# Study designs belong to a field, not to the engine: the live list comes from the running
# domain's taxonomy. This is the generic one, used when no domain is installed.
STUDY_TYPES = GENERIC.taxonomy.keys
VERDICTS = ["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "NOT_SUPPORTED"]

# DOI prefixes of preprint servers. A preprint is labelled wherever it is used; it is not excluded.
PREPRINT_PREFIXES = {
    "10.1101": "bioRxiv/medRxiv",
    "10.48550": "arXiv",
    "10.31234": "PsyArXiv",
    "10.31219": "OSF",
    "10.21203": "Research Square",
    "10.20944": "Preprints.org",
    "10.2139": "SSRN",
    "10.26434": "ChemRxiv",
    "10.22541": "Authorea",
    "10.12688": "F1000/Wellcome Open (post-pub review)",
}


def source_status(doi: str | None) -> str:
    if not doi:
        return "unknown"
    return "preprint" if doi.lower().split("/", 1)[0] in PREPRINT_PREFIXES else "published"


def title_key(title: str | None) -> str:
    """Normalised title, used to fold a preprint and its published version into one source."""
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


@dataclass
class SubQuestion:
    id: str
    text: str
    kind: str = "evidence"  # evidence | falsification | gap
    queries: list[str] = field(default_factory=list)


@dataclass
class Plan:
    question: str
    mode: str
    core_question: str
    subquestions: list[SubQuestion]


@dataclass
class Candidate:
    """A paper surfaced by discovery, before anyone has read it."""

    key: str  # doi, else pmid/pmcid/openalex, else normalised title
    title: str | None
    year: int | None
    ids: dict[str, Any]
    authors: list[str]
    is_oa: bool | None
    providers: list[str]
    queries: list[str] = field(default_factory=list)
    work: str | None = None
    full_text_in_library: bool = False
    relevance: float = 0.0
    outcome: str = (
        "not_attempted"  # indexed | not_obtainable | fetch_failed | over_budget | retracted
    )


@dataclass
class Evidence:
    """One passage's bearing on one sub-question. `quote` is verbatim from the stored full text."""

    id: str
    subquestion: str
    work: str
    passage_id: int
    start: int
    end: int
    direction: str
    finding: str
    quote: str
    study_type: str
    population: str
    secondhand: bool  # the passage reports another paper's result, not this paper's own
    role: (
        str  # E evidential (a first-hand result) | O orienting (context, or another paper's result)
    )
    rerank_score: float | None = None
    flags: list[str] = field(default_factory=list)


@dataclass
class Check:
    verifier: str
    evidence_id: str
    verdict: str
    reason: str


@dataclass
class Claim:
    id: str
    subquestion: str
    text: str
    evidence_ids: list[str]
    checks: list[Check] = field(default_factory=list)
    verdict: str = (
        "UNVERIFIED"  # SUPPORTED | DISPUTED | PARTIALLY_SUPPORTED | CONTRADICTED | NOT_SUPPORTED
    )
    supported_by: list[str] = field(default_factory=list)  # evidence ids every verifier accepted
    flags: list[str] = field(default_factory=list)
    contested_by: list[str] = field(default_factory=list)  # evidence ids pointing the other way
    label: str = ""


def to_json(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [to_json(o) for o in obj]
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    return obj
