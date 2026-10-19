"""Named evidence policies for a research run.

The names are deliberately qualitative.  A model cannot estimate a calibrated probability that it
hallucinated, but the pipeline can state which checks ran and refuse a publication-grade run when
there is no independent verifier.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Strictness:
    name: str
    description: str
    require_independent_verifier: bool = False
    corpus_ready: bool = False


PROFILES = {
    "explore": Strictness(
        "explore",
        "Broad discovery with the normal citation and number guards; "
        "conclusions remain exploratory.",
    ),
    "grounded": Strictness(
        "grounded",
        "Every reported claim must survive passage, quotation, number and entailment checks.",
    ),
    "publication": Strictness(
        "publication",
        "Grounded checks plus at least one verifier independent of the claim-writing model.",
        require_independent_verifier=True,
        corpus_ready=True,
    ),
}


def resolve(name: str | None) -> Strictness:
    key = (name or "grounded").strip().lower()
    if key not in PROFILES:
        raise ValueError(f"strictness must be one of: {', '.join(PROFILES)}")
    return PROFILES[key]
