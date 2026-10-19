"""Read and deterministically route the provider-neutral scientific agent briefs.

The profile repository remains the source of truth.  This adapter does not execute Markdown as a
prompt.  It records the selected, versioned briefs and translates only their routing boundary into
the pipeline's small domain vocabulary.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORD = re.compile(r"[a-z][a-z0-9-]+")
SPECIALISTS = (
    "auditory-evoked-potentials",
    "visual-evoked-potentials",
    "bone-conduction-auditory",
)
SPECIALIST_TERMS = {
    "auditory-evoked-potentials": {
        "abr",
        "assr",
        "auditory",
        "binaural",
        "cochlear",
        "ffr",
        "neurophonic",
        "tinnitus",
    },
    "visual-evoked-potentials": {
        "erg",
        "ocular",
        "photic",
        "retinal",
        "ssvep",
        "vep",
        "visual",
    },
    "bone-conduction-auditory": {
        "bone",
        "bone-conduction",
        "cross-talk",
        "mastoid",
        "transducer",
    },
}
ROLE_TERMS = {
    "analyst": {"analysis", "dataset", "data", "statistics", "statistical", "regression"},
    "mathematician": {"equation", "derive", "derivation", "constant", "units", "proof"},
    "computational-engineer": {
        "code",
        "software",
        "simulation",
        "modeling",
        "modelling",
        "pipeline",
    },
}
DOMAIN_TERMS = {
    "cardiovascular": {"cardiac", "cardiovascular", "heart", "blood", "pressure", "ecg"},
    "respiratory": {"airway", "breathing", "lung", "pulmonary", "respiratory", "spirometry"},
    "vestibular": {"balance", "vestibular", "vertigo", "vemp", "otolith"},
    "neuroscience": {
        "brain",
        "cortical",
        "eeg",
        "electrophysiology",
        "neural",
        "neuron",
        "neuroscience",
    },
}


@dataclass(frozen=True)
class Brief:
    name: str
    description: str
    path: Path
    sha256: str
    positive: tuple[str, ...]
    negative: tuple[str, ...]


def _words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    block = text.split("---\n", 2)[1]
    out: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _examples(text: str, heading: str) -> tuple[str, ...]:
    match = re.search(rf"(?ms)^{re.escape(heading)}\s*$\n(.*?)(?=^#{{2,3}} |\Z)", text)
    if not match:
        return ()
    return tuple(
        re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", line).strip().strip('“”"')
        for line in match.group(1).splitlines()
        if re.match(r"^\s*(?:[-*]|\d+\.)\s+", line)
    )


def load(root: Path) -> dict[str, Brief]:
    briefs: dict[str, Brief] = {}
    for path in sorted((root / "agents").glob("*.md")):
        text = path.read_text()
        meta = _frontmatter(text)
        name = meta.get("name") or path.stem
        briefs[name] = Brief(
            name=name,
            description=meta.get("description", ""),
            path=path,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            positive=_examples(text, "Positive routing examples:"),
            negative=_examples(text, "Negative routing examples:"),
        )
    return briefs


def _score(question: str, brief: Brief) -> int:
    query = _words(question)
    positive = _words(" ".join((brief.description, *brief.positive)))
    negative = _words(" ".join(brief.negative))
    return 3 * len(query & positive) - 4 * len(query & negative)


def _commit(root: Path) -> str:
    try:
        return subprocess.run(  # noqa: S603 - fixed git argv against an operator-configured repo
            ["/usr/bin/git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def route(question: str, root: Path, explicit_domain: str = "") -> dict[str, Any]:
    """Return an auditable route; never silently invent a missing profile."""
    if not root.is_dir():
        return {
            "available": False,
            "repository": str(root),
            "roles": ["researcher"],
            "specialist": None,
            "domain": explicit_domain or "",
            "reason": "agent profile repository is unavailable",
        }
    briefs = load(root)
    words = _words(question)
    roles = ["researcher"]
    roles.extend(name for name, terms in ROLE_TERMS.items() if words & terms and name in briefs)
    ranked = sorted(
        ((name, _score(question, briefs[name])) for name in SPECIALISTS if name in briefs),
        key=lambda row: (-row[1], row[0]),
    )
    # Profile prose contains intentionally broad research vocabulary, so overlap against the
    # prose alone is too permissive. Require at least one specialist anchor term as well; the
    # score then breaks ties and remains useful in the provenance record.
    anchored = [row for row in ranked if words & SPECIALIST_TERMS.get(row[0], set())]
    specialist = anchored[0][0] if anchored and anchored[0][1] >= 3 else None
    domain = explicit_domain
    if not domain:
        if specialist:
            domain = "neuroscience"
        else:
            scored = sorted(
                ((name, len(words & terms)) for name, terms in DOMAIN_TERMS.items()),
                key=lambda row: (-row[1], row[0]),
            )
            domain = scored[0][0] if scored and scored[0][1] else ""
    selected = [*roles, *([specialist] if specialist else [])]
    records = [
        {
            "name": name,
            "path": str(briefs[name].path.relative_to(root)),
            "sha256": briefs[name].sha256,
            "description": briefs[name].description,
        }
        for name in dict.fromkeys(selected)
        if name in briefs
    ]
    return {
        "available": True,
        "repository": str(root),
        "commit": _commit(root),
        "roles": roles,
        "specialist": specialist,
        "domain": domain,
        "profiles": records,
        "reason": (
            f"specialist score {ranked[0][1]} for {specialist}" if specialist else "general route"
        ),
    }
