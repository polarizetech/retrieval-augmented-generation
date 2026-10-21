"""Read and deterministically route the provider-neutral scientific agent briefs.

The public profile repository supplies general roles; an optional private overlay supplies
researcher extensions. Only the bounded Pipeline guidance section of a selected brief reaches
the model. The core evidence rules remain controlled by the pipeline.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORD = re.compile(r"[a-z][a-z0-9-]+")
MAX_GUIDANCE_CHARS = 2000
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}
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
    "honeybee-communications-analyst": {
        "bee",
        "bees",
        "honeybee",
        "honeybees",
        "waggle",
        "colony",
        "foraging",
        "electroreception",
    },
    "signal-processing": {
        "fft",
        "fourier",
        "frequency",
        "harmonics",
        "intermodulation",
        "mixing",
        "psychoacoustic",
        "rf",
        "signal-processing",
        "spectral",
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
    root: Path
    path: Path
    sha256: str
    extends: str
    guidance: str
    positive: tuple[str, ...]
    negative: tuple[str, ...]


def _words(text: str) -> set[str]:
    tokens = set(WORD.findall(text.lower()))
    return (tokens | {part for token in tokens for part in token.split("-")}) - STOPWORDS


def _has_specialist_anchor(name: str, words: set[str]) -> bool:
    if name == "bone-conduction-auditory":
        return {"bone", "conduction"} <= words or "mastoid" in words
    return bool(words & SPECIALIST_TERMS.get(name, set(name.split("-"))))


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


def _section(text: str, heading: str) -> str:
    match = re.search(rf"(?ms)^## {re.escape(heading)}\s*$\n(.*?)(?=^## |\Z)", text)
    section = match.group(1).strip() if match else ""
    return section if len(section) <= MAX_GUIDANCE_CHARS else ""


def load(root: Path) -> dict[str, Brief]:
    briefs: dict[str, Brief] = {}
    for path in sorted((root / "agents").glob("*.md")):
        text = path.read_text()
        meta = _frontmatter(text)
        name = meta.get("name") or path.stem
        briefs[name] = Brief(
            name=name,
            description=meta.get("description", ""),
            root=root,
            path=path,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            extends=meta.get("extends", ""),
            guidance=_section(text, "Pipeline guidance"),
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


def route(
    question: str,
    root: Path,
    explicit_domain: str = "",
    extensions_root: Path | None = None,
) -> dict[str, Any]:
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
    extensions_available = extensions_root is None or extensions_root.is_dir()
    if extensions_root is not None and extensions_available:
        for name, brief in load(extensions_root).items():
            if brief.extends == "researcher" and name not in briefs:
                briefs[name] = brief
    words = _words(question)
    roles = ["researcher"]
    roles.extend(name for name, terms in ROLE_TERMS.items() if words & terms and name in briefs)
    ranked = sorted(
        (
            (name, _score(question, brief))
            for name, brief in briefs.items()
            if brief.extends == "researcher" or name in SPECIALIST_TERMS
        ),
        key=lambda row: (-row[1], row[0]),
    )
    # Profile prose contains intentionally broad research vocabulary, so overlap against the
    # prose alone is too permissive. Require at least one specialist anchor term as well; the
    # score then breaks ties and remains useful in the provenance record.
    anchored = [row for row in ranked if _has_specialist_anchor(row[0], words)]
    specialist = anchored[0][0] if anchored and anchored[0][1] >= 3 else None
    domain = explicit_domain
    if not domain:
        if specialist in {
            "auditory-evoked-potentials",
            "visual-evoked-potentials",
            "bone-conduction-auditory",
        }:
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
            "repository": str(briefs[name].root),
            "commit": _commit(briefs[name].root),
            "path": str(briefs[name].path.relative_to(briefs[name].root)),
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
        "extensions_repository": str(extensions_root) if extensions_root else "",
        "extensions_available": extensions_available,
        "extensions_commit": (
            _commit(extensions_root) if extensions_root and extensions_root.is_dir() else ""
        ),
        "roles": roles,
        "specialist": specialist,
        "guidance": briefs[specialist].guidance if specialist else "",
        "domain": domain,
        "profiles": records,
        "reason": (
            f"specialist score {next(score for name, score in ranked if name == specialist)} "
            f"for {specialist}"
            if specialist
            else "general route"
        )
        + (
            f"; private overlay unavailable at {extensions_root}"
            if not extensions_available
            else ""
        ),
    }
