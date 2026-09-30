"""Domain extensions: what a scientific field adds on top of the domain-neutral engine.

The engine in this package knows how to plan searches, retrieve passages, extract findings,
synthesise claims, verify them against their sources and grade what survived. It does not know
what a study design is in your field, what a number with a unit means, or when a result may not be
generalised. That knowledge is a *domain*, and a
domain is a separately versioned package that declares itself here.

The contract is deliberately data, not behaviour. A domain hands the engine a `DomainPolicy`: a
study-design taxonomy, grading and generalisation rules, measures and their units, and critique
rules. That is the *evidence* half of a field. Its *search* half (the terms the field is indexed
under, its authoritative sources, how to write its queries) is a discipline profile in the paper
library, paper-fetch, named by `DomainPolicy.profile`. The engine reads that policy; it never calls
back into the domain to make a decision, and it never imports a domain package. This keeps the
dependency arrow pointing one way and keeps a run reproducible from its log, because a policy
serialises.

    domain package __init__.py            engine
    ---------------------------           ------
    declare(name=..., policy=...)  ---->  _REGISTRY
    entry point "rag.domains" -->  discover() / load()

The usual plugin pattern: a distribution name prefix, import-time declaration, a version gate,
entry-point discovery, and no core-side list of extensions.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

ENTRY_POINT_GROUP = "rag.domains"
NAME_PREFIX = "rag-domain-"


class DomainError(Exception):
    """Base class for every problem with a domain declaration."""


class IncompatibleCore(DomainError):
    """The installed engine does not satisfy a domain's declared version range."""


class MissingDomain(DomainError):
    """A domain extends another domain that is not installed, or is the wrong version."""


class UnknownDomain(DomainError):
    """A domain was asked for by name or slug and is not registered."""


# -- version arithmetic --------------------------------------------------------------------
# Deliberately small: dotted integers and comma-separated comparisons. A domain that needs more
# than this is expressing a packaging problem, not a compatibility one.

_OPS = ("<=", ">=", "==", "!=", "<", ">")


def parse_version(value: str) -> tuple[int, ...]:
    text = value.strip().split("+", 1)[0]
    if not re.fullmatch(r"\d+(\.\d+)*", text):
        raise ValueError(f"version {value!r} must be dotted integers")
    return tuple(int(part) for part in text.split("."))


def _pad(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    width = max(len(a), len(b))
    return a + (0,) * (width - len(a)), b + (0,) * (width - len(b))


def satisfies(version: str, spec: str) -> bool:
    """True when `version` satisfies every comma-separated comparison in `spec`."""
    found = parse_version(version)
    for raw in spec.split(","):
        clause = raw.strip()
        if not clause:
            continue
        op = next((o for o in _OPS if clause.startswith(o)), None)
        if op is None:
            raise ValueError(f"specifier {clause!r} must start with one of {_OPS}")
        left, right = _pad(found, parse_version(clause[len(op) :]))
        if not {
            "<=": left <= right,
            ">=": left >= right,
            "==": left == right,
            "!=": left != right,
            "<": left < right,
            ">": left > right,
        }[op]:
            return False
    return True


def require(domain: str, spec: str) -> str:
    """Check the imported engine against a domain's declared range, and return its version."""
    from . import __version__

    if not satisfies(__version__, spec):
        raise IncompatibleCore(
            f"{domain} requires the research pipeline {spec}, but imported {__version__}."
        )
    return __version__


# -- the policy ----------------------------------------------------------------------------


@dataclass(frozen=True)
class StudyDesign:
    """One kind of study the field produces, and what the engine may do with it.

    `primary` marks a design that reports its own first-hand observations. Only first-hand primary
    evidence can raise a claim's label above "weak": a review restating a result is a pointer to
    the study, not a second study. `rank` orders designs by how much weight the field gives them
    (higher is stronger) and is what a GRADE-style rubric sorts on; it is a field's opinion, so it
    lives here and not in the engine.
    """

    key: str
    label: str
    definition: str  # one clause, written to be read by a small model inside a classifier prompt
    primary: bool = False
    rank: int = 0
    human: bool = False  # evidence about people, i.e. no species gap to cross

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.key):
            raise ValueError(f"study design key {self.key!r} must be lower_snake_case")
        if not self.definition.strip():
            raise ValueError(f"study design {self.key!r} needs a definition for the classifier")


@dataclass(frozen=True)
class Taxonomy:
    """The study designs of a field, plus the `unclear` escape hatch the engine always needs."""

    designs: tuple[StudyDesign, ...]

    def __post_init__(self) -> None:
        keys = [d.key for d in self.designs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate study design keys")
        if "unclear" not in keys:
            raise ValueError(
                "a taxonomy must include an 'unclear' design: the classifier needs "
                "somewhere to put a paper whose text does not say"
            )

    @property
    def keys(self) -> list[str]:
        return [d.key for d in self.designs]

    @property
    def primary(self) -> frozenset[str]:
        return frozenset(d.key for d in self.designs if d.primary)

    @property
    def human(self) -> frozenset[str]:
        return frozenset(d.key for d in self.designs if d.human)

    def get(self, key: str) -> StudyDesign | None:
        return next((d for d in self.designs if d.key == key), None)

    def rank(self, key: str) -> int:
        design = self.get(key)
        return design.rank if design else 0

    def classifier_prose(self) -> str:
        """The definitions, rendered for the paper-classification prompt."""
        return " ".join(f"{d.key} = {d.definition.rstrip('.')};" for d in self.designs)


@dataclass(frozen=True)
class Measure:
    """A quantity the field reports, with the units it is legitimately reported in.

    The engine checks every number in a claim against its cited passages. That check is blind to
    units, so "10 Hz" matches a passage that only ever said "10 ms". A domain that declares its
    measures gets the stricter check: a number carrying a unit must be found in the source with a
    unit the same measure allows.
    """

    key: str
    label: str
    units: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if not self.units:
            raise ValueError(f"measure {self.key!r} declares no units; omit it instead")


@dataclass(frozen=True)
class CritiqueRule:
    """A weakness the field recognises in an evidence set, stated as a design test.

    When every design behind a claim falls inside `designs`, the critic raises `problem` and the
    planner is asked for a search that would fix it. This is how "animal or in-vitro evidence
    only" stops being an assumption in the engine and becomes something a field declares.
    """

    key: str
    designs: frozenset[str]
    problem: str

    def triggers(self, designs: Iterable[str]) -> bool:
        found = set(designs)
        return bool(found) and found <= set(self.designs)


@dataclass(frozen=True)
class DomainPolicy:
    """Everything a field tells the engine. Serialisable, inspectable, and version-stamped."""

    slug: str
    label: str
    scope: str  # one sentence, shown in the answer so a reader knows what was searched

    taxonomy: Taxonomy
    measures: tuple[Measure, ...] = ()
    critique_rules: tuple[CritiqueRule, ...] = ()

    # Prompt fragments. Each is appended to an engine prompt; none replaces one, so a domain can
    # tighten the rules but cannot talk the engine out of them.
    generalisation_rule: str = ""  # what may not be carried across populations or preparations
    reporting_rule: str = ""  # what a finding must keep to stay meaningful in this field
    gap_guidance: str = ""  # what the critic should look for that is missing

    # Grading. `min_primary_for_strong` is the number of independent first-hand primary sources a
    # claim needs before the field calls its retrieved support strong.
    min_primary_for_strong: int = 2
    examples: tuple[str, ...] = ()  # questions this domain is built to answer, for docs and evals
    # The paper library's discipline profile for this field: its search half. None means the
    # profile with the domain's own slug; "" means none (plan and search without one).
    profile: str | None = None

    def __post_init__(self) -> None:
        if self.profile is None:
            object.__setattr__(self, "profile", self.slug)
        if not re.fullmatch(r"[a-z][a-z0-9-]*", self.slug):
            raise ValueError(f"domain slug {self.slug!r} must be lower kebab-case")
        if not self.scope.strip():
            raise ValueError(f"{self.slug}: describe the scope; a reader is shown it in the answer")
        if self.min_primary_for_strong < 1:
            raise ValueError(f"{self.slug}: min_primary_for_strong must be at least 1")
        unknown = {k for r in self.critique_rules for k in r.designs} - set(self.taxonomy.keys)
        if unknown:
            raise ValueError(f"{self.slug}: critique rules name unknown designs {sorted(unknown)}")

    def unit_words(self) -> frozenset[str]:
        return frozenset(u.lower() for m in self.measures for u in m.units)

    def measure_for_unit(self, unit: str) -> Measure | None:
        lowered = unit.lower()
        return next((m for m in self.measures if lowered in {u.lower() for u in m.units}), None)

    def summary(self) -> dict[str, Any]:
        """What goes in the run log: enough to reproduce the policy's effect on a run."""
        return {
            "slug": self.slug,
            "label": self.label,
            "scope": self.scope,
            "study_designs": self.taxonomy.keys,
            "primary_designs": sorted(self.taxonomy.primary),
            "profile": self.profile,
            "measures": [m.key for m in self.measures],
            "critique_rules": [r.key for r in self.critique_rules],
            "min_primary_for_strong": self.min_primary_for_strong,
        }


@dataclass(frozen=True)
class Domain:
    """What a domain package says about itself. Declared once, in its `__init__.py`."""

    name: str  # distribution-style, "rag-domain-<slug>"
    module: str
    version: str
    core_requires: str
    job: str  # one line: the one field this adds
    policy: DomainPolicy
    extends: dict[str, str] = field(default_factory=dict)
    core_version: str = ""

    @property
    def slug(self) -> str:
        return self.policy.slug


# -- registry ------------------------------------------------------------------------------

_REGISTRY: dict[str, Domain] = {}


def declare(
    *,
    name: str,
    module: str,
    version: str,
    core_requires: str,
    job: str,
    policy: DomainPolicy,
    extends: dict[str, str] | None = None,
) -> Domain:
    """Declare a domain, check it against the engine and anything it extends, and register it."""
    if not name.startswith(NAME_PREFIX):
        raise ValueError(f"domain name {name!r} must start with {NAME_PREFIX!r}")
    if not job.strip():
        raise ValueError(f"{name}: declare the one job this domain adds")

    parse_version(version)
    found = require(name, core_requires)

    extends = dict(extends or {})
    for base, spec in extends.items():
        have = _REGISTRY[base].version if base in _REGISTRY else _installed_version(base)
        if have is None:
            raise MissingDomain(f"{name} extends {base}, which is not installed")
        if not satisfies(have, spec):
            raise MissingDomain(f"{name} requires {base} {spec}, but found {have}")

    prior = _REGISTRY.get(name)
    if prior is not None and prior.module != module:
        raise ValueError(f"two packages declare {name}: {prior.module} and {module}")
    clash = next((d for d in _REGISTRY.values() if d.slug == policy.slug and d.name != name), None)
    if clash is not None:
        raise ValueError(f"{name} and {clash.name} both claim the slug {policy.slug!r}")

    domain = Domain(
        name=name,
        module=module,
        version=version,
        core_requires=core_requires,
        job=job,
        policy=policy,
        extends=extends,
        core_version=found,
    )
    _REGISTRY[name] = domain
    return domain


def _installed_version(distribution: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def loaded() -> list[Domain]:
    """Domains declared in this process so far."""
    return list(_REGISTRY.values())


def get(ref: str) -> Domain:
    """Look a domain up by distribution name or by slug."""
    if ref in _REGISTRY:
        return _REGISTRY[ref]
    found = next((d for d in _REGISTRY.values() if d.slug == ref), None)
    if found is None:
        known = sorted(d.slug for d in _REGISTRY.values())
        raise UnknownDomain(f"no domain {ref!r} is loaded; loaded: {known or '(none)'}")
    return found


def discover() -> list[dict[str, Any]]:
    """Every INSTALLED domain registered under the `rag.domains` entry point.

    An incompatible or broken domain is reported, not skipped: a run that quietly lost its field's
    grading rules would still produce an answer, and that answer would be wrong in a way nobody
    could see from the output.
    """
    from importlib import metadata

    rows: list[dict[str, Any]] = []
    for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
        row: dict[str, Any] = {
            "entry_point": ep.name,
            "module": ep.value,
            "distribution": ep.dist.name if ep.dist else None,
        }
        try:
            ep.load()  # importing the package runs its declare()
            module = ep.value.split(":")[0]
            domain = next((d for d in _REGISTRY.values() if d.module == module), None)
            if domain is None:
                row.update(
                    status="undeclared", reason="module imported but never called domains.declare()"
                )
            else:
                row.update(
                    status="ok",
                    name=domain.name,
                    slug=domain.slug,
                    version=domain.version,
                    core_requires=domain.core_requires,
                    extends=domain.extends,
                    job=domain.job,
                )
        except (IncompatibleCore, MissingDomain) as exc:
            row.update(status="incompatible", reason=str(exc))
        except Exception as exc:  # noqa: BLE001 - a broken domain must not take the engine down
            row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def load(ref: str) -> Domain:
    """Get a domain, importing installed domains first if it is not already declared."""
    try:
        return get(ref)
    except UnknownDomain:
        discover()
        return get(ref)


def derive(base: DomainPolicy, **changes: Any) -> DomainPolicy:
    """A policy built from another, for a sub-field that differs in only a few places."""
    return replace(base, **changes)


# -- the default domain --------------------------------------------------------------------
# The engine must answer a question with no domain installed, and it must behave then exactly as
# it did before domains existed. This policy is that behaviour, written down.

GENERIC = DomainPolicy(
    slug="generic",
    profile="",
    label="General science",
    scope="Peer-reviewed experimental and clinical literature, with no field-specific rules.",
    taxonomy=Taxonomy(
        (
            StudyDesign(
                "primary_human",
                "Primary human study",
                "new data from human participants or patients",
                primary=True,
                rank=70,
                human=True,
            ),
            StudyDesign(
                "primary_animal",
                "Primary animal study",
                "new data from live animals (in vivo)",
                primary=True,
                rank=50,
            ),
            StudyDesign(
                "in_vitro",
                "In vitro study",
                "cells, slices, cultures or organoids",
                primary=True,
                rank=40,
            ),
            StudyDesign(
                "computational_model",
                "Computational model",
                "simulation or theory without new biological data",
                rank=30,
            ),
            StudyDesign(
                "meta_analysis",
                "Meta-analysis",
                "pooled quantitative analysis of other studies",
                primary=True,
                rank=80,
                human=True,
            ),
            StudyDesign("review", "Review", "narrative summary of other papers", rank=20),
            StudyDesign("methods", "Methods paper", "a tool or protocol paper", rank=10),
            StudyDesign("unclear", "Unclear", "the text does not say", rank=0),
        )
    ),
    generalisation_rule=("Do not generalise animal or in-vitro results to humans."),
    reporting_rule=("Keep the species or population, the conditions, and any hedging."),
    gap_guidance=(
        "Look for a claim resting on one paper, an over-generalisation, missing "
        "contradicting or replication evidence, missing human (or animal) evidence, "
        "only old studies."
    ),
    critique_rules=(
        CritiqueRule(
            "no_human_evidence",
            frozenset({"primary_animal", "in_vitro"}),
            "animal or in-vitro evidence only.",
        ),
    ),
)


def active(ref: str | None) -> Domain:
    """The domain a run should use. No name means the generic policy, not a failure."""
    if not ref or ref == "generic":
        return _GENERIC_DOMAIN
    return load(ref)


_GENERIC_DOMAIN = Domain(
    name=NAME_PREFIX + "generic",
    module=__name__,
    version="0.1.0",
    core_requires=">=0.1.0",
    job="Answer a scientific question with no field-specific rules.",
    policy=GENERIC,
)
