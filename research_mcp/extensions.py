"""Extensions: what an installed package adds to the gateway, and nothing it may change.

The one rule: an extension ADDS tools and steps. It never changes evidence, retrieval,
verification, grading, coverage or another step's results. Its output lives only under its own
key, `results["extensions"][<extension name>]`. This is the same plugin pattern as domains
(research_pipeline/domains.py): a distribution-name prefix, import-time declaration, a version
gate, entry-point discovery, and no core-side list of extensions. Scientific software "may require
scripting or plugin support for extensions" (Gewaltig and Cannon 2014, DOI
10.1371/journal.pcbi.1003376). See docs/EXTENSIONS.md.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp.types import Tool

from research_mcp.routing import _has
from research_pipeline import domains

ENTRY_POINT_GROUP = "rag.extensions"
NAME_PREFIX = "rag-ext-"
_PREFIX = re.compile(r"[a-z][a-z0-9_]*")
RESERVED_PREFIXES = frozenset(
    {
        "ask",
        "gateway_status",
        "web",
        "rag",
        "math",
        "rules",
        "session",
        "hub",
        "sky",
        "polarize",
        "papers",
        "pipeline",
        "datasets",
        "patents",
        "repos",
        "research",
    }
)


class ExtensionError(Exception):
    """Base class for every problem with an extension declaration."""


class IncompatibleCore(ExtensionError):
    """The installed engine does not satisfy an extension's declared version range."""


class UnknownExtension(ExtensionError):
    """An extension was asked for by name and is not registered."""


@dataclass(frozen=True)
class Context:
    """What a tool or step may use: the gateway's tools, its data folder, its config section."""

    call_tool: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
    has_tool: Callable[[str], bool]
    data_dir: Path
    config: dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    """One MCP tool an extension adds. The handler may be sync or async."""

    tool: Tool
    handler: Callable[[dict[str, Any], Context], Any]
    label: str
    component: str = "extension"


@dataclass(frozen=True)
class StepInput:
    """What an ask step is given: the request and a deep copy of the results so far."""

    question: str
    mode: str
    project: str
    repo: str
    results: dict[str, Any]
    context: Context


@dataclass
class StepOutput:
    """What an ask step returns. Its `result` is stored under its own key; the lists are added to
    the reply's progress, summary and relay, and nothing else is changed."""

    result: dict[str, Any]
    progress: list[str] = field(default_factory=list)
    found: list[str] = field(default_factory=list)
    next: list[str] = field(default_factory=list)
    for_you: list[str] = field(default_factory=list)
    relay: str = ""


@dataclass(frozen=True)
class StepSpec:
    """An ask step. It runs when the request contains one of its triggers (or always, if set)."""

    name: str
    run: Callable[[StepInput], Awaitable[StepOutput | None]]
    triggers: tuple[str, ...] = ()
    always: bool = False


@dataclass(frozen=True)
class Extension:
    """What an extension package says about itself. Declared once, in its `__init__.py`."""

    name: str  # distribution-style, "rag-ext-<prefix>"
    module: str
    version: str
    core_requires: str
    job: str  # one line: the one thing this adds
    prefix: str  # its tools are named <prefix>__<tool>
    tools: tuple[ToolSpec, ...]
    steps: tuple[StepSpec, ...]
    instructions: str
    core_version: str


_REGISTRY: dict[str, Extension] = {}


def declare(
    *,
    name: str,
    module: str,
    version: str,
    core_requires: str,
    job: str,
    prefix: str,
    tools: Iterable[ToolSpec] = (),
    steps: Iterable[StepSpec] = (),
    instructions: str = "",
) -> Extension:
    """Declare an extension, check it against the engine and its own names, and register it."""
    if not name.startswith(NAME_PREFIX):
        raise ValueError(f"extension name {name!r} must start with {NAME_PREFIX!r}")
    if not job.strip():
        raise ValueError(f"{name}: declare the one job this extension adds")

    domains.parse_version(version)
    try:
        found = domains.require(name, core_requires)
    except domains.IncompatibleCore as exc:
        raise IncompatibleCore(str(exc)) from exc

    if not _PREFIX.fullmatch(prefix):
        raise ValueError(
            f"{name}: prefix {prefix!r} must be lowercase letters, digits and underscores, "
            "starting with a letter"
        )
    if prefix in RESERVED_PREFIXES:
        raise ValueError(f"{name}: prefix {prefix!r} is reserved for the core")

    tool_specs = tuple(tools)
    for spec in tool_specs:
        if not spec.tool.name.startswith(f"{prefix}__"):
            raise ValueError(f"{name}: tool {spec.tool.name!r} must start with '{prefix}__'")

    step_specs = tuple(steps)
    step_names = [s.name for s in step_specs]
    if len(set(step_names)) != len(step_names):
        raise ValueError(f"{name}: step names must be unique")

    prior = _REGISTRY.get(name)
    if prior is not None and prior.module != module:
        raise ValueError(f"two packages declare {name}: {prior.module} and {module}")
    clash = next((e for e in _REGISTRY.values() if e.prefix == prefix and e.name != name), None)
    if clash is not None:
        raise ValueError(f"{name} and {clash.name} both claim the prefix {prefix!r}")

    extension = Extension(
        name=name,
        module=module,
        version=version,
        core_requires=core_requires,
        job=job,
        prefix=prefix,
        tools=tool_specs,
        steps=step_specs,
        instructions=instructions,
        core_version=found,
    )
    _REGISTRY[name] = extension
    return extension


def loaded() -> list[Extension]:
    """Extensions declared in this process so far."""
    return list(_REGISTRY.values())


def get(name: str) -> Extension:
    """Look an extension up by distribution name."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    raise UnknownExtension(
        f"no extension {name!r} is loaded; loaded: {sorted(_REGISTRY) or '(none)'}"
    )


def discover() -> list[dict[str, Any]]:
    """Every INSTALLED extension registered under the `rag.extensions` entry point.

    A broken or incompatible extension is reported, not skipped, and the gateway keeps running
    without it. The statuses are the same as research_pipeline.domains.discover(): ok, undeclared,
    incompatible and error.
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
            ext = next((e for e in _REGISTRY.values() if e.module == module), None)
            if ext is None:
                row.update(
                    status="undeclared",
                    reason="module imported but never called extensions.declare()",
                )
            else:
                row.update(
                    status="ok",
                    name=ext.name,
                    version=ext.version,
                    prefix=ext.prefix,
                    job=ext.job,
                    tools=[t.tool.name for t in ext.tools],
                    steps=[s.name for s in ext.steps],
                )
        except IncompatibleCore as exc:
            row.update(status="incompatible", reason=str(exc))
        except Exception as exc:  # noqa: BLE001 - a broken extension must not take the gateway down
            row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def matches(step: StepSpec, question: str, extra_words: Iterable[str]) -> bool:
    """True when the step runs for this question: always, or one of its trigger words (its own
    and any the operator's routing file adds) occurs in the question as whole words."""
    if step.always:
        return True
    low = question.lower()
    return any(_has(low, word.lower()) for word in (*step.triggers, *extra_words))
