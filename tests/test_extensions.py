"""The extension API: declaration checks, discovery, trigger words and the ask step runner."""

from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
from collections.abc import Awaitable, Callable, Iterator
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest
from mcp.types import Tool

from research_mcp import brief, extensions
from research_mcp.ask import Asker
from research_mcp.extensions import Context, StepInput, StepOutput, StepSpec, ToolSpec

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    extensions._REGISTRY.clear()
    yield
    extensions._REGISTRY.clear()


def _declare(**overrides: Any) -> extensions.Extension:
    fields: dict[str, Any] = {
        "name": "rag-ext-demo",
        "module": "rag_ext_demo",
        "version": "0.1.0",
        "core_requires": ">=0.3.0",
        "job": "demo",
        "prefix": "demo",
    }
    fields.update(overrides)
    return extensions.declare(**fields)


def _context() -> Context:
    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"name": name, "arguments": arguments}

    return Context(call_tool=call_tool, has_tool=lambda name: False, data_dir=ROOT, config={})


async def retrieve(query: str, limit: int, max_chars: int) -> dict[str, Any]:
    return {
        "retrieval": "hybrid",
        "n": 1,
        "excluded": [],
        "results": [
            {
                "evidence_id": "W1#p0.abcdef12",
                "text": "Pink noise has equal energy per octave.",
                "title": "Colours of noise",
                "year": 2017,
                "doi": "10.1/x",
            }
        ],
    }


def _asker(
    tmp_path: Path,
    steps: tuple[tuple[extensions.Extension, StepSpec, Context], ...] = (),
    local: str = "",
) -> Asker:
    public, private = tmp_path / "public", tmp_path / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    (tmp_path / "routing.toml").write_text(local)

    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {}

    return Asker(
        retrieve=retrieve,
        call_tool=call_tool,
        has_tool=lambda n: False,
        rules_paths=(ROOT / "config" / "routing.default.toml", tmp_path / "routing.toml"),
        profiles_dir=public,
        extensions_dir=private,
        steps=steps,
    )


async def _world(step_input: StepInput) -> StepOutput:
    return StepOutput(result={"hello": "world"})


async def broken(step_input: StepInput) -> StepOutput:
    raise RuntimeError("nope")


async def clearing(step_input: StepInput) -> StepOutput:
    step_input.results.clear()
    return StepOutput(result={})


async def _probe(step_input: StepInput) -> StepOutput:
    return StepOutput(result={"saw_evidence": "evidence" in step_input.results})


async def _skipping(step_input: StepInput) -> StepOutput:
    return StepOutput(
        result={},
        skip_retrieval="a demo request",
        limits="Demo numbers are data, not a finding.",
    )


async def _ruling(step_input: StepInput) -> StepOutput:
    return StepOutput(result={}, brief={"rules": ["be kind"]})


def _hello_step(
    run: Callable[[StepInput], Awaitable[StepOutput | None]] = _world,
) -> StepSpec:
    return StepSpec(name="hello", run=run, triggers=("demo word",))


def test_declare_rejects_a_name_without_the_prefix() -> None:
    with pytest.raises(ValueError, match="must start with 'rag-ext-'"):
        _declare(name="my-ext")


@pytest.mark.parametrize("prefix", sorted(extensions.RESERVED_PREFIXES))
def test_declare_rejects_a_reserved_prefix(prefix: str) -> None:
    with pytest.raises(ValueError, match="reserved"):
        _declare(prefix=prefix)


def test_declare_rejects_an_incompatible_core() -> None:
    with pytest.raises(extensions.IncompatibleCore):
        _declare(core_requires=">=99.0")


def test_two_extensions_cannot_share_a_prefix() -> None:
    _declare()
    with pytest.raises(ValueError, match="both claim the prefix"):
        _declare(name="rag-ext-other")


def test_a_tool_must_carry_its_prefix() -> None:
    tool = ToolSpec(
        tool=Tool(name="other__x", inputSchema={"type": "object"}),
        handler=lambda arguments, context: {},
        label="other x",
    )
    with pytest.raises(ValueError, match="must start with 'demo__'"):
        _declare(tools=(tool,))


def test_discover_reports_a_broken_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self: EntryPoint) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(EntryPoint, "load", boom)
    entry = EntryPoint(name="demo", value="rag_ext_demo", group=extensions.ENTRY_POINT_GROUP)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **_: [entry])
    rows = extensions.discover()
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert "boom" in rows[0]["reason"]


def test_discover_reports_an_undeclared_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EntryPoint, "load", lambda self: None)
    entry = EntryPoint(name="demo", value="rag_ext_demo", group=extensions.ENTRY_POINT_GROUP)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **_: [entry])
    rows = extensions.discover()
    assert len(rows) == 1
    assert rows[0]["status"] == "undeclared"


def test_step_runs_only_on_its_trigger_words(tmp_path: Path) -> None:
    ext = _declare(steps=(_hello_step(),))
    steps = ((ext, ext.steps[0], _context()),)
    out = asyncio.run(_asker(tmp_path, steps).ask("what is a demo word?"))
    assert out["results"]["extensions"]["rag-ext-demo"]["hello"] == {"hello": "world"}
    assert "demo: ran" in out["progress"]

    other = asyncio.run(_asker(tmp_path, steps).ask("unrelated question"))
    assert "extensions" not in other["results"]


def test_routing_toml_adds_trigger_words(tmp_path: Path) -> None:
    ext = _declare(steps=(_hello_step(),))
    steps = ((ext, ext.steps[0], _context()),)
    local = '[extensions."rag-ext-demo"]\nhello = ["extra phrase"]\n'
    out = asyncio.run(_asker(tmp_path, steps, local).ask("an extra phrase here"))
    assert out["results"]["extensions"]["rag-ext-demo"]["hello"] == {"hello": "world"}


def test_a_failing_step_does_not_fail_ask(tmp_path: Path) -> None:
    ext = _declare(steps=(_hello_step(broken),))
    steps = ((ext, ext.steps[0], _context()),)
    plain = asyncio.run(_asker(tmp_path).ask("What is a demo word?"))
    out = asyncio.run(_asker(tmp_path, steps).ask("What is a demo word?"))
    assert out["status"] == plain["status"]
    assert out["results"]["extensions"]["rag-ext-demo"]["hello"] == {"error": "RuntimeError: nope"}
    assert "demo/hello: failed (RuntimeError: nope)" in out["progress"]


def test_a_step_cannot_change_other_results(tmp_path: Path) -> None:
    ext = _declare(steps=(_hello_step(clearing),))
    steps = ((ext, ext.steps[0], _context()),)
    out = asyncio.run(_asker(tmp_path, steps).ask("What is a demo word?"))
    evidence = out["results"]["evidence"]
    assert evidence["results"][0]["evidence_id"] == "W1#p0.abcdef12"
    assert evidence["n"] == 1


def test_the_template_declares() -> None:
    path = ROOT / "docs" / "extension-template" / "rag_ext_example" / "__init__.py"
    spec = importlib.util.spec_from_file_location("rag_ext_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ext = extensions.get("rag-ext-example")
    assert ext.prefix == "example"
    assert [t.tool.name for t in ext.tools] == ["example__hello"]


def test_declare_rejects_an_unknown_stage() -> None:
    step = StepSpec(name="hello", run=_world, triggers=("demo word",), stage="later")
    with pytest.raises(ValueError, match="unknown stage"):
        _declare(steps=(step,))


def test_a_shared_prefix_tool_is_allowed() -> None:
    tool = ToolSpec(
        tool=Tool(name="math__demo", inputSchema={"type": "object"}),
        handler=lambda arguments, context: {},
        label="math demo",
    )
    ext = _declare(tools=(tool,))
    assert ext.tools[0].tool.name == "math__demo"


def test_a_shared_prefix_tool_cannot_take_a_core_name() -> None:
    tool = ToolSpec(
        tool=Tool(name="math__lookup", inputSchema={"type": "object"}),
        handler=lambda arguments, context: {},
        label="math lookup",
    )
    with pytest.raises(ValueError, match="core tool"):
        _declare(tools=(tool,))


def test_declare_rejects_a_core_brief_key() -> None:
    with pytest.raises(ValueError, match="core brief key"):
        _declare(brief_key="passages")


def test_two_extensions_cannot_share_a_brief_key() -> None:
    _declare(brief_key="demo_rules")
    with pytest.raises(ValueError, match="both claim the brief key"):
        _declare(name="rag-ext-other", prefix="other", brief_key="demo_rules")


def test_an_early_step_runs_before_retrieval(tmp_path: Path) -> None:
    step = StepSpec(name="hello", run=_probe, triggers=("demo word",), stage="early")
    ext = _declare(steps=(step,))
    steps = ((ext, ext.steps[0], _context()),)
    out = asyncio.run(_asker(tmp_path, steps).ask("What is a demo word?"))
    assert out["results"]["extensions"]["rag-ext-demo"]["hello"] == {"saw_evidence": False}


def test_an_after_step_sees_the_evidence(tmp_path: Path) -> None:
    ext = _declare(steps=(_hello_step(_probe),))
    steps = ((ext, ext.steps[0], _context()),)
    out = asyncio.run(_asker(tmp_path, steps).ask("What is a demo word?"))
    assert out["results"]["extensions"]["rag-ext-demo"]["hello"] == {"saw_evidence": True}


def test_a_before_retrieval_step_can_skip_retrieval(tmp_path: Path) -> None:
    step = StepSpec(name="hello", run=_skipping, triggers=("demo word",), stage="before_retrieval")
    ext = _declare(steps=(step,))
    steps = ((ext, ext.steps[0], _context()),)
    out = asyncio.run(_asker(tmp_path, steps).ask("What is a demo word?"))
    assert "evidence" not in out["results"]
    assert "Retrieval: skipped (a demo request)" in out["progress"]
    assert out["summary"]["results"][-1] == "Demo numbers are data, not a finding."


def test_a_brief_key_puts_the_result_in_the_brief(tmp_path: Path) -> None:
    step = StepSpec(name="hello", run=_ruling, triggers=("demo word",), stage="early")
    ext = _declare(
        brief_key="demo_rules",
        brief_note="Follow `brief.demo_rules`. ",
        steps=(step,),
    )
    steps = ((ext, ext.steps[0], _context()),)
    out = asyncio.run(_asker(tmp_path, steps).ask("What is a demo word?"))
    short = brief.compact(out)
    assert short["brief"]["demo_rules"] == {"rules": ["be kind"]}
    assert short["answer_format"].startswith("Follow `brief.demo_rules`. ")
    assert "demo_rules" not in short["brief"].get("tool_results", {})
