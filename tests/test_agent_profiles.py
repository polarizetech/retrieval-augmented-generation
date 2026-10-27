from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.agent_profiles import route
from research_pipeline.prompts import Prompts


def _brief(root: Path, name: str, description: str, example: str = "") -> None:
    agents = root / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    (agents / f"{name}.md").write_text(
        f'---\nname: {name}\ndescription: "{description}"\n---\n'
        f"Positive routing examples:\n\n- {example}\n"
    )


def test_router_requires_a_specialist_anchor(tmp_path: Path) -> None:
    _brief(tmp_path, "researcher", "general literature researcher")
    _brief(
        tmp_path,
        "auditory-evoked-potentials",
        "scientific evidence electrophysiology researcher",
        "Compare ABR latency after cochlear stimulation",
    )
    ordinary = route("Does exercise lower blood pressure?", tmp_path)
    assert ordinary["specialist"] is None
    assert ordinary["domain"] == "cardiovascular"

    auditory = route("Compare ABR latency after cochlear stimulation", tmp_path)
    assert auditory["specialist"] == "auditory-evoked-potentials"
    assert auditory["domain"] == "neuroscience"
    assert auditory["profiles"][0]["sha256"]


def test_router_reports_missing_profile_repository(tmp_path: Path) -> None:
    result = route("question", tmp_path / "missing")
    assert result["available"] is False
    assert result["roles"] == ["researcher"]


def test_private_specialists_are_routed_and_guide_prompts(tmp_path: Path) -> None:
    public = tmp_path / "public"
    private = tmp_path / "private"
    _brief(public, "researcher", "literature evidence researcher")
    agents = private / "agents"
    agents.mkdir(parents=True)
    (agents / "signal-processing.md").write_text(
        "---\nname: signal-processing\n"
        "description: Use first for Fourier and RF intermodulation questions.\n"
        "extends: researcher\n---\n"
        "Positive routing examples:\n\n- Explain RF intermodulation from an amplifier.\n\n"
        "## Pipeline guidance\n\nSeparate predicted products from measured peaks.\n"
    )
    selected = route(
        "Explain RF intermodulation from an amplifier", public, extensions_root=private
    )
    assert selected["specialist"] == "signal-processing"
    assert selected["domain"] == ""
    assert selected["guidance"] == "Separate predicted products from measured peaks."
    assert selected["profiles"][-1]["repository"] == str(private)
    assert selected["profiles"][-1]["path"] == "agents/signal-processing.md"
    prompts = Prompts(agent_guidance=selected["guidance"])
    assert selected["guidance"] in prompts.plan_system
    assert selected["guidance"] in prompts.synth_system

    general = route("Does exercise lower blood pressure?", public, extensions_root=private)
    assert general["specialist"] is None
    assert general["guidance"] == ""

    (agents / "bone-conduction-auditory.md").write_text(
        "---\nname: bone-conduction-auditory\n"
        "description: Use first for bone-conduction hearing and cross-talk.\n"
        "extends: researcher\n---\n"
        "## Pipeline guidance\n\nCheck both cochlear pathways.\n"
    )
    bone = route(
        "How does bone conduction cross talk affect hearing?", public, extensions_root=private
    )
    assert bone["specialist"] == "bone-conduction-auditory"
    assert bone["guidance"] == "Check both cochlear pathways."
    assert (
        route("How does bone density change?", public, extensions_root=private)["specialist"]
        is None
    )


def test_missing_private_overlay_keeps_public_route(tmp_path: Path) -> None:
    _brief(tmp_path, "researcher", "literature evidence researcher")
    result = route("How do bees communicate?", tmp_path, extensions_root=tmp_path / "missing")
    assert result["available"] is True
    assert result["specialist"] is None
    assert result["extensions_commit"] == ""
    assert result["extensions_available"] is False
    assert "private overlay unavailable" in result["reason"]


def test_relative_mapping_routes_geometric_layouts_not_compass_widgets(tmp_path: Path) -> None:
    public = tmp_path / "public"
    private = tmp_path / "private"
    _brief(public, "researcher", "literature evidence researcher")
    agents = private / "agents"
    agents.mkdir(parents=True)
    (agents / "relative-mapping.md").write_text(
        "---\nname: relative-mapping\n"
        "description: Use first for relative spatial mapping with compass bearings, "
        "ruler measurements, scale drawings, and coordinate geometry.\n"
        "extends: researcher\n---\n"
        "Positive routing examples:\n\n"
        "- Map these landmarks from two bearings and a baseline.\n"
        "- Use a straightedge and drawing compass to place the vertices.\n\n"
        "Negative routing examples:\n\n"
        "- Build a compass widget for a web app.\n\n"
        "## Pipeline guidance\n\nDefine the frame and check closure.\n"
    )
    for question in (
        "I need an expert in relative mapping using a compass, ruler, and geometry",
        "Map these landmarks from two bearings and a baseline",
        "Use a straightedge and drawing compass to place the vertices",
    ):
        selected = route(question, public, extensions_root=private)
        assert selected["specialist"] == "relative-mapping"
        assert selected["guidance"] == "Define the frame and check closure."
        assert selected["profiles"][-1]["repository"] == str(private)
    for question in (
        "Build a compass widget for a web app",
        "Prove a theorem in Euclidean geometry",
    ):
        assert route(question, public, extensions_root=private)["specialist"] is None


def test_the_command_line_routes_like_the_research_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`research-pipeline ask` gets the same specialist briefs as pipeline__research_start."""
    from research_pipeline import __main__ as cli
    from research_pipeline import pipeline as pipeline_module

    _brief(tmp_path, "researcher", "general literature researcher")
    _brief(
        tmp_path,
        "auditory-evoked-potentials",
        "scientific evidence electrophysiology researcher",
        "Compare ABR latency after cochlear stimulation",
    )
    monkeypatch.setenv("PIPELINE_AGENT_PROFILES_DIR", str(tmp_path))
    monkeypatch.delenv("PIPELINE_AGENT_EXTENSIONS_DIR", raising=False)
    seen: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, settings: object, progress: object, **kw: object) -> None:
            seen.update(kw)

        async def run(self, question: str) -> dict[str, object]:
            return {"answer": "", "run_dir": "", "seconds": 0}

    monkeypatch.setattr(pipeline_module, "Pipeline", FakePipeline)
    monkeypatch.setattr(
        "sys.argv", ["research-pipeline", "ask", "Compare ABR latency after cochlear stimulation"]
    )
    assert cli.main() == 0
    route = seen["agent_route"]
    assert isinstance(route, dict)
    assert route["specialist"] == "auditory-evoked-potentials"
