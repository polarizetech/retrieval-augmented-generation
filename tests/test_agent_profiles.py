from __future__ import annotations

from pathlib import Path

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


def test_missing_private_overlay_keeps_public_route(tmp_path: Path) -> None:
    _brief(tmp_path, "researcher", "literature evidence researcher")
    result = route("How do bees communicate?", tmp_path, extensions_root=tmp_path / "missing")
    assert result["available"] is True
    assert result["specialist"] is None
    assert result["extensions_commit"] == ""
    assert result["extensions_available"] is False
    assert "private overlay unavailable" in result["reason"]
