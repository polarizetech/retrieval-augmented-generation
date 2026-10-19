from __future__ import annotations

from pathlib import Path

from research_pipeline.agent_profiles import route


def _brief(root: Path, name: str, description: str, example: str = "") -> None:
    agents = root / "agents"
    agents.mkdir(exist_ok=True)
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
