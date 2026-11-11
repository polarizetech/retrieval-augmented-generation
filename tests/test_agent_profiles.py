from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.agent_profiles import catalogue, route
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


def test_plasma_toroids_routes_spheromaks_and_frcs_not_grid_cell_tori(tmp_path: Path) -> None:
    public = tmp_path / "public"
    private = tmp_path / "private"
    _brief(public, "researcher", "literature evidence researcher")
    agents = private / "agents"
    agents.mkdir(parents=True)
    (agents / "plasma-toroids.md").write_text(
        "---\nname: plasma-toroids\n"
        "description: Use first for compact toroids: spheromaks, field-reversed configurations "
        "(FRCs), and co- and counter-helicity spheromak merging.\n"
        "extends: researcher\n---\n"
        "Positive routing examples:\n\n"
        "- What decides whether two merging spheromaks become an FRC?\n\n"
        "Negative routing examples:\n\n"
        "- Design a tokamak divertor.\n\n"
        "## Pipeline guidance\n\nName the configuration first.\n"
    )
    for question in (
        "What decides whether two counter-helicity spheromaks merge into an FRC?",
        "Why is the field-reversed configuration tilt mode stable in experiments?",
        "Write down the Taylor state of a spheromak",
    ):
        selected = route(question, public, extensions_root=private)
        assert selected["specialist"] == "plasma-toroids"
        assert selected["guidance"] == "Name the configuration first."
    for question in (
        "Do grid cells lie on a toroidal manifold?",
        "Design a tokamak divertor",
    ):
        assert route(question, public, extensions_root=private)["specialist"] is None


def test_macgyver_routes_diy_rigs_not_plasma_physics_or_mixing_maths(tmp_path: Path) -> None:
    public = tmp_path / "public"
    private = tmp_path / "private"
    _brief(public, "researcher", "literature evidence researcher")
    agents = private / "agents"
    agents.mkdir(parents=True)
    (agents / "macgyver.md").write_text(
        "---\nname: macgyver\n"
        "description: Use first for DIY and home-built experiment rigs: household parts, Arduino "
        "boards, sound cards, coils and a plasma ball as instruments, signal mixers and sensors.\n"
        "extends: researcher\n---\n"
        "Positive routing examples:\n\n"
        "- How can I turn a battery plasma ball into a signal mixer with a stereo input?\n\n"
        "Negative routing examples:\n\n"
        "- Derive the intermodulation products of a cubic nonlinearity.\n\n"
        "## Pipeline guidance\n\nStart from the question the rig must answer.\n"
    )
    (agents / "plasma-toroids.md").write_text(
        "---\nname: plasma-toroids\n"
        "description: Compact toroids: spheromaks and field-reversed configurations.\n"
        "extends: researcher\n---\n"
    )
    (agents / "signal-processing.md").write_text(
        "---\nname: signal-processing\n"
        "description: Fourier analysis, nonlinear frequency mixing and intermodulation products.\n"
        "extends: researcher\n---\n"
    )
    for question in (
        "How do I make a DIY signal mixer from a battery plasma ball?",
        "Build an Arduino magnetometer from household copper wire",
        "Tape a photodiode to the plasma globe and record it on line-in",
        "Turn a regular battery powered plasma ball into a signal mixer with a stereo input",
    ):
        selected = route(question, public, extensions_root=private)
        assert selected["specialist"] == "macgyver"
        assert selected["guidance"] == "Start from the question the rig must answer."
    assert (
        route("Why do two spheromaks merge into an FRC?", public, extensions_root=private)[
            "specialist"
        ]
        == "plasma-toroids"
    )
    assert (
        route(
            "Derive the intermodulation products of two tones in a cubic mixer",
            public,
            extensions_root=private,
        )["specialist"]
        == "signal-processing"
    )


def test_musical_acoustics_routes_instruments_not_code_strings_or_spectra(tmp_path: Path) -> None:
    public = tmp_path / "public"
    private = tmp_path / "private"
    _brief(public, "researcher", "literature evidence researcher")
    agents = private / "agents"
    agents.mkdir(parents=True)
    (agents / "musical-acoustics.md").write_text(
        "---\nname: musical-acoustics\n"
        "description: Use first for how instruments make sound: vibrating strings, standing "
        "waves and harmonics, plucking point, fretting, natural harmonics, decay, pitch glide.\n"
        "extends: researcher\n---\n"
        "Positive routing examples:\n\n"
        "- Why does a guitar string ring out, and what sets how long it sustains?\n\n"
        "Negative routing examples:\n\n"
        "- Is this FFT peak leakage or a real partial?\n\n"
        "## Pipeline guidance\n\nName the resonator, excitation, constraints and loss paths.\n"
    )
    (agents / "signal-processing.md").write_text(
        "---\nname: signal-processing\n"
        "description: Fourier analysis, spectral leakage, harmonics, FFT.\n"
        "extends: researcher\n---\n"
    )
    for question in (
        "Why does a guitar string ring out, and what sets how long it sustains?",
        "Why does the pitch start sharp when a string is plucked hard?",
        "Why does a vibrating string under more tension sound higher?",
        "How does the soundboard change the decay of a note?",
        "Why does fretting a guitar string change the note?",
    ):
        selected = route(question, public, extensions_root=private)
        assert selected["specialist"] == "musical-acoustics", question
        assert selected["guidance"] == "Name the resonator, excitation, constraints and loss paths."
    for question in (
        "Parse this string and split it on commas",
        "Why does the build fail on a format string?",
        "Raise the pitch of the sales deck",
    ):
        assert route(question, public, extensions_root=private)["specialist"] is None, question
    assert (
        route("Is this FFT peak spectral leakage?", public, extensions_root=private)["specialist"]
        == "signal-processing"
    )


def test_classical_mechanics_routes_motion_not_ui_signals_or_optimisers(tmp_path: Path) -> None:
    public = tmp_path / "public"
    private = tmp_path / "private"
    _brief(public, "researcher", "literature evidence researcher")
    agents = private / "agents"
    agents.mkdir(parents=True)
    (agents / "classical-mechanics.md").write_text(
        "---\nname: classical-mechanics\n"
        "description: Use first for motion, forces and energy: work and energy (conservation, "
        "transfer and dissipation), ramps and inclined planes (sliding, rolling), friction "
        "(static, kinetic, stick-slip), drag (linear and quadratic, terminal velocity).\n"
        "extends: researcher\n---\n"
        "Positive routing examples:\n\n"
        "- How much energy does a ball lose to air drag on the way up and down?\n\n"
        "Negative routing examples:\n\n"
        "- Implement a symplectic integrator in JAX.\n\n"
        "## Pipeline guidance\n\nDraw the free-body diagram first.\n"
    )
    (agents / "signal-processing.md").write_text(
        "---\nname: signal-processing\n"
        "description: Fourier analysis, spectral leakage, ramp and sawtooth signals, FFT.\n"
        "extends: researcher\n---\n"
    )
    (agents / "plasma-toroids.md").write_text(
        "---\nname: plasma-toroids\n"
        "description: Compact toroids: spheromaks and field-reversed configurations.\n"
        "extends: researcher\n---\n"
    )
    for question in (
        "How much energy does a ball lose to air drag on the way up and down?",
        "Why does a cylinder rolling down a ramp arrive later than a sliding block?",
        "Measure the coefficient of friction with a phone on an inclined plane",
        "What is the terminal velocity of a falling coffee filter?",
        "What causes stick-slip squeaking in a sliding block?",
    ):
        selected = route(question, public, extensions_root=private)
        assert selected["specialist"] == "classical-mechanics", question
        assert selected["guidance"] == "Draw the free-body diagram first."
    for question in (
        "Implement drag and drop for the file list",
        "Add momentum to the optimiser's energy function",
        "Open a terminal and run the tests",
    ):
        assert route(question, public, extensions_root=private)["specialist"] is None, question
    assert (
        route("Generate a ramp signal and compute its FFT", public, extensions_root=private)[
            "specialist"
        ]
        == "signal-processing"
    )
    assert (
        route("Why do two spheromaks merge into an FRC?", public, extensions_root=private)[
            "specialist"
        ]
        == "plasma-toroids"
    )


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


def test_a_reference_brief_is_read_from_its_source_repository(tmp_path: Path) -> None:
    public, private, study = tmp_path / "public", tmp_path / "private", tmp_path / "study"
    _brief(public, "researcher", "literature evidence researcher")
    (study / ".git").mkdir(parents=True)
    (study / "agents").mkdir()
    (study / "agents" / "aramaic-translator.md").write_text(
        "---\nname: aramaic-translator\n"
        "description: Use first for Aramaic translation into English with every meaning.\n"
        "extends: researcher\n---\n"
        "Positive routing examples:\n\n- Translate this Aramaic phrase into English.\n\n"
        "## Pipeline guidance\n\nReport every plausible sense.\n"
    )
    (private / "agents").mkdir(parents=True)
    (private / "agents" / "aramaic-translator.md").write_text(
        "---\nname: aramaic-translator\nextends: researcher\n"
        "source: ../study/agents/aramaic-translator.md\n---\nMaintained in the study.\n"
    )
    result = route("Translate this Aramaic phrase into English", public, extensions_root=private)
    assert result["specialist"] == "aramaic-translator"
    profile = next(p for p in result["profiles"] if p["name"] == "aramaic-translator")
    assert profile["repository"] == str(study.resolve())
    assert profile["path"] == "agents/aramaic-translator.md"


def test_a_reference_with_a_missing_source_is_skipped(tmp_path: Path) -> None:
    public, private = tmp_path / "public", tmp_path / "private"
    _brief(public, "researcher", "literature evidence researcher")
    (private / "agents").mkdir(parents=True)
    (private / "agents" / "aramaic-translator.md").write_text(
        "---\nname: aramaic-translator\nextends: researcher\nsource: ../absent/a.md\n---\n"
    )
    result = route("Translate this Aramaic phrase", public, extensions_root=private)
    assert result["specialist"] is None


def test_a_generic_part_of_a_profile_name_does_not_anchor_it(tmp_path: Path) -> None:
    # Profiles without curated anchor terms are anchored by every distinctive part of their name.
    _brief(tmp_path, "researcher", "general literature researcher")
    for name, words in (
        ("antenna-array-engineering", "antenna array beamforming element patterns coupling noise"),
        ("noise-engineering", "noise colour spectral exponent coupling array"),
    ):
        agents = tmp_path / "agents"
        (agents / f"{name}.md").write_text(
            f"---\nname: {name}\ndescription: Use for {words}.\nextends: researcher\n---\n"
            f"Positive routing examples:\n\n- {words}?\n"
        )
    question = "Ensemble with array-enhanced stochastic resonance and mean-field coupling of noise"
    assert route(question, tmp_path)["specialist"] == "noise-engineering"
    no_noise = route("Array-enhanced resonance with mean-field coupling", tmp_path)
    assert no_noise["specialist"] is None
    antenna = route("Embedded element patterns in phased antenna arrays", tmp_path)
    assert antenna["specialist"] == "antenna-array-engineering"


def test_a_reference_brief_is_anchored_by_its_curated_terms(tmp_path: Path) -> None:
    public, private, study = tmp_path / "public", tmp_path / "private", tmp_path / "study"
    _brief(public, "researcher", "literature evidence researcher")
    (study / ".git").mkdir(parents=True)
    (study / "agents").mkdir()
    (study / "agents" / "nonlinear-signal-mixing.md").write_text(
        "---\nname: nonlinear-signal-mixing\n"
        "description: Use first for keyed polarity streams, Righetti adaptive phase oscillators, "
        "Adler lock ranges and Arnold tongues.\nextends: researcher\n---\n"
    )
    (private / "agents").mkdir(parents=True)
    (private / "agents" / "nonlinear-signal-mixing.md").write_text(
        "---\nname: nonlinear-signal-mixing\nextends: researcher\n"
        "source: ../study/agents/nonlinear-signal-mixing.md\n---\n"
    )
    question = "Derive the Adler lock range for a Righetti loop driven by the keyed polarity stream"
    result = route(question, public, extensions_root=private)
    assert result["specialist"] == "nonlinear-signal-mixing"


def _specialist(root: Path, name: str, description: str, example: str, guidance: str) -> None:
    agents = root / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    (agents / f"{name}.md").write_text(
        f"---\nname: {name}\ndescription: Use first for {description} in {{{{name}}}}. More.\n"
        f"extends: researcher\n---\n\nPositive routing examples:\n\n1. {example}\n\n"
        f"## Pipeline guidance\n\n{guidance}\n"
    )


@pytest.fixture
def three_fields(tmp_path: Path) -> tuple[Path, Path]:
    public, private = tmp_path / "public", tmp_path / "private"
    _brief(public, "researcher", "literature evidence researcher")
    _specialist(
        private,
        "classical-mechanics",
        "friction, drag and energy of moving objects",
        "How much energy does a block lose to friction and drag?",
        "Draw the free-body diagram first.",
    )
    _specialist(
        private,
        "plasma-toroids",
        "spheromaks and plasma energy",
        "How much energy does a spheromak lose?",
        "Name the confinement regime.",
    )
    _specialist(
        private,
        "signal-processing",
        "Fourier analysis",
        "Take the FFT of a recording",
        "State the sample rate.",
    )
    return public, private


def test_up_to_three_specialists_answer_a_question_that_spans_fields(
    three_fields: tuple[Path, Path],
) -> None:
    public, private = three_fields
    both = route(
        "How much energy does a spheromak lose to drag and friction?",
        public,
        extensions_root=private,
    )
    assert both["specialists"] == ["classical-mechanics", "plasma-toroids"]
    assert both["specialist"] == "classical-mechanics"
    assert both["guidances"]["plasma-toroids"] == "Name the confinement regime."
    # Several specialists' guidance reaches the prompts labelled by profile.
    assert "[classical-mechanics] Draw the free-body" in both["guidance"]
    assert "[plasma-toroids] Name the confinement" in both["guidance"]
    assert {p["name"] for p in both["profiles"]} >= {"classical-mechanics", "plasma-toroids"}
    prompts = Prompts(agent_guidance=both["guidance"], agent_specialists=both["specialists"])
    assert "these selected specialists'" in prompts.plan_system

    # One field: its guidance unlabelled, as before.
    one = route("Take the FFT of a recording", public, extensions_root=private)
    assert one["specialists"] == ["signal-processing"]
    assert one["guidance"] == "State the sample rate."


def test_a_named_profile_is_recruited_and_unknown_names_are_reported(
    three_fields: tuple[Path, Path],
) -> None:
    public, private = three_fields
    named = route(
        "Ask signal processing and @plasma-toroids: why does friction heat a block?",
        public,
        extensions_root=private,
    )
    # Named profiles come first, in the order named, then the scored one; never more than three.
    assert named["specialists"] == ["signal-processing", "plasma-toroids", "classical-mechanics"]
    assert "named in the question (signal-processing)" in named["reason"]

    asked = route(
        "Why does friction heat a block?",
        public,
        extensions_root=private,
        requested=["plasma-toroids", "no-such-profile", "researcher"],
    )
    assert asked["specialists"][0] == "plasma-toroids"
    assert asked["unknown_requested"] == ["no-such-profile", "researcher"]
    # A profile's name inside a longer word is not a request for it.
    assert (
        route("Read the signal-processingx notes", public, extensions_root=private)["specialists"]
        == []
    )


def test_the_catalogue_lists_specialists_with_what_they_are_for(
    three_fields: tuple[Path, Path],
) -> None:
    public, private = three_fields
    rows = catalogue(public, private)
    assert [r["name"] for r in rows] == [
        "classical-mechanics",
        "plasma-toroids",
        "signal-processing",
    ]
    assert rows[0] == {
        "name": "classical-mechanics",
        "source": "private",
        "for": "Friction, drag and energy of moving objects",
    }
