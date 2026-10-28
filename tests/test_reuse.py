"""The build-vs-reuse runner (docs/reuse-register.md): scanning, register checks and the report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_mcp import reuse


@pytest.fixture
def catalogue() -> dict:
    return reuse.load_catalogue()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "analysis").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "node_modules" / "x").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["scipy>=1.10", "mne-bids"]\n'
        '[dependency-groups]\ndev = ["pytest"]\n'
    )
    (root / "requirements.txt").write_text("# deps\nstatsmodels==0.14\n")
    (root / "package.json").write_text(json.dumps({"dependencies": {"react": "1"}}))
    (root / "analysis" / "power.py").write_text("import numpy as np\nfrom mne import io\n")
    (root / "analysis" / "pac.ts").write_text("import { x } from './local'\n")
    (root / "analysis" / "periph.m").write_text("amt_start;\n")
    (root / "node_modules" / "x" / "a.py").write_text("import brian2\n")
    (root / "tests" / "test_power.py").write_text("import mne\n# compares ours with MNE\n")
    (root / "reuse.toml").write_text(
        """
[[custom]]
name = "Band power"
path = "analysis/power.py"
does = "band power"
state = "tandem"
alternative = "mne"
check = "tests/test_power.py"

[[custom]]
name = "FDR"
path = "analysis/power.py"
state = "candidate"
alternative = "statsmodels"

[[custom]]
name = "Gone"
path = "analysis/missing.py"
state = "none-found"

[[custom]]
name = "Odd one"
path = "analysis/pac.ts"
state = "candidate"
alternative = "SomethingNew"
"""
    )
    return root


def test_catalogue_parses_with_papers(catalogue: dict) -> None:
    assert "mne" in catalogue
    for key, entry in catalogue.items():
        assert entry.get("name"), key
        assert entry.get("paper"), key
        assert entry.get("review") in {"paper", "joss", "registry", "none"}, key


def test_scan_finds_declared_and_imported_packages(repo: Path, catalogue: dict) -> None:
    used = reuse.scan_packages(repo, catalogue)
    assert set(used) == {"scipy", "mne-bids", "statsmodels", "numpy", "mne", "amt"}
    assert used["scipy"]["declared"] == ["pyproject.toml"]
    assert used["mne"]["files"] == 2
    assert "brian2" not in used  # node_modules is skipped


def test_register_checks(repo: Path, catalogue: dict) -> None:
    result = reuse.scan_repo("org/repo", repo, "local", catalogue, repo / "home")
    by_name = {c["name"]: c for c in result["custom"]}
    assert by_name["Band power"]["issues"] == []
    assert by_name["Band power"]["alternative_paper"].startswith("Gramfort")
    assert by_name["FDR"]["easy_swap"] is True
    assert by_name["Gone"]["issues"] == ["path not found"]
    assert "alternative not in the catalogue" in by_name["Odd one"]["issues"][0]
    assert result["register"] == "repo"
    assert result["counts"]["reused_methods"] == 4  # mne, mne-bids, statsmodels, amt
    assert result["counts"]["reused_foundation"] == 2


def test_tandem_check_must_mention_the_alternative(tmp_path: Path, catalogue: dict) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "t.py").write_text("assert True\n")
    entry = {
        "name": "a",
        "path": "a.py",
        "state": "tandem",
        "alternative": "ordpy",
        "check": "t.py",
    }
    out = reuse.check_entry(entry, tmp_path, {}, catalogue)
    assert out["issues"] == ["check does not mention the alternative"]


def test_seed_register_is_used_when_the_repo_has_none(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "registers").mkdir(parents=True)
    (home / "registers" / "org__r.toml").write_text('[[custom]]\nname = "x"\n')
    entries, source = reuse.load_register(tmp_path, "org/r", home)
    assert source == "seed"
    assert entries == [{"name": "x"}]


def test_report_files_and_history(repo: Path, catalogue: dict, tmp_path: Path) -> None:
    home = tmp_path / "home"
    report = reuse.build_report({"org/repo": (repo, "local")}, catalogue, home)
    t = report["totals"]
    # (4 method packages + 1 tandem) / (4 + 4 own pieces) = 62%
    assert t["peer_reviewed_percent"] == 62
    md = reuse.render_markdown(report)
    assert "Replace or pair next" in md
    assert "yes, already a dependency" in md
    page = reuse.render_html(report)
    assert "<title>Build vs Reuse</title>" in page
    assert "prefers-color-scheme" in page
    home.mkdir()
    reuse.append_history(home, "2026-10-28", report["repos"], t)
    reuse.append_history(home, "2026-11-04", report["repos"], t)
    rows = (home / "history.csv").read_text().splitlines()
    assert rows[0].startswith("date,repo")
    assert len(rows) == 5


def test_check_dois_reports_each(catalogue: dict) -> None:
    def fake(url: str) -> dict:
        if "nature" in url or "s41586" in url:
            raise OSError("down")
        return {"responseCode": 1 if "fnins" in url else 100}

    got = reuse.check_dois(catalogue, fetch=fake)
    assert got["mne"] == "ok"
    assert got["numpy"] == "error: OSError"
    assert got["scipy"] == "not found"
    assert "scikit-learn" not in got  # no DOI


def test_local_discovery_prefers_the_named_main_checkout(tmp_path: Path) -> None:
    import subprocess

    for name in ("tool", "tool-feature"):
        subprocess.run(["git", "init", "-q", str(tmp_path / name)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path / name),
                "remote",
                "add",
                "origin",
                "git@github.com:org/tool.git",
            ],
            check=True,
        )
    found = reuse.discover_local(tmp_path)
    assert found == {"org/tool": tmp_path / "tool"}


def test_dashboard_rows_are_flat(repo: Path, catalogue: dict, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    report = reuse.build_report({"org/repo": (repo, "local")}, catalogue, home)
    reuse.append_history(home, "2026-10-28", report["repos"], report["totals"])
    rows = reuse.dashboard_rows(report, home / "history.csv")
    assert rows["summary"][0]["repos"] == 1
    assert {c["id"] for c in rows["custom"]} == {"repo#0", "repo#1", "repo#2", "repo#3"}
    assert rows["history"][0]["peer_reviewed_percent"] == 62
    for table in rows.values():
        for row in table:
            assert all(not isinstance(v, (list, dict)) for v in row.values())
