from __future__ import annotations

import pytest

from research_pipeline import github_mcp, patent_mcp


def test_patent_adapter_bounds_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patent_mcp, "_run", lambda *args, **kwargs: "x" * 2000)
    result = patent_mcp.text("US123", max_chars=1000)
    assert len(result["text"]) == 1000
    assert result["truncated"] is True


def test_github_repository_boundary() -> None:
    assert github_mcp._repository("polarizetech/example") == "polarizetech/example"
    with pytest.raises(ValueError, match="repository owner"):
        github_mcp._repository("someone-else/example")
    with pytest.raises(ValueError, match="owner/name"):
        github_mcp._repository("not-a-repository")


def test_github_read_file_rejects_traversal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(github_mcp, "_gh", lambda *args, **kwargs: "should not run")
    with pytest.raises(ValueError, match="inside the repository"):
        github_mcp.read_file("polarizetech/example", "../secret")
