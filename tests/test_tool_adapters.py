from __future__ import annotations

from pathlib import PurePosixPath

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


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/etc/passwd",
        "../outside",
        ".git/config",
        "a/.GIT/hooks/pre-commit",
        ".github/workflows",
        ".github/workflows/ci.yml",
        ".GitHub/Workflows/ci.yml",
        ".github/actions/setup/action.yml",
    ],
)
def test_github_proposal_rejects_protected_paths(path: str) -> None:
    with pytest.raises(ValueError, match=r"repository path|git's own files|GitHub Actions"):
        github_mcp._proposal_path(path)


@pytest.mark.parametrize(
    "path", [".github/dependabot.yml", "docs/.github-notes.md", "src/git/main.py", "README.md"]
)
def test_github_proposal_allows_ordinary_paths(path: str) -> None:
    assert github_mcp._proposal_path(path).as_posix() == path


def test_github_proposal_refuses_symlinks(tmp_path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    assert github_mcp._inside(tmp_path, PurePosixPath("real/a.txt")) == tmp_path / "real/a.txt"
    with pytest.raises(ValueError, match="symbolic link"):
        github_mcp._inside(tmp_path, PurePosixPath("link/a.txt"))


def test_github_propose_checks_paths_before_running_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("no command may run for a rejected proposal")

    monkeypatch.setattr(github_mcp, "WRITE", True)
    monkeypatch.setattr(github_mcp, "_run", fail)
    files = [{"path": "ok.md", "content": "x"}, {"path": ".git/config", "content": "x"}]
    with pytest.raises(ValueError, match="git's own files"):
        github_mcp.propose_files("polarizetech/example", "t", "b", files)


def test_github_token_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    token = tmp_path / "token"
    token.write_text("bogus-test-token\n")
    monkeypatch.setenv("GITHUB_TOKEN", "other")
    monkeypatch.setattr(github_mcp, "TOKEN_FILE", str(token))
    env = github_mcp._env()
    assert env["GH_TOKEN"] == "bogus-test-token"
    assert "GITHUB_TOKEN" not in env
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_github_token_file_errors_do_not_leak(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(github_mcp, "TOKEN_FILE", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="cannot be read"):
        github_mcp._env()
    empty = tmp_path / "empty"
    empty.write_text("  \n")
    monkeypatch.setattr(github_mcp, "TOKEN_FILE", str(empty))
    with pytest.raises(ValueError, match="is empty"):
        github_mcp._env()


def test_github_without_token_file_keeps_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(github_mcp, "TOKEN_FILE", "")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert "GH_TOKEN" not in github_mcp._env()


def test_github_git_disables_hooks_and_uses_gh_credentials() -> None:
    argv = github_mcp._git("status")
    assert argv[0] == "git"
    assert argv[-1] == "status"
    joined = " ".join(argv)
    assert "core.hooksPath=/dev/null" in joined
    assert "core.fsmonitor=false" in joined
    assert "auth git-credential" in joined
