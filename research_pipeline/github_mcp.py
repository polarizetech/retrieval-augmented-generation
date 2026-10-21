"""Small, PR-only GitHub MCP boundary for repositories in approved organizations.

Authentication remains in GitHub CLI's credential store.  The RAG OAuth key controls who may
reach this single-user service; GitHub still enforces the signed-in account's repository rights.
No tool pushes to a default branch, deletes a repository, merges a PR, or changes settings.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

GH = os.environ.get("GITHUB_MCP_GH", "/Users/joshuaanderton/.local/bin/gh")
ORGS = frozenset(
    item.strip().lower()
    for item in os.environ.get("GITHUB_MCP_ORGS", "polarizetech").split(",")
    if item.strip()
)
WRITE = os.environ.get("GITHUB_MCP_ALLOW_WRITE", "0") == "1"
READ = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True)
PROPOSE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
)

mcp = FastMCP(
    "organization-repositories",
    instructions=(
        "Search and read repositories only in the configured organization allowlist. Changes are "
        "full-file proposals on a new agent/* branch and pull request. Never merge or write a "
        "default branch. Read the current file before proposing its replacement."
    ),
)


def _run(argv: list[str], *, cwd: Path | None = None, timeout: int = 180) -> str:
    result = subprocess.run(  # noqa: S603 - argv is separated and targets gh/git only
        argv, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode:
        raise ValueError((result.stderr or result.stdout or "command failed")[:1200])
    return result.stdout


def _gh(*args: str, timeout: int = 180) -> str:
    return _run([GH, *args], timeout=timeout)


def _repository(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ValueError("repository must be owner/name")
    owner = value.split("/", 1)[0].lower()
    if owner not in ORGS:
        raise ValueError(f"repository owner must be one of: {', '.join(sorted(ORGS))}")
    return value


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"GitHub returned invalid JSON: {text[:300]}") from exc


@mcp.tool(annotations=READ)
def auth_status() -> dict[str, Any]:
    """Check GitHub authentication without returning credentials."""
    result = subprocess.run(  # noqa: S603 - fixed gh command
        [GH, "auth", "status", "--hostname", "github.com"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return {
        "authenticated": result.returncode == 0,
        "organizations": sorted(ORGS),
        "write_proposals_enabled": WRITE,
        "detail": "authenticated" if result.returncode == 0 else "run `gh auth login` on the host",
    }


@mcp.tool(annotations=READ)
def list_repositories(organization: str = "polarizetech", limit: int = 100) -> list[dict[str, Any]]:
    """List repositories visible to the authenticated GitHub account in an approved organization."""
    if organization.lower() not in ORGS:
        raise ValueError(f"organization must be one of: {', '.join(sorted(ORGS))}")
    return _json(
        _gh(
            "repo",
            "list",
            organization,
            "--limit",
            str(max(1, min(limit, 500))),
            "--json",
            "nameWithOwner,description,isPrivate,isArchived,defaultBranchRef,url,updatedAt",
        )
    )


@mcp.tool(annotations=READ)
def search_code(query: str, organization: str = "polarizetech", limit: int = 30) -> Any:
    """Search code across every visible repository in one approved organization."""
    if organization.lower() not in ORGS:
        raise ValueError(f"organization must be one of: {', '.join(sorted(ORGS))}")
    return _json(
        _gh(
            "search",
            "code",
            query,
            "--owner",
            organization,
            "--limit",
            str(max(1, min(limit, 100))),
            "--json",
            "path,repository,url,textMatches",
        )
    )


@mcp.tool(annotations=READ)
def read_file(repository: str, path: str, ref: str = "") -> dict[str, Any]:
    """Read one UTF-8 repository file from GitHub. Binary or oversized responses are refused."""
    repo = _repository(repository)
    rel = PurePosixPath(path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("path must stay inside the repository")
    endpoint = f"repos/{repo}/contents/{rel.as_posix()}"
    if ref:
        endpoint += f"?ref={ref}"
    text = _gh("api", endpoint, "-H", "Accept: application/vnd.github.raw+json")
    if len(text) > 256_000:
        raise ValueError("file exceeds the 256 kB text limit")
    return {"repository": repo, "path": rel.as_posix(), "ref": ref or "default", "text": text}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "change"


@mcp.tool(annotations=PROPOSE)
def propose_files(
    repository: str,
    title: str,
    body: str,
    files: list[dict[str, str]],
    base: str = "",
) -> dict[str, Any]:
    """Create a branch and pull request containing complete UTF-8 file replacements.

    `files` contains `{path, content}` objects. Deletion, merge and default-branch writes are not
    supported. Existing files should be read first. The user reviews the resulting PR.
    """
    repo = _repository(repository)
    if not WRITE:
        raise ValueError("GitHub write proposals are disabled (set GITHUB_MCP_ALLOW_WRITE=1)")
    if not files or len(files) > 20:
        raise ValueError("files must contain between one and twenty replacements")
    total = sum(len(item.get("content", "").encode()) for item in files)
    if total > 1_000_000:
        raise ValueError("the proposed text exceeds the 1 MB limit")
    branch = f"agent/{_slug(title)}-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory(prefix="github-mcp-") as tmp:
        root = Path(tmp) / "repo"
        # `gh repo clone owner/name` follows the user's preferred Git protocol. On a machine
        # authenticated for HTTPS but without an SSH key, that preference can make proposals
        # fail even though GitHub API reads work. Use the allowlisted repository's HTTPS URL.
        _run(
            ["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", str(root)],
            timeout=300,
        )
        default = (
            base
            or _json(_gh("repo", "view", repo, "--json", "defaultBranchRef"))["defaultBranchRef"][
                "name"
            ]
        )
        _run(["git", "switch", "-c", branch, f"origin/{default}"], cwd=root)
        written = []
        for item in files:
            rel = PurePosixPath(item.get("path", ""))
            if not rel.parts or rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"invalid repository path: {rel}")
            target = root.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item.get("content", ""))
            written.append(rel.as_posix())
        _run(["git", "add", "--", *written], cwd=root)
        changed = _run(["git", "diff", "--cached", "--name-only"], cwd=root).splitlines()
        if not changed:
            raise ValueError("the proposal makes no changes")
        _run(
            [
                "git",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-m",
                title[:200],
            ],
            cwd=root,
        )
        _run(["git", "push", "-u", "origin", branch], cwd=root, timeout=300)
        url = _gh(
            "pr",
            "create",
            "--repo",
            repo,
            "--base",
            default,
            "--head",
            branch,
            "--title",
            title[:200],
            "--body",
            body,
        ).strip()
    return {"repository": repo, "branch": branch, "base": default, "files": changed, "pr": url}


def main() -> int:
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
