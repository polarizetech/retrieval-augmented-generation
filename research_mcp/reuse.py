"""Build vs reuse: our own scientific code beside open-source, peer-reviewed packages.

Reads each repository's dependency files, imports and build-vs-reuse register (`reuse.toml`, or a
private seed), checks the register against the code, and writes a report plus a running tally.
No model is used. The register format and the states are in docs/reuse-register.md.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fnmatch
import html
import json
import os
import re
import subprocess
import sys
import tomllib
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
STATES = ("tandem", "candidate", "none-found", "replaced")
SKIP_DIRS = {
    ".git",
    ".claude",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    "site-packages",
    ".next",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "vendor",
    "archive",
}
CODE_SUFFIXES = {".py", ".ipynb", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".m"}
MAX_FILE_BYTES = 2_000_000
DEFAULT_SKIP = ("*-history", "*-archive", "*-knowledge")
HISTORY_FIELDS = (
    "date",
    "repo",
    "reused_methods",
    "reused_foundation",
    "tandem",
    "candidate",
    "none_found",
    "replaced",
    "issues",
)

PY_IMPORT = re.compile(r"^\s*(?:from\s+([A-Za-z_]\w*)|import\s+([A-Za-z_]\w*))", re.M)
NB_IMPORT = re.compile(r"(?:^|[\"\s])(?:from|import)\s+([A-Za-z_]\w*)", re.M)
JS_IMPORT = re.compile(r"""(?:from\s+|require\(\s*|import\(\s*)['"]([^'"]+)['"]""")
REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.\-]*)")
REMOTE = re.compile(r"github\.com[:/]([^/\s]+)/([^/\s]+?)(?:\.git)?/?$")


def reuse_home(value: str | None = None) -> Path:
    raw = value or os.environ.get("REUSE_HOME") or "~/.local/share/research-rag/reuse"
    return Path(raw).expanduser()


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


# --- catalogue -------------------------------------------------------------------------------


def load_catalogue(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """The default catalogue with the operator's own entries merged over it, key by key."""
    catalogue: dict[str, dict[str, Any]] = {}
    for name in ("reuse-catalogue.default.toml", "reuse-catalogue.toml"):
        path = root / "config" / name
        if path.exists():
            for key, entry in tomllib.loads(path.read_text()).items():
                catalogue.setdefault(key, {}).update(entry)
    return catalogue


def catalogue_index(catalogue: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Lookups from an import name, a PyPI or npm package name, to a catalogue key."""
    index: dict[str, dict[str, str]] = {"python": {}, "pypi": {}, "npm": {}}
    for key, entry in catalogue.items():
        for field in ("python", "pypi", "npm"):
            for name in entry.get(field, []):
                index[field][name if field == "python" else norm(name)] = key
    return index


def resolve_alternative(name: str, catalogue: dict[str, dict[str, Any]]) -> str | None:
    """A register's `alternative` as a catalogue key: the key itself, or any of its names."""
    if not name:
        return None
    if name in catalogue:
        return name
    wanted = norm(name)
    for key, entry in catalogue.items():
        names = [key, entry.get("name", ""), *entry.get("python", []), *entry.get("pypi", [])]
        if wanted in {norm(n) for n in names if n}:
            return key
    return None


def check_dois(
    catalogue: dict[str, dict[str, Any]], fetch: Callable[[str], Any] | None = None
) -> dict[str, str]:
    """Each catalogue DOI: "ok", "not found", or the error. Uses the doi.org handle API."""

    def http_fetch(url: str) -> Any:
        with urllib.request.urlopen(url, timeout=15) as response:
            return json.load(response)

    fetch = fetch or http_fetch
    results: dict[str, str] = {}
    for key, entry in catalogue.items():
        doi = entry.get("doi", "")
        if not doi:
            continue
        try:
            got = fetch(f"https://doi.org/api/handles/{doi}")
            results[key] = "ok" if got.get("responseCode") == 1 else "not found"
        except Exception as exc:  # noqa: BLE001 - any failure is reported, never fatal
            results[key] = f"error: {type(exc).__name__}"
    return results


# --- finding repositories ----------------------------------------------------------------------


def git(path: Path, *args: str) -> str:
    try:
        out = subprocess.run(  # noqa: S603 - fixed git arguments
            ["git", "-C", str(path), *args],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def slug_of(path: Path) -> str:
    match = REMOTE.search(git(path, "remote", "get-url", "origin"))
    return f"{match.group(1)}/{match.group(2)}" if match else f"local/{path.name}"


def discover_local(sites: Path) -> dict[str, Path]:
    """One checkout per repository: the folder named like the repo, preferring a main checkout."""
    found: dict[str, list[Path]] = {}
    if not sites.is_dir():
        return {}
    for child in sorted(sites.iterdir()):
        if child.is_dir() and (child / ".git").exists():
            found.setdefault(slug_of(child), []).append(child)

    def rank(slug: str, path: Path) -> tuple[bool, bool]:
        return (path.name != slug.rsplit("/", maxsplit=1)[-1], not (path / ".git").is_dir())

    return {slug: min(paths, key=lambda p, s=slug: rank(s, p)) for slug, paths in found.items()}


def list_org(org: str) -> list[str]:
    """The organisation's repositories on GitHub, archived ones left out."""
    try:
        out = subprocess.run(  # noqa: S603
            ["gh", "repo", "list", org, "--limit", "300", "--json", "nameWithOwner,isArchived"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [r["nameWithOwner"] for r in json.loads(out.stdout) if not r.get("isArchived")]


def clone_or_update(slug: str, clones: Path) -> Path | None:
    """A shallow copy of the default branch in our own cache; never a working checkout."""
    target = clones / slug.replace("/", "__")
    helper = ["-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential"]
    if (target / ".git").is_dir():
        git(target, *helper, "fetch", "--quiet", "--depth", "1", "origin")
        if git(target, "rev-parse", "--verify", "FETCH_HEAD"):
            git(target, "reset", "--quiet", "--hard", "FETCH_HEAD")
        return target
    clones.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(  # noqa: S603
            [  # noqa: S607 - git from PATH, as everywhere else here
                "git",
                *helper,
                "clone",
                "--quiet",
                "--depth",
                "1",
                f"https://github.com/{slug}.git",
                str(target),
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return target


# --- scanning one repository -------------------------------------------------------------------


def walk(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            yield Path(dirpath) / name


def read_text(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return ""
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def declared_packages(path: Path) -> tuple[set[str], set[str]]:
    """PyPI and npm package names from one dependency file."""
    pypi: set[str] = set()
    npm: set[str] = set()
    text = read_text(path)
    if path.name == "package.json":
        try:
            data = json.loads(text)
        except ValueError:
            return pypi, npm
        for field in (
            "dependencies",
            "devDependencies",
            "peerDependencies",
            "optionalDependencies",
        ):
            npm.update(norm(n) for n in (data.get(field) or {}))
    elif path.name == "pyproject.toml":
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            return pypi, npm
        project = data.get("project", {})
        specs: list[Any] = list(project.get("dependencies", []))
        for group in (project.get("optional-dependencies") or {}).values():
            specs.extend(group)
        for group in (data.get("dependency-groups") or {}).values():
            specs.extend(group)
        specs.extend((data.get("tool", {}).get("poetry", {}).get("dependencies") or {}).keys())
        for spec in specs:
            if isinstance(spec, str) and (m := REQ_NAME.match(spec)):
                pypi.add(norm(m.group(1)))
    else:  # requirements*.txt, environment.yml
        for raw in text.splitlines():
            line = raw.strip().lstrip("- ").strip()
            if line and not line.startswith(("#", "-")) and (m := REQ_NAME.match(line)):
                pypi.add(norm(m.group(1)))
    return pypi, npm


def is_dependency_file(path: Path) -> bool:
    name = path.name
    return name in {"pyproject.toml", "package.json", "environment.yml"} or (
        name.startswith("requirements") and name.endswith(".txt")
    )


def scan_packages(repo: Path, catalogue: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Catalogue packages the repository uses: where declared, and how many files use them."""
    index = catalogue_index(catalogue)
    markers = {m: k for k, e in catalogue.items() for m in e.get("markers", [])}
    used: dict[str, dict[str, Any]] = {}

    def hit(key: str, how: str, where: str) -> None:
        row = used.setdefault(key, {"declared": set(), "files": set()})
        row[how].add(where)

    for path in walk(repo):
        rel = str(path.relative_to(repo))
        if is_dependency_file(path):
            pypi, npm = declared_packages(path)
            for name in pypi:
                if name in index["pypi"]:
                    hit(index["pypi"][name], "declared", rel)
            for name in npm:
                if name in index["npm"]:
                    hit(index["npm"][name], "declared", rel)
            continue
        if path.suffix not in CODE_SUFFIXES:
            continue
        text = read_text(path)
        if not text:
            continue
        if path.suffix == ".py":
            names = {a or b for a, b in PY_IMPORT.findall(text)}
            keys = {index["python"][n] for n in names if n in index["python"]}
        elif path.suffix == ".ipynb":
            keys = {index["python"][n] for n in NB_IMPORT.findall(text) if n in index["python"]}
        elif path.suffix == ".m":
            keys = set()
        else:
            roots = set()
            for spec in JS_IMPORT.findall(text):
                if spec.startswith("."):
                    continue
                parts = spec.split("/")
                roots.add(norm("/".join(parts[:2]) if spec.startswith("@") else parts[0]))
            keys = {index["npm"][r] for r in roots if r in index["npm"]}
        keys |= {k for m, k in markers.items() if m in text}
        for key in keys:
            hit(key, "files", rel)
    return {
        key: {"declared": sorted(v["declared"]), "files": len(v["files"])}
        for key, v in sorted(used.items())
    }


def load_register(repo: Path, slug: str, home: Path) -> tuple[list[dict[str, Any]], str]:
    """The register's custom entries and where they came from: "repo", "seed" or ""."""
    for path, source in (
        (repo / "reuse.toml", "repo"),
        (home / "registers" / f"{slug.replace('/', '__')}.toml", "seed"),
    ):
        if path.exists():
            data = tomllib.loads(path.read_text())
            return list(data.get("custom", [])), source
    return [], ""


def check_entry(
    entry: dict[str, Any], repo: Path, used: dict[str, Any], catalogue: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """One register entry, with the catalogue paper of its alternative and anything that is off."""
    out = {
        k: entry.get(k, "")
        for k in ("name", "path", "does", "state", "alternative", "check", "note")
    }
    issues: list[str] = []
    state = out["state"]
    if state not in STATES:
        issues.append(f"unknown state {state!r}")
    if not out["path"] or not (repo / out["path"]).exists():
        issues.append("path not found")
    key = resolve_alternative(out["alternative"], catalogue)
    out["alternative_key"] = key or ""
    if key:
        ref = catalogue[key]
        out["alternative_name"] = ref.get("name", key)
        out["alternative_paper"] = ref.get("paper", "")
        out["alternative_doi"] = ref.get("doi", "")
        out["alternative_review"] = ref.get("review", "")
    elif out["alternative"]:
        out["alternative_name"] = out["alternative"]
        issues.append("alternative not in the catalogue: add it with its paper")
    if state in ("tandem", "candidate", "replaced") and not out["alternative"]:
        issues.append(f"{state} without an alternative")
    if state == "tandem":
        check = repo / out["check"] if out["check"] else None
        if check is None:
            issues.append("tandem without a check")
        elif not check.exists():
            issues.append("check not found")
        elif key or out["alternative"]:
            names = [out["alternative"]]
            if key:
                names += [key, *catalogue[key].get("python", []), *catalogue[key].get("npm", [])]
            text = read_text(check) if check.is_file() else ""
            if check.is_file() and not any(n and n.lower() in text.lower() for n in names):
                issues.append("check does not mention the alternative")
    out["easy_swap"] = bool(state == "candidate" and key and key in used)
    out["issues"] = issues
    return out


def scan_repo(
    slug: str, repo: Path, source: str, catalogue: dict[str, dict[str, Any]], home: Path
) -> dict[str, Any]:
    used = scan_packages(repo, catalogue)
    entries, register = load_register(repo, slug, home)
    custom = [check_entry(e, repo, used, catalogue) for e in entries]
    reused = [
        {
            "key": key,
            "name": catalogue[key].get("name", key),
            "paper": catalogue[key].get("paper", ""),
            "doi": catalogue[key].get("doi", ""),
            "review": catalogue[key].get("review", ""),
            "foundation": bool(catalogue[key].get("foundation")),
            **info,
        }
        for key, info in used.items()
    ]
    counts = {s: sum(1 for c in custom if c["state"] == s) for s in STATES}
    return {
        "repo": slug,
        "source": source,
        "branch": git(repo, "branch", "--show-current"),
        "commit": git(repo, "rev-parse", "--short", "HEAD"),
        "register": register,
        "reused": reused,
        "custom": custom,
        "counts": {
            "reused_methods": sum(1 for r in reused if not r["foundation"]),
            "reused_foundation": sum(1 for r in reused if r["foundation"]),
            "tandem": counts["tandem"],
            "candidate": counts["candidate"],
            "none_found": counts["none-found"],
            "replaced": counts["replaced"],
            "issues": sum(len(c["issues"]) for c in custom),
        },
    }


# --- the report ----------------------------------------------------------------------------------


def totals(repos: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "reused_methods",
        "reused_foundation",
        "tandem",
        "candidate",
        "none_found",
        "replaced",
        "issues",
    )
    out = {k: sum(r["counts"][k] for r in repos) for k in keys}
    # How much of the scientific work leans on peer-reviewed code: method packages in use, plus our
    # own code that runs in tandem with one, over everything (method packages plus our own code).
    own = out["tandem"] + out["candidate"] + out["none_found"]
    base = out["reused_methods"] + own
    out["peer_reviewed_percent"] = (
        round(100 * (out["reused_methods"] + out["tandem"]) / base) if base else 0
    )
    out["own_code"] = own
    return out


def append_history(
    home: Path, date: str, repos: list[dict[str, Any]], total: dict[str, int]
) -> None:
    path = home / "history.csv"
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if new:
            writer.writeheader()
        for row in [*({"repo": r["repo"], **r["counts"]} for r in repos), {"repo": "ALL", **total}]:
            writer.writerow({"date": date, **{k: row.get(k, "") for k in HISTORY_FIELDS[1:]}})


def render_markdown(report: dict[str, Any]) -> str:
    t = report["totals"]
    lines = [
        f"# Build vs reuse ({report['generated'][:10]})",
        "",
        f"{t['peer_reviewed_percent']}% of our scientific work leans on peer-reviewed open-source "
        f"code: {t['reused_methods']} method packages in use, {t['tandem']} of our own pieces run "
        f"in tandem with one, {t['candidate']} have a known alternative not wired in, "
        f"{t['none_found']} have none found yet. Foundations (NumPy, SciPy and the like) in use: "
        f"{t['reused_foundation']}.",
        "",
        "| " + " | ".join(REPO_HEAD) + " |",
        "|---" * len(REPO_HEAD) + "|",
    ]
    for r in report["repos"]:
        c = r["counts"]
        lines.append(
            f"| {r['repo']} | {c['reused_methods']} | {c['reused_foundation']} | {c['tandem']} | "
            f"{c['candidate']} | {c['none_found']} | {c['issues']} | {r['register'] or 'none'} |"
        )
    swaps = [
        (r["repo"], c) for r in report["repos"] for c in r["custom"] if c["state"] == "candidate"
    ]
    swaps.sort(key=lambda x: (not x[1]["easy_swap"], x[0]))
    lines += [
        "",
        "## Replace or pair next",
        "",
        "| Repo | Our code | Alternative | Paper | Easy swap |",
        "|---|---|---|---|---|",
    ]
    for repo, c in swaps:
        paper = c.get("alternative_paper", "") + (
            f" doi:{c['alternative_doi']}" if c.get("alternative_doi") else ""
        )
        lines.append(
            f"| {repo} | {c['name']} (`{c['path']}`) | {c.get('alternative_name', '')} | "
            f"{paper} | {'yes, already a dependency' if c['easy_swap'] else ''} |"
        )
    lines += ["", "## Search targets (no alternative found yet)", ""]
    lines += [
        f"- {r['repo']}: {c['name']} (`{c['path']}`): {c['does']}"
        for r in report["repos"]
        for c in r["custom"]
        if c["state"] == "none-found"
    ] or ["- none"]
    lines += ["", "## Running in tandem", ""]
    lines += [
        f"- {r['repo']}: {c['name']} with "
        f"{c.get('alternative_name', c['alternative'])} (`{c['check']}`)"
        for r in report["repos"]
        for c in r["custom"]
        if c["state"] == "tandem"
    ] or ["- none"]
    lines += ["", "## Register issues", ""]
    lines += [
        f"- {r['repo']}: {c['name']}: {'; '.join(c['issues'])}"
        for r in report["repos"]
        for c in r["custom"]
        if c["issues"]
    ] or ["- none"]
    missing = [r["repo"] for r in report["repos"] if not r["register"]]
    lines += ["", "## Repositories with no register", "", ", ".join(missing) or "none"]
    if report.get("dois"):
        bad = {k: v for k, v in report["dois"].items() if v != "ok"}
        lines += ["", "## Catalogue DOIs", "", f"{len(report['dois']) - len(bad)} resolve."]
        lines += [f"- {k}: {v}" for k, v in bad.items()]
    return "\n".join(lines) + "\n"


DARK = "--bg:#161615;--fg:#ecebe6;--muted:#a3a29b;--line:#33322f;--card:#1f1f1d;--accent:#8db4dc"
CSS = "\n".join(
    [
        ":root{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#6b6b66;--line:#e3e2dc;--card:#fff;"
        "--accent:#2f5d8a}",
        f'@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{{DARK}}}}}',
        f':root[data-theme="dark"]{{{DARK}}}',
        "body{margin:0;background:var(--bg);color:var(--fg);"
        "font:15px/1.5 system-ui,-apple-system,sans-serif}",
        "main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}",
        "h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px}",
        ".muted{color:var(--muted)}a{color:var(--accent)}",
        ".stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));"
        "gap:12px;margin:20px 0}",
        ".stats div{background:var(--card);border:1px solid var(--line);"
        "border-radius:10px;padding:12px}",
        ".stats b{display:block;font-size:26px}.stats span{color:var(--muted);font-size:13px}",
        ".scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13.5px}",
        "th,td{text-align:left;vertical-align:top;padding:6px 8px;"
        "border-bottom:1px solid var(--line)}",
        "th{color:var(--muted);font-weight:600}code{font-size:12px;color:var(--muted)}",
    ]
)
REPO_HEAD = [
    "Repo",
    "Method pkgs",
    "Foundations",
    "Tandem",
    "Candidate",
    "None found",
    "Issues",
    "Register",
]


def render_html(report: dict[str, Any]) -> str:
    t = report["totals"]
    esc = html.escape

    def table(head: list[str], rows: list[list[str]]) -> str:
        if not rows:
            return '<p class="muted">None.</p>'
        th = "".join(f"<th>{esc(h)}</th>" for h in head)
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
        return (
            f'<div class="scroll"><table><thead><tr>{th}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>"
        )

    def ref(c: dict[str, Any]) -> str:
        doi = c.get("alternative_doi", "")
        paper = esc(c.get("alternative_paper", ""))
        return f'<a href="https://doi.org/{esc(doi)}">{paper}</a>' if doi else paper

    repo_rows = [
        [
            esc(r["repo"]),
            *(
                str(r["counts"][k])
                for k in (
                    "reused_methods",
                    "reused_foundation",
                    "tandem",
                    "candidate",
                    "none_found",
                    "issues",
                )
            ),
            esc(r["register"] or "none"),
        ]
        for r in report["repos"]
    ]
    customs = [(r["repo"], c) for r in report["repos"] for c in r["custom"]]
    candidates = sorted(
        (x for x in customs if x[1]["state"] == "candidate"),
        key=lambda x: (not x[1]["easy_swap"], x[0]),
    )
    cand_rows = [
        [
            esc(repo),
            f"{esc(c['name'])}<br><code>{esc(c['path'])}</code>",
            esc(c.get("alternative_name", "")),
            ref(c),
            "already a dependency" if c["easy_swap"] else "",
            esc(c["note"]),
        ]
        for repo, c in candidates
    ]
    tandem_rows = [
        [
            esc(repo),
            esc(c["name"]),
            esc(c.get("alternative_name", "")),
            f"<code>{esc(c['check'])}</code>",
            esc("; ".join(c["issues"])),
        ]
        for repo, c in customs
        if c["state"] == "tandem"
    ]
    none_rows = [
        [esc(repo), f"{esc(c['name'])}<br><code>{esc(c['path'])}</code>", esc(c["does"])]
        for repo, c in customs
        if c["state"] == "none-found"
    ]
    used: dict[str, dict[str, Any]] = {}
    for r in report["repos"]:
        for u in r["reused"]:
            row = used.setdefault(u["key"], {**u, "repos": []})
            row["repos"].append(r["repo"].split("/")[-1])
    pkg_rows = [
        [
            esc(u["name"]),
            (
                f'<a href="https://doi.org/{esc(u["doi"])}">{esc(u["paper"])}</a>'
                if u["doi"]
                else esc(u["paper"])
            ),
            "foundation" if u["foundation"] else "method",
            esc(", ".join(u["repos"])),
        ]
        for u in sorted(used.values(), key=lambda u: (u["foundation"], -len(u["repos"])))
    ]
    issue_rows = [
        [esc(repo), esc(c["name"]), esc("; ".join(c["issues"]))]
        for repo, c in customs
        if c["issues"]
    ]
    tiles = [
        (f"{t['peer_reviewed_percent']}%", "leans on peer-reviewed code"),
        (t["reused_methods"], "method packages in use"),
        (t["tandem"], "ours, run in tandem"),
        (t["candidate"], "ours, alternative known"),
        (t["none_found"], "ours, none found yet"),
    ]
    stat = (
        '<div class="stats">'
        + "".join(f"<div><b>{v}</b><span>{label}</span></div>" for v, label in tiles)
        + "</div>"
    )
    generated = esc(report["generated"][:16].replace("T", " "))
    sections = [
        (
            "Replace or pair next",
            ["Repo", "Our code", "Alternative", "Paper", "Swap", "How"],
            cand_rows,
        ),
        ("Running in tandem", ["Repo", "Our code", "Alongside", "Check", "Issues"], tandem_rows),
        ("Search targets: no alternative found yet", ["Repo", "Our code", "Does"], none_rows),
        ("Peer-reviewed packages in use", ["Package", "Paper", "Kind", "Repos"], pkg_rows),
        ("By repository", REPO_HEAD, repo_rows),
        ("Register issues", ["Repo", "Entry", "Issue"], issue_rows),
    ]
    body = "\n".join(f"<h2>{esc(title)}</h2>{table(head, rows)}" for title, head, rows in sections)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Build vs Reuse</title>\n<style>\n{CSS}\n</style></head><body><main>\n"
        "<h1>Build vs reuse</h1>\n"
        '<p class="muted">Our own scientific code beside open-source, peer-reviewed packages. '
        f"Generated {generated} UTC from {len(report['repos'])} repositories. Peer-reviewed share "
        "= (method packages in use + our code run in tandem) / (method packages + all our own "
        f"code).</p>\n{stat}\n{body}\n</main></body></html>\n"
    )


def build_report(
    repos: dict[str, tuple[Path, str]], catalogue: dict[str, dict[str, Any]], home: Path
) -> dict[str, Any]:
    scanned = [
        scan_repo(slug, path, source, catalogue, home)
        for slug, (path, source) in sorted(repos.items())
    ]
    return {
        "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "repos": scanned,
        "totals": totals(scanned),
    }


def gather(args: argparse.Namespace, home: Path) -> dict[str, tuple[Path, str]]:
    skip = [*DEFAULT_SKIP, *args.skip]

    def wanted(slug: str) -> bool:
        name = slug.rsplit("/", maxsplit=1)[-1]
        if any(fnmatch.fnmatch(name, p) for p in skip):
            return False
        return not args.only or slug in args.only or name in args.only

    repos: dict[str, tuple[Path, str]] = {}
    for slug, path in discover_local(Path(args.sites).expanduser()).items():
        owner = slug.split("/")[0]
        if not wanted(slug) or (args.org and owner not in (args.org, "local")):
            continue
        if owner == "local" and not load_register(path, slug, home)[1]:
            continue  # a folder with no GitHub remote counts only once it has a register
        repos[slug] = (path, "local")
    if args.org and not args.no_clone:
        for slug in list_org(args.org):
            if slug not in repos and wanted(slug):
                path = clone_or_update(slug, home / "clones")
                if path:
                    repos[slug] = (path, "clone")
    return repos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="research-reuse",
        description="Build vs reuse: our own scientific code beside peer-reviewed packages.",
    )
    parser.add_argument("--sites", default="~/Sites", help="folder of local checkouts")
    parser.add_argument("--org", default="polarizetech", help="GitHub organisation ('' for none)")
    parser.add_argument("--no-clone", action="store_true", help="local checkouts only")
    parser.add_argument("--home", default=None, help="report folder (default $REUSE_HOME)")
    parser.add_argument("--skip", action="append", default=[], help="repo name pattern to skip")
    parser.add_argument("--only", action="append", default=[], help="scan only this repo")
    parser.add_argument("--check-dois", action="store_true", help="confirm catalogue DOIs resolve")
    parser.add_argument("--no-history", action="store_true", help="do not add to history.csv")
    args = parser.parse_args(argv)

    home = reuse_home(args.home)
    home.mkdir(parents=True, exist_ok=True)
    catalogue = load_catalogue()
    report = build_report(gather(args, home), catalogue, home)
    if args.check_dois:
        report["dois"] = check_dois(catalogue)
    (home / "report.json").write_text(json.dumps(report, indent=1, default=list) + "\n")
    markdown = render_markdown(report)
    (home / "report.md").write_text(markdown)
    (home / "report.html").write_text(render_html(report))
    if not args.no_history:
        append_history(home, report["generated"][:10], report["repos"], report["totals"])
    sys.stdout.write(markdown.split("\n## ")[0] + f"\nReport: {home / 'report.md'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
