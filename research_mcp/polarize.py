"""The polarize tools: questions about polarize-holo, -mixer, -particle, -physics and -clock,
answered from their own repositories, and audio rendered by polarize-holo.

Each tool is its own repository (polarizetech/<repo>). This module only reads them and runs holo's
own render scripts; it holds no tool's knowledge and synthesises no sound.

**Pinning.** A tool is read at its newest release tag, found in its local checkout
(POLARIZE_<TOOL>_DIR, else POLARIZE_TOOLS_DIR/<repo>, else the folder beside this repository). A
tool with no tag, or no local checkout, is read at its default branch, and the answer says so.

**Questions** (`read`). README.md and TOOL.toml at the pinned tag, through the gateway's GitHub
tools (`repos__read_file`; `repos__search_code` for code, which GitHub searches on the default
branch only). When those tools are not connected, the same files are read from the local
checkout with `git show <tag>:<path>`. Every file comes back with the tool, tag and commit it
was read at.

**Audio** (`render`). polarize-holo's renders are folders `renders/<slug>/` holding `render_*.py`
scripts that write WAV files. A render is copied out of holo's git history with `git archive` (at
the tag when the render exists there, else at the checkout's HEAD, said in the result) into a new
folder under POLARIZE_RENDER_DIR, and its script runs there with `uv run`, in the render's own
project when it has a pyproject.toml (whose tool pins are reported) and holo's otherwise. holo's
checkout is never written to. A script's `usage:` line names its arguments: OUT_DIR is the run's
folder; any other (SUN_AZ_DEG, ...) must be given. The result lists every WAV written with its
format and length, every JSON the script wrote beside it, and the exact command.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import wave
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ORG = "polarizetech"
#: tool key -> repository name
TOOLS = {
    "holo": "polarize-holo",
    "mixer": "polarize-mixer",
    "particle": "polarize-particle",
    "physics": "polarize-physics",
    "clock": "polarize-clock",
}
FILES = ("README.md", "TOOL.toml")
MAX_FILE = 12_000  # characters of each file sent back
MAX_JSON = 4_000
RENDER_TIMEOUT = float(os.environ.get("POLARIZE_RENDER_TIMEOUT", "900"))
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_SCRIPT = re.compile(r"^render_[A-Za-z0-9_]{1,60}\.py$")
_ARG = re.compile(r"^[A-Za-z0-9._:+-]{1,60}$")
_USAGE = re.compile(r"usage:\s*(?:uv run\s+)?python\s+(\S+\.py)((?:\s+[A-Z][A-Z0-9_]*)*)")
_STOP = frozenset(
    """
    a an and are as at be by can could do does for from how i in is it its me of on or
    please polarize show tell that the this to tool use uses what when where which why with
    would you your
    """.split()  # noqa: SIM905 - a word list reads better as words
)
CallTool = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


# -- where each tool is --------------------------------------------------------------------------
def tools_dir() -> Path:
    return Path(os.environ.get("POLARIZE_TOOLS_DIR") or ROOT.parent).expanduser()


def checkout(tool: str) -> Path | None:
    """The tool's local git checkout, or None."""
    direct = os.environ.get(f"POLARIZE_{tool.upper()}_DIR", "").strip()
    path = Path(direct).expanduser() if direct else tools_dir() / TOOLS[tool]
    return path if (path / ".git").exists() else None


def _git(repo: Path, *args: str, timeout: float = 30, strip: bool = True) -> str:
    done = subprocess.run(  # noqa: S603 (git with fixed arguments)
        ["git", "-C", str(repo), *args],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return done.stdout.strip() if strip else done.stdout


def pin(tool: str) -> dict[str, Any]:
    """Where `tool` is read from: its newest tag, or why it is not pinned."""
    repo = TOOLS[tool]
    out: dict[str, Any] = {"tool": repo, "repository": f"{ORG}/{repo}", "ref": "", "pinned": False}
    path = checkout(tool)
    if path is None:
        out["note"] = (
            f"no local checkout (POLARIZE_{tool.upper()}_DIR or POLARIZE_TOOLS_DIR/{repo}): "
            "read at the default branch, not pinned to a tag"
        )
        return out
    out["checkout"] = str(path)
    try:
        out["head"] = _git(path, "rev-parse", "--short", "HEAD")
        tag = _git(path, "describe", "--tags", "--abbrev=0")
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        out["note"] = f"{repo} has no release tag: read at its checkout's HEAD"
        out["ref"] = out.get("head", "")
        return out
    out.update(
        ref=tag,
        tag=tag,
        pinned=True,
        commit=_git(path, "rev-parse", "--short", f"{tag}^{{commit}}"),
        ahead=int(_git(path, "rev-list", "--count", f"{tag}..HEAD") or 0),
    )
    if out["ahead"]:
        out["note"] = (
            f"read at {tag}; the checkout is {out['ahead']} commit(s) past it (HEAD {out['head']})"
        )
    return out


# -- which tools a request names -----------------------------------------------------------------
def named(question: str, aliases: dict[str, tuple[str, ...]]) -> list[str]:
    """Tool keys the request names, in TOOLS order. `aliases` comes from the routing rules."""
    low = question.lower()
    return [
        tool
        for tool in TOOLS
        if any(
            re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", low)
            for w in aliases.get(tool, ())
        )
    ]


def wants_audio(question: str, words: tuple[str, ...]) -> bool:
    low = question.lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", low) for w in words)


def terms(question: str, aliases: dict[str, tuple[str, ...]]) -> list[str]:
    """A few content words for a code search, without the tools' own names."""
    low = question.lower()
    for words in aliases.values():
        for w in words:
            low = low.replace(w, " ")
    words = [w for w in re.findall(r"[a-z][a-z0-9_]{2,}", low) if w not in _STOP]
    return list(dict.fromkeys(words))[:4]


# -- questions: read the tool's own files --------------------------------------------------------
async def read(
    tool: str,
    question: str,
    aliases: dict[str, tuple[str, ...]],
    call_tool: CallTool | None,
    has_tool: Callable[[str], bool],
) -> dict[str, Any]:
    """README.md, TOOL.toml and a code search for one tool, at its pinned tag."""
    where = pin(tool)
    out: dict[str, Any] = {**where, "files": {}, "code": [], "not_read": []}
    remote = call_tool is not None and has_tool("repos__read_file")
    for name in FILES:
        text, how = "", ""
        if remote:
            assert call_tool is not None
            try:
                got = await call_tool(
                    "repos__read_file",
                    {"repository": where["repository"], "path": name, "ref": where["ref"]},
                )
                text, how = str(got.get("text") or ""), "GitHub"
            except Exception as exc:  # noqa: BLE001 - one file never stops the answer
                out["not_read"].append(f"{name} via GitHub: {type(exc).__name__}")
        if not text and where.get("checkout"):
            try:
                text = _git(
                    Path(where["checkout"]), "show", f"{where['ref'] or 'HEAD'}:{name}", strip=False
                )
                how = "local checkout"
            except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
                out["not_read"].append(f"{name}: not in the checkout at {where['ref'] or 'HEAD'}")
        if text:
            out["files"][name] = {
                "text": text[:MAX_FILE],
                "truncated": len(text) > MAX_FILE,
                "via": how,
            }
    words = terms(question, aliases)
    if words and call_tool is not None and has_tool("repos__search_code"):
        try:
            got = await call_tool(
                "repos__search_code",
                {"query": f"{' '.join(words)} repo:{where['repository']}", "limit": 8},
            )
            rows = got if isinstance(got, list) else got.get("result") or got.get("items") or []
            out["code"] = [
                {"path": r.get("path"), "url": r.get("url")} for r in rows if isinstance(r, dict)
            ]
            out["code_note"] = "GitHub code search reads the default branch, not the pinned tag."
        except Exception as exc:  # noqa: BLE001
            out["not_read"].append(f"code search: {type(exc).__name__}")
    return out


# -- audio: holo's renders -----------------------------------------------------------------------
def _uv() -> str:
    return os.environ.get("POLARIZE_UV") or shutil.which("uv") or "uv"


def render_dir() -> Path:
    return Path(os.environ.get("POLARIZE_RENDER_DIR") or ROOT / "data" / "polarize-renders")


def _usage(text: str, script: str) -> list[str]:
    """The arguments a render script's `usage:` line names, in order."""
    for m in _USAGE.finditer(text):
        if Path(m.group(1)).name == script:
            return m.group(2).split()
    return []


def renders(holo: Path | None = None) -> dict[str, Any]:
    """Every render in holo, at the tag where it exists there, else at HEAD."""
    holo = holo or checkout("holo")
    if holo is None:
        return {"error": "no polarize-holo checkout (POLARIZE_HOLO_DIR or POLARIZE_TOOLS_DIR)"}
    try:
        tag: str | None = _git(holo, "describe", "--tags", "--abbrev=0")
    except subprocess.CalledProcessError:
        tag = None
    head = _git(holo, "rev-parse", "--short", "HEAD")
    at_tag: set[str] = set()
    if tag:
        at_tag = set(_git(holo, "ls-tree", "-d", "--name-only", f"{tag}:renders").split())
    out = []
    for slug in _git(holo, "ls-tree", "-d", "--name-only", "HEAD:renders").split():
        ref = tag if slug in at_tag else "HEAD"
        files = _git(holo, "ls-tree", "--name-only", f"{ref}:renders/{slug}").split()
        scripts = []
        for name in files:
            if not _SCRIPT.match(name):
                continue
            text = _git(holo, "show", f"{ref}:renders/{slug}/{name}")
            scripts.append({"script": name, "args": _usage(text, name)})
        if scripts:
            out.append(
                {
                    "slug": slug,
                    "ref": ref,
                    "at_tag": ref == tag,
                    "own_project": "pyproject.toml" in files,
                    "scripts": scripts,
                }
            )
    return {"tool": "polarize-holo", "tag": tag, "head": head, "renders": out}


def choose(question: str, listed: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The render a request names by a word of its slug ("spiral", "guitar pluck"), else None."""
    low = question.lower()
    for row in listed:
        words = [w for w in row["slug"].split("-") if not w.isdigit()]
        if words and all(re.search(rf"(?<![a-z0-9]){re.escape(w)}", low) for w in words):
            return row
    return None


def _main_script(row: dict[str, Any]) -> str:
    tail = "_".join(w for w in row["slug"].split("-") if not w.isdigit())
    names = [s["script"] for s in row["scripts"]]
    return f"render_{tail}.py" if f"render_{tail}.py" in names else names[0]


def _wav(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size}
    try:
        with wave.open(str(path), "rb") as w:
            info.update(
                channels=w.getnchannels(),
                sample_rate=w.getframerate(),
                bits=8 * w.getsampwidth(),
                seconds=round(w.getnframes() / w.getframerate(), 3),
            )
    except (wave.Error, EOFError):
        info["format"] = "not plain PCM (header not read)"
    return info


def _pins(project: Path) -> dict[str, str]:
    try:
        data = tomllib.loads(project.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    return {
        name: str(src.get("tag") or src.get("rev") or src.get("branch") or src.get("path") or src)
        for name, src in sources.items()
        if isinstance(src, dict)
    }


def render(
    slug: str, script: str = "", args: dict[str, Any] | None = None, *, holo: Path | None = None
) -> dict[str, Any]:
    """Run one holo render script on a copy of its folder; report what it wrote."""
    holo = holo or checkout("holo")
    if holo is None:
        return {"status": "not configured", "note": "set POLARIZE_HOLO_DIR or POLARIZE_TOOLS_DIR"}
    if not _SLUG.match(slug):
        raise ValueError(f"render: bad slug {slug!r}")
    listed = renders(holo)
    row = next((r for r in listed.get("renders", []) if r["slug"] == slug), None)
    if row is None:
        names = ", ".join(r["slug"] for r in listed.get("renders", []))
        raise ValueError(f"holo has no render {slug!r}; it has: {names or 'none'}")
    script = script or _main_script(row)
    entry = next((s for s in row["scripts"] if s["script"] == script), None)
    if entry is None or not _SCRIPT.match(script):
        raise ValueError(f"render {slug} has no script {script!r}")
    args = {str(k).upper(): v for k, v in (args or {}).items()}
    ref = row["ref"]
    commit = _git(holo, "rev-parse", "--short", f"{'HEAD' if ref == 'HEAD' else ref}^{{commit}}")
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = render_dir() / f"{stamp}-{slug}"
    out_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "render.tar"
        _git(holo, "archive", "--format=tar", "-o", str(archive), ref, f"renders/{slug}")
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        shutil.copytree(Path(tmp) / "renders" / slug, out_dir, dirs_exist_ok=True)
    argv: list[str] = []
    missing = []
    for name in entry["args"]:
        if name == "OUT_DIR":
            argv.append(str(out_dir))
        elif name in args and _ARG.match(str(args[name])):
            argv.append(str(args[name]))
        else:
            missing.append(name)
    result: dict[str, Any] = {
        "tool": "polarize-holo",
        "render": slug,
        "script": script,
        "ref": ref if ref != "HEAD" else None,
        "pinned": ref != "HEAD",
        "commit": commit,
        "version": listed.get("tag"),
        "output_dir": str(out_dir),
        "tier": "EXPLORATORY: a rendering by polarize-holo, not a finding",
    }
    if ref == "HEAD":
        result["note"] = (
            f"{slug} is not in holo's newest tag ({listed.get('tag') or 'none'}): rendered from "
            f"the checkout's HEAD, commit {commit}"
        )
    if missing:
        result.update(status="needs arguments", missing=missing)
        return result
    own = (out_dir / "pyproject.toml").is_file()
    project = out_dir if own else holo
    command = [_uv(), "run", "--project", str(project), "python", script, *argv]
    result["command"] = command
    result["environment"] = "the render's own pyproject.toml" if own else "holo's root project"
    if own:
        result["pins"] = _pins(out_dir / "pyproject.toml")
    # The environment lives outside the output, shared by renders with the same lock file.
    lock = (out_dir / "uv.lock") if (out_dir / "uv.lock").is_file() else project / "pyproject.toml"
    digest = hashlib.sha256(lock.read_bytes() if lock.is_file() else b"").hexdigest()[:12]
    env = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": str(render_dir() / ".envs" / f"{project.name}-{digest}"),
    }
    before = {p: p.stat().st_mtime for p in _files(out_dir)}
    try:
        done = subprocess.run(  # noqa: S603 (uv, a checked script name and checked arguments)
            command,
            cwd=out_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=RENDER_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
        return result
    written = [p for p in _files(out_dir) if p not in before or p.stat().st_mtime > before[p]]
    result["wavs"] = [_wav(p) for p in written if p.suffix.lower() == ".wav"]
    result["other_outputs"] = [str(p) for p in written if p.suffix.lower() not in (".wav", ".json")]
    result["settings"] = {
        p.name: _json(p) for p in written if p.suffix.lower() == ".json" and p.stat().st_size
    }
    if done.returncode != 0:
        result.update(status="failed", error=(done.stderr or done.stdout)[-1500:])
    elif not result["wavs"]:
        result.update(status="failed", error="the script finished but wrote no WAV file")
    else:
        result["status"] = "done"
    return result


def _files(folder: Path) -> list[Path]:
    """The render folder's own files: nothing under a hidden folder such as .venv."""
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(folder).parts)
    )


def _json(path: Path) -> Any:
    text = path.read_text(errors="replace")
    try:
        data = json.loads(text)
    except ValueError:
        return text[:MAX_JSON]
    dumped = json.dumps(data)
    return data if len(dumped) <= MAX_JSON else dumped[:MAX_JSON] + "…"


# -- what ask shows --------------------------------------------------------------------------------
def summary(result: dict[str, Any]) -> list[str]:
    """One line per tool read or render run, for ask's short reply."""
    lines = []
    for row in result.get("read", []):
        files = ", ".join(row["files"]) or "no files"
        at = f"at {row['tag']} ({row['commit']})" if row.get("pinned") else "not pinned to a tag"
        lines.append(f"Read {row['tool']} {at}: {files}.")
    audio = result.get("render")
    if audio:
        if audio.get("status") == "done":
            wavs = "; ".join(w["path"] for w in audio["wavs"])
            lines.append(
                f"polarize-holo rendered {audio['render']} ({audio.get('ref') or 'HEAD'}, commit "
                f"{audio['commit']}): {wavs}."
            )
        elif audio.get("status") == "needs arguments":
            lines.append(
                f"polarize-holo's {audio['render']} needs {', '.join(audio['missing'])} to render."
            )
        else:
            lines.append(
                f"polarize-holo's render didn't finish: {str(audio.get('error') or audio)[:200]}"
            )
    elif result.get("renders") is not None:
        names = ", ".join(r["slug"] for r in result["renders"]) or "none"
        lines.append(f"Audio was asked for but no render was named; holo has: {names}.")
    return lines
