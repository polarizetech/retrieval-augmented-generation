"""Decide which parts of the pipeline a request recruits, deterministically and cheaply.

`ask` sends every request through `classify`: a quick passage retrieval by default, the research
pipeline when the request is about evidence, and the coder only for a task or a code review. The
words come from config/routing.default.toml plus the operator's gitignored config/routing.toml,
so the person decides what triggers what. No model runs here; every decision carries its reason.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODES = ("quick", "research", "code")
WORD = re.compile(r"[a-z0-9][a-z0-9'_-]*")
QUESTION_STARTS = {
    "what", "why", "how", "is", "are", "does", "do", "did", "can", "could", "should", "would",
    "which", "when", "where", "who", "whose", "will", "was", "were", "has", "have",
}  # fmt: skip
SOFTENERS = {"just", "also", "quickly", "now", "then", "please"}
EXTENSIONS = r"(?:py|ts|tsx|js|jsx|json|toml|md|yml|yaml|sh|css|html|swift|rs|go|sql|txt|cfg|ini)"
FILE = re.compile(rf"^[\w./-]*\w\.{EXTENSIONS}$")
SNAKE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")


def _looks_like_code(token: str) -> bool:
    """A file name, a path of two named parts, or a snake_case name; not "1/f" or "and/or"."""
    token = token.strip("`'\"(),:;!?")
    if FILE.match(token) or SNAKE.match(token):
        return True
    parts = [p for p in token.split("/") if p]
    return (
        token.count("/") >= 1
        and len(parts) >= 2
        and sum(len(p) >= 3 for p in parts) >= 2
        and (token.count("/") >= 2 or token.endswith("/") or any(c in token for c in "._-"))
    )


def _prose(text: str) -> str:
    """`text` without its code-shaped tokens, so a path like research_mcp/rag.py adds no words."""
    return " ".join(t for t in text.split() if not _looks_like_code(t))


@dataclass(frozen=True)
class Rules:
    prefixes: dict[str, str]
    code_verbs: frozenset[str]
    task_verbs: frozenset[str]
    code_objects: frozenset[str]
    review: tuple[str, ...]
    request_leads: tuple[str, ...]
    research: tuple[str, ...]
    profiles: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Route:
    mode: str
    reason: str
    question: str
    matched: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "reason": self.reason, "matched": list(self.matched)}


def load_rules(default: Path, local: Path | None = None) -> Rules:
    """The default rules, with the local file's lists added and its prefixes and profiles merged."""
    data = tomllib.loads(default.read_text())
    extra = tomllib.loads(local.read_text()) if local is not None and local.is_file() else {}

    def words(section: str, key: str) -> list[str]:
        out = [str(w).lower() for w in data.get(section, {}).get(key, [])]
        out += [str(w).lower() for w in extra.get(section, {}).get(key, [])]
        return list(dict.fromkeys(out))

    merged = {**data.get("prefixes", {}), **extra.get("prefixes", {})}
    prefixes = {k.lower(): v for k, v in merged.items()}
    bad = sorted(v for v in prefixes.values() if v not in MODES)
    if bad:
        raise ValueError(f"routing prefixes name unknown modes: {', '.join(bad)}")
    profiles = {
        str(name): tuple(str(w).lower() for w in terms)
        for name, terms in {**data.get("profiles", {}), **extra.get("profiles", {})}.items()
    }
    return Rules(
        prefixes=prefixes,
        code_verbs=frozenset(words("code", "code_verbs")),
        task_verbs=frozenset(words("code", "task_verbs")),
        code_objects=frozenset(words("code", "code_objects")),
        review=tuple(words("code", "review")),
        request_leads=tuple(words("code", "request_leads")),
        research=tuple(words("research", "triggers")),
        profiles=profiles,
    )


def _words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def _has(text: str, phrase: str) -> bool:
    """`phrase` occurs in `text` as whole words."""
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _request_verbs(low: str, rules: Rules) -> list[str]:
    """Verbs the request asks for: an imperative opening, or one right after a request lead."""
    clauses = re.split(r"[.;!?\n]+|\band then\b|\bthen\b", low)
    verbs: list[str] = []
    for clause in clauses:
        words = _words(clause)
        if not words:
            continue
        starts = [0]
        for lead in rules.request_leads:
            n = len(_words(lead))
            starts += [i + n for i in range(len(words) - n + 1) if words[i : i + n] == _words(lead)]
        for i in starts:
            # Skip a few softeners: "please just quickly fix", "can you also add".
            j = i
            while j < len(words) and words[j] in SOFTENERS:
                j += 1
            asked = i > 0 or words[0] not in QUESTION_STARTS
            if asked and j < len(words) and words[j] in rules.code_verbs | rules.task_verbs:
                verbs.append(words[j])
    return list(dict.fromkeys(verbs))


def classify(question: str, rules: Rules, mode: str = "", repo: str = "") -> Route:
    """The cheapest mode that serves the request, and why."""
    text = question.strip()
    low = text.lower()
    if mode:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        return Route(mode, f"mode '{mode}' was asked for", text)
    for prefix, chosen in sorted(rules.prefixes.items(), key=lambda kv: -len(kv[0])):
        if low.startswith(prefix):
            rest = text[len(prefix) :].strip()
            return Route(chosen, f"the request starts with '{prefix}'", rest, (prefix,))

    reviews = [p for p in rules.review if _has(low, p)]
    if reviews:
        return Route("code", "a code review was asked for", text, tuple(reviews))
    prose = _prose(low)
    verbs = _request_verbs(prose, rules)
    shaped = [t for t in text.split() if _looks_like_code(t)]
    names_code = bool(repo) or bool(shaped) or any(w in rules.code_objects for w in _words(prose))
    code_verbs = [v for v in verbs if v in rules.code_verbs]
    if code_verbs:
        reason = f"a code task was asked for ('{code_verbs[0]}')"
        return Route("code", reason, text, tuple(code_verbs))
    task_verbs = [v for v in verbs if v in rules.task_verbs]
    if task_verbs and names_code:
        objects = [w for w in _words(prose) if w in rules.code_objects and w not in task_verbs]
        hint = objects[0] if objects else (shaped[0].strip("`'\"(),:;!?") if shaped else "the repo")
        return Route(
            "code",
            f"a task on code was asked for ('{task_verbs[0]}' … '{hint}')",
            text,
            tuple(task_verbs + objects[:1]),
        )
    research = [p for p in rules.research if _has(prose, p)]
    if research:
        return Route("research", f"it asks about evidence ('{research[0]}')", text, tuple(research))
    return Route("quick", "a question with no research or code trigger: one retrieval", text)


def profile_override(question: str, rules: Rules) -> tuple[str, str] | None:
    """The specialist the operator's own words pick, if any: (name, matched word)."""
    low = question.lower()
    for name, terms in rules.profiles.items():
        for term in terms:
            if _has(low, term):
                return name, term
    return None
