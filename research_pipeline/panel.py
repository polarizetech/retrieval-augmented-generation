"""The model panel: other models read the same passages as the run's model and write their own
claims, so a research answer can show where models agree and where they differ.

Discovery, fetching and passage selection run once. Each panel model then reads every passage
the run's model read, in the same order, and drafts its own claims from what it extracted. Every
model's claims go through the same local verifier, so the comparison is between readings of one
evidence set, checked one way. The run's own answer is unchanged: the panel adds a section.

PIPELINE_PANEL names the panel by hub provider (config/hub.default.toml, [providers.<name>]),
for example "openai,gemini". A member whose key is missing, or whose calls fail, is named in the
answer and the run carries on without it.

The comparison is computed, never written by a model:

- **readings**: for each passage, the direction each model read in it (affirms, denies, mixed,
  neutral) or that it found it not relevant; passages where two models read opposite directions
  are listed;
- **claims**: per sub-question, each model's verified claims, and for each one which other models
  reached a verified claim resting on at least one of the same passages.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .config import Settings
from .llm import Endpoint
from .usage import _read_key, load_config

if TYPE_CHECKING:
    from .pipeline import State

KEPT = ("SUPPORTED", "DISPUTED")
OPPOSITE = {("affirms", "denies"), ("denies", "affirms")}


@dataclass
class Member:
    name: str  # the hub provider name
    label: str
    model: str
    url: str
    retention: str = ""  # what the provider keeps, said in the answer's limits
    llm: Endpoint | None = None
    skipped: str = ""  # why this member did not run, when it did not
    error: str = ""  # why its reading failed part way, when it did
    state: State | None = field(default=None, repr=False)


def members(settings: Settings, config: dict[str, Any] | None = None) -> list[Member]:
    """The panel named by PIPELINE_PANEL, each with its endpoint or the reason it can't run."""
    names = [n.strip() for n in settings.panel.split(",") if n.strip()]
    if not names:
        return []
    providers = (config if config is not None else load_config()).get("providers", {})
    out: list[Member] = []
    for name in dict.fromkeys(names):
        spec = providers.get(name)
        if spec is None:
            out.append(Member(name, name, "", "", skipped="no such provider in the hub config"))
            continue
        member = Member(
            name,
            str(spec.get("label", name)),
            str(spec.get("model", "")),
            str(spec.get("url", "")).rstrip("/"),
            retention=str(spec.get("retention", "")),
        )
        if spec.get("kind") != "openai" or not member.model or not member.url:
            member.skipped = "a panel member must be an OpenAI-compatible API with a model"
        elif not (key := _read_key(str(spec.get("key_file", "")))):
            member.skipped = f"no API key in {spec.get('key_file') or 'its key_file'}"
        else:
            member.llm = Endpoint(
                replace(
                    settings,
                    endpoint_url=member.url,
                    endpoint_model=member.model,
                    endpoint_key=key,
                    endpoint_zdr=bool(spec.get("zdr", settings.endpoint_zdr)),
                )
            )
        out.append(member)
    return out


def _readings(state: State) -> dict[str, str]:
    """Passage id -> what this model read in it: a direction, or "not relevant"."""
    out: dict[str, str] = {}
    read = {pid for ids in state.seen.values() for pid in ids}
    for pid in read:
        out[pid] = "not relevant"
    for e in state.evidence.values():
        out[e.passage_id] = e.direction
    dropped = {d["passage_id"] for d in state.dropped_passages}
    for pid in dropped:
        out.setdefault(pid, "unread")
        if out[pid] == "not relevant":
            out[pid] = "unread"
    return out


def _kept(state: State) -> list[dict[str, Any]]:
    rows = []
    for c in state.claims:
        if c.verdict not in KEPT:
            continue
        support = c.supported_by or c.evidence_ids
        rows.append(
            {
                "id": c.id,
                "subquestion": c.subquestion,
                "text": c.text,
                "verdict": c.verdict,
                "label": c.label,
                "passages": sorted({state.evidence[e].passage_id for e in support}),
                "directions": sorted({state.evidence[e].direction for e in support}),
            }
        )
    return rows


def compare(models: dict[str, State]) -> dict[str, Any]:
    """Readings and claims, model by model, from the states alone. No model is called."""
    readings = {name: _readings(st) for name, st in models.items()}
    kept = {name: _kept(st) for name, st in models.items()}
    passages = sorted({pid for r in readings.values() for pid in r})
    disagreements = []
    for pid in passages:
        seen = {name: r[pid] for name, r in readings.items() if pid in r}
        dirs = set(seen.values())
        if any((a, b) in OPPOSITE for a in dirs for b in dirs):
            disagreements.append({"passage_id": pid, "readings": seen})
    for name, rows in kept.items():
        for row in rows:
            row["also_reached_by"] = sorted(
                other
                for other, others in kept.items()
                if other != name
                and any(
                    o["subquestion"] == row["subquestion"]
                    and set(o["passages"]) & set(row["passages"])
                    and set(o["directions"]) & set(row["directions"])
                    for o in others
                )
            )
    totals = {
        name: {
            "passages_read": len(readings[name]),
            "relevant": sum(v not in ("not relevant", "unread") for v in readings[name].values()),
            "claims": len(st.claims),
            "kept": len(kept[name]),
        }
        for name, st in models.items()
    }
    return {
        "totals": totals,
        "claims": kept,
        "opposite_readings": disagreements,
        "agreeing_passages": sum(
            1
            for pid in passages
            if len({r[pid] for r in readings.values() if pid in r}) == 1
            and sum(pid in r for r in readings.values()) > 1
        ),
        "passages": len(passages),
    }


def render(result: dict[str, Any], labels: dict[str, str]) -> str:
    """The answer's "Model comparison" section, from `compare` and each member's status."""
    lines: list[str] = []
    totals = result.get("totals", {})
    if totals:
        lines.append(
            "Each model read the same passages and wrote its own claims; every claim was checked "
            "by the same local verifier. Nothing in this section was written by a model."
        )
        lines.append("")
        lines.append("| Model | Passages read | Relevant | Claims | Verified |")
        lines.append("|---|---|---|---|---|")
        for name, t in totals.items():
            lines.append(
                f"| {labels.get(name, name)} | {t['passages_read']} | {t['relevant']} | "
                f"{t['claims']} | {t['kept']} |"
            )
        if result.get("passages"):
            lines.append("")
            lines.append(
                f"{result['agreeing_passages']} of {result['passages']} passages were read the "
                "same way by every model that read them."
            )
    opposite = result.get("opposite_readings", [])
    if opposite:
        lines.append("")
        lines.append("**Passages read in opposite directions**")
        for row in opposite[:8]:
            seen = "; ".join(f"{labels.get(n, n)}: {d}" for n, d in row["readings"].items())
            lines.append(f"- `{row['passage_id']}`: {seen}")
        if len(opposite) > 8:
            lines.append(f"- and {len(opposite) - 8} more (run log, `panel`)")
    claims = result.get("claims", {})
    for name, rows in claims.items():
        if not rows:
            continue
        lines.append("")
        lines.append(f"**{labels.get(name, name)}: verified claims**")
        for row in rows:
            others = ", ".join(labels.get(str(o), str(o)) for o in row["also_reached_by"])
            agree = f" (also reached by {others})" if others else " (no other model reached this)"
            lines.append(f"- [{row['subquestion']}] {row['text']}{agree}")
    for status in result.get("not_run", []):
        lines.append("")
        lines.append(f"- {status}")
    return "\n".join(lines).strip()
