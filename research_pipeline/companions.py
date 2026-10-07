"""Companion tools: other programs a run may loop in when the question calls for one.

The pipeline answers from papers. Some questions are also served by something papers cannot give:
a list of open datasets that could test the claim, for instance. A companion is a tool that
provides that, declared here so the pipeline can use it the way it uses everything else:

  1. the planner is told what each configured companion is for, and may request it, with
     queries, in the plan it already writes (one schema-constrained field; usually left empty);
  2. code calls the tool, over MCP, exactly as requested and exactly as bounded here;
  3. the answer prints what the tool's own records say. No model writes a word of that section.

So "intelligently" means the model decides *whether* a companion is relevant and *what to ask it*,
and nothing else. A companion never feeds the evidence table: its section is a pointer for the
reader, not support for a claim.

A companion is configured in the gateway config, apart from the federated upstreams (its tools
are not offered to MCP clients through the gateway):

    "companions": {"datasets": {"command": "dataset-fetch-mcp"},
                   "patents": {"command": "patent-fetch", "args": ["--json"]}}

A companion is reached the way its own repository offers: an MCP server (`transport="mcp"`, an
`upstream.Upstream`) or a command line that prints JSON (`transport="command"`, an
`upstream.Command`).

To add one: write its `run` (call the tool, return plain records) and `render` (print them), and
register a `Companion` below. Nothing in the pipeline changes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .upstream import Command, Upstream, UpstreamError, spec_for

SECTION = "companions"
Rank = Callable[[list[str]], Awaitable[list[float]]]  # texts -> closeness to the question, 0..1


@dataclass(frozen=True)
class Companion:
    key: str  # the name in the config, in the plan and in the run log
    title: str  # the heading of its section in the answer
    when: str  # for the planner: when requesting it serves the question
    ask: str  # for the planner: what each query should look like
    run: Callable[[Any, list[str], Rank], Awaitable[dict[str, Any]]]  # Any: its transport's client
    render: Callable[[dict[str, Any]], list[str]]
    max_queries: int = 2
    transport: str = "mcp"  # "mcp": an Upstream; "command": a Command

    def limits(self, result: dict[str, Any]) -> list[str]:
        """Lines for the answer's limits: what the tool could not be asked or did not answer."""
        return list(result.get("limits", []))


# ---------------------------------------------------------------------------------- datasets
# dataset-fetch's `recommend(topic, hints, limit)`: it decides which of its catalogues to ask from
# each one's declared scope, phrases the query in each catalogue's own terms, reads the records,
# sets apart those that hold only documents, and ranks the rest by the topic's terms, giving the
# reason. Every catalogue gets a status: ok, unavailable, skipped (and why) or error. This module
# only asks it, once per query the planner wrote, and prints what it returned.

MAX_LISTED = 6  # datasets shown in the answer
NOT_ANSWERED = ("unavailable", "error")  # statuses that are silence, never "nothing found"


async def find_datasets(up: Upstream, queries: list[str], rank: Rank) -> dict[str, Any]:
    """Ask dataset-fetch's `recommend` for each query; merge the ranked cards by reference.

    `rank` is unused: dataset-fetch ranks, and says how.
    """
    del rank
    cards: dict[str, dict[str, Any]] = {}
    providers: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    more = 0
    failed: list[str] = []
    for query in queries:
        try:
            got = await up.call("recommend", topic=query, limit=MAX_LISTED)
        except UpstreamError as exc:
            # One query the tool refused (no content words, say) does not discard the others.
            failed.append(f"“{query}” ({exc.code}: {str(exc)[:120]})")
            continue
        for card in got.get("results", []):
            seen = cards.get(card["ref"])
            if seen is None or card.get("score", 0) > seen.get("score", 0):
                cards[card["ref"]] = {**card, "asked_as": query}
        for doc in got.get("documents_only", []):
            documents.setdefault(doc["ref"], doc)
        more += int(got.get("more_results") or 0)
        for name, row in (got.get("providers") or {}).items():
            # A catalogue that answered any query answered; one silent on every query did not.
            if providers.get(name, {}).get("status") != "ok":
                providers[name] = {"status": row.get("status"), "why": row.get("why")}

    # Known to hold data first, then unverified; each by dataset-fetch's own score.
    items = sorted(
        cards.values(), key=lambda c: (c.get("data_files") is not True, -c.get("score", 0))
    )
    asked = sorted(n for n, r in providers.items() if r["status"] not in ("skipped", None))
    silent = sorted(n for n, r in providers.items() if r["status"] in NOT_ANSWERED)
    limits = []
    if failed:
        if len(failed) == len(queries):
            raise UpstreamError("recommend answered none of the queries: " + "; ".join(failed))
        limits.append("dataset-fetch did not answer for: " + "; ".join(failed) + ".")
    if silent:
        limits.append(
            f"Dataset catalogues that did not answer: {', '.join(silent)}. Their silence is "
            "not an absence of datasets."
        )
    return {
        "queries": queries,
        "asked": asked,
        "providers": providers,
        "items": items[:MAX_LISTED],
        "more": more + max(0, len(items) - MAX_LISTED),
        "documents_only": len(documents),
        "limits": limits,
    }


def _size(n: int | None) -> str:
    if not n:
        return "size not stated"
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} bytes"


def _files(card: dict[str, Any]) -> str:
    files = card.get("files") or {}
    n = files.get("n_files")
    if not n:
        return "files not listed by the catalogue"
    kinds = files.get("extensions") or {}
    shown = ", ".join(f"{ext} ×{k}" for ext, k in sorted(kinds.items())[:4])
    return (
        f"{n:,} file(s)" + (f" ({shown})" if shown else "") + f", {_size(card.get('total_bytes'))}"
    )


def render_datasets(result: dict[str, Any]) -> list[str]:
    asked = "; ".join(f"“{q}”" for q in result["queries"])
    out = [
        f"dataset-fetch was asked for datasets on: {asked}. It chose the catalogues whose declared "
        f"scope covers the topic ({', '.join(result['asked']) or 'none'}), and ranked what they "
        "returned by the topic's terms. Each entry is the catalogue's own record. Nothing was "
        "downloaded, opened or assessed: whether a dataset can answer the question is for the "
        "reader to check.",
        "",
    ]
    if not result["items"]:
        out.append("- No dataset record with data files was found.")
    for c in result["items"]:
        licence = c.get("license")
        if not licence or str(licence).lower() == "unknown":
            licence = "no licence stated, which is not permission to reuse"
        creators = list(c.get("creators") or [])
        who = ", ".join(creators[:3]) + (" et al." if len(creators) > 3 else "")
        doi = str(c.get("doi") or "").removeprefix("doi:").removeprefix("https://doi.org/")
        facts = [
            f"{c.get('provider')}" + (f" ({c['kind']})" if c.get("kind") else ""),
            licence,
            _files(c),
        ]
        if isinstance(c.get("n_subjects"), int):
            facts.append(f"{c['n_subjects']} subjects")
        if c.get("modalities"):
            facts.append(", ".join(c["modalities"]))
        if who:
            facts.append(who)
        if doi:
            facts.append(f"doi:{doi}")
        line = f"- **{c.get('title') or c['ref']}** — {'; '.join(facts)}. Ref `{c['ref']}`."
        if c.get("score_reason"):
            line += f" *Why listed: {c['score_reason']}.*"
        if c.get("data_files") is not True:
            line += " *Whether it holds data files could not be verified.*"
        out.append(line)
    tail = []
    if result.get("more"):
        tail.append(f"{result['more']} further record(s) ranked below these")
    if result.get("documents_only"):
        tail.append(
            f"{result['documents_only']} record(s) hold only documents (an article PDF, a "
            "protocol) and describe data rather than hold it"
        )
    if tail:
        out.append("- Not listed: " + "; ".join(tail) + ".")
    return out


DATASETS = Companion(
    key="datasets",
    title="Datasets that could test this",
    when=(
        "the question is about something measured (a physiological signal, a trial outcome, a "
        "recording, a survey), so a reader could test or re-analyse it with openly available "
        "research data"
    ),
    ask=(
        "1-2 short keyword queries, each naming the measurement and the population the way a "
        "dataset would be titled"
    ),
    run=find_datasets,
    render=render_datasets,
)

# ---------------------------------------------------------------------------------- patents
# patent-fetch ships a command line and no MCP server: `patent-fetch --json search "<words>"`
# asks its patent services, merges what they answer by publication number, and says by name
# which answered. It does not rank, so the hits are ordered here by how close each title sits to
# the question (the paper library's relevance, the same measure that orders fetches). Nothing is
# fetched: this lists what a search matched, from the services' own records.

PATENTS_LISTED = 6
PATENTS_PER_QUERY = 8


async def find_patents(tool: Command, queries: list[str], rank: Rank) -> dict[str, Any]:
    """Run patent-fetch's search for each query; merge the hits by publication number."""
    hits: dict[str, dict[str, Any]] = {}
    providers: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for query in queries:
        try:
            got = await tool.run("search", query, "-n", str(PATENTS_PER_QUERY))
        except UpstreamError as exc:
            failed.append(f"“{query}” ({str(exc)[:120]})")
            continue
        for hit in got.get("hits", []):
            hits.setdefault(hit["id"], {**hit, "asked_as": query})
        for row in got.get("providers", []):
            # A service that answered any query answered; one silent on every query did not.
            if providers.get(row["provider"], {}).get("status") not in ("ok", "cache"):
                providers[row["provider"]] = {
                    "status": row.get("status"),
                    "why": row.get("detail"),
                }
    if failed and len(failed) == len(queries):
        raise UpstreamError("patent-fetch answered none of the queries: " + "; ".join(failed))

    items = list(hits.values())
    scores = await rank([h.get("title") or "" for h in items]) if items else []
    for hit, score in zip(items, scores, strict=True):
        hit["relevance"] = round(float(score), 4)
    items.sort(key=lambda h: (-h["relevance"], -(h.get("year") or 0)))

    answered = sorted(n for n, r in providers.items() if r["status"] in ("ok", "cache"))
    no_key = sorted(n for n, r in providers.items() if r["status"] == "skipped")
    silent = sorted(n for n, r in providers.items() if r["status"] in NOT_ANSWERED)
    limits = []
    if failed:
        limits.append("patent-fetch did not answer for: " + "; ".join(failed) + ".")
    if not answered:
        limits.append("No patent service answered, so the patent search says nothing either way.")
    elif answered == ["europepmc"]:
        limits.append(
            "The only patent service that answered was Europe PMC's archive: life-science "
            "patents only, and none after 2012. Later patents, and other fields, were not searched."
        )
    if no_key:
        limits.append(f"Patent services not asked (no account configured): {', '.join(no_key)}.")
    if silent:
        limits.append(
            f"Patent services that did not answer: {', '.join(silent)}. Their silence is not an "
            "absence of patents."
        )
    return {
        "queries": queries,
        "answered": answered,
        "providers": providers,
        "items": [
            {
                k: h.get(k)
                for k in (
                    "id",
                    "kind",
                    "title",
                    "year",
                    "applicants",
                    "inventors",
                    "classifications",
                    "status",
                    "url",
                    "found_by",
                    "asked_as",
                    "relevance",
                )
            }
            for h in items[:PATENTS_LISTED]
        ],
        "more": max(0, len(items) - PATENTS_LISTED),
        "limits": limits,
    }


def render_patents(result: dict[str, Any]) -> list[str]:
    asked = "; ".join(f"“{q}”" for q in result["queries"])
    out = [
        f"patent-fetch searched for: {asked}. Services that answered: "
        f"{', '.join(result['answered']) or 'none'}. Each entry is the service's own record, "
        "ordered by how close its title sits to the question; none was opened or read. This is "
        "what a keyword search matched, not a legal opinion: it does not say whether a patent is "
        "valid, in force, or covers anything, and finding none is not evidence that none exists.",
        "",
    ]
    if not result["items"]:
        out.append("- No patent record was returned.")
    for h in result["items"]:
        who = list(h.get("applicants") or []) or list(h.get("inventors") or [])
        named = ", ".join(who[:3]) + (" et al." if len(who) > 3 else "")
        facts = [str(h["year"]) if h.get("year") else "year not stated"]
        if named:
            facts.append(named)
        if h.get("classifications"):
            facts.append("classes " + ", ".join(h["classifications"][:3]))
        if h.get("status"):
            facts.append(f"status as the service reports it: {h['status']}")
        line = f"- **{h.get('title') or h['id']}** — `{h['id']}`; {'; '.join(facts)}."
        if h.get("url"):
            line += f" {h['url']}"
        out.append(line)
    if result.get("more"):
        out.append(f"- Not listed: {result['more']} further record(s) ranked below these.")
    return out


PATENTS = Companion(
    key="patents",
    title="Existing patents on this",
    when=(
        "the question is about a device, an apparatus, a method of treatment or measurement, a "
        "compound or another invention, so a reader would want to know what has already been "
        "patented"
    ),
    ask=(
        "1-2 short keyword queries, each naming the invention and what it does the way a patent "
        "title would"
    ),
    run=find_patents,
    render=render_patents,
    transport="command",
)

REGISTRY: dict[str, Companion] = {c.key: c for c in (DATASETS, PATENTS)}


def configured(gateway_config: Path) -> dict[str, Companion]:
    """The registered companions the gateway config has a server for."""
    return {k: c for k, c in REGISTRY.items() if spec_for(gateway_config, k, SECTION)}


def connect(gateway_config: Path, key: str) -> Upstream | Command:
    """The client for a configured companion, by the transport it declares."""
    if REGISTRY[key].transport == "command":
        return Command(gateway_config, key, SECTION)
    return Upstream(gateway_config, key, SECTION)
