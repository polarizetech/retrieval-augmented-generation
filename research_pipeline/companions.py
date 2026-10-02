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

    "companions": {"datasets": {"command": "dataset-fetch-mcp"}}

To add one: write its `run` (call the tool, return plain records) and `render` (print them), and
register a `Companion` below. Nothing in the pipeline changes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .upstream import Upstream, UpstreamError, spec_for

SECTION = "companions"
Rank = Callable[[list[str]], Awaitable[list[float]]]  # texts -> closeness to the question, 0..1


@dataclass(frozen=True)
class Companion:
    key: str  # the name in the config, in the plan and in the run log
    title: str  # the heading of its section in the answer
    when: str  # for the planner: when requesting it serves the question
    ask: str  # for the planner: what each query should look like
    run: Callable[[Upstream, list[str], Rank], Awaitable[dict[str, Any]]]
    render: Callable[[dict[str, Any]], list[str]]
    max_queries: int = 2

    def limits(self, result: dict[str, Any]) -> list[str]:
        """Lines for the answer's limits: what the tool could not be asked or did not answer."""
        return list(result.get("limits", []))


# ---------------------------------------------------------------------------------- datasets
# dataset-fetch: discovery over open research datasets. Its contract (see its MCP instructions):
# `providers` lists catalogues and the search filters each takes; `search(provider, filters)`
# returns pinned refs; `describe(ref)` returns a card. `unavailable` never means "nothing exists".

TEXT_FILTERS = ("q", "query", "keywords")  # the free-text filter, under whichever name it has
# How a query is put to a catalogue whose free-text search also returns things that are not
# datasets. Zenodo holds papers, posters and software too, and a plain query returns mostly PDFs
# of articles (observed: six of six); its query language can ask for records typed as datasets.
QUERY_FORM = {"zenodo": "({query}) AND resource_type.type:dataset"}
PER_SEARCH = 5  # refs asked of one catalogue for one query
MAX_DESCRIBED = 12  # data cards read per run
MAX_LISTED = 6  # datasets shown in the answer
DOCUMENTS = frozenset({".pdf", ".doc", ".docx", ".ppt", ".pptx", ".md", ".txt", ".html", ".rtf"})


async def find_datasets(up: Upstream, queries: list[str], rank: Rank) -> dict[str, Any]:
    """Search every catalogue that takes a free-text query, read the cards of what was found."""
    providers = (await up.call("providers"))["providers"]
    searchable = [
        (p["name"], f)
        for p in providers
        if (f := next((x for x in TEXT_FILTERS if x in p.get("search_filters", [])), None))
    ]
    searches: list[dict[str, Any]] = []
    found: dict[str, str] = {}  # ref -> the query that found it first
    for query in queries:
        for provider, text_filter in searchable:
            row: dict[str, Any] = {"query": query, "provider": provider}
            try:
                asked = QUERY_FORM.get(provider, "{query}").format(query=query)
                data = await up.call(
                    "search", provider=provider, filters={text_filter: asked}, limit=PER_SEARCH
                )
            except UpstreamError as exc:
                searches.append(row | {"status": exc.code, "error": str(exc)[:200]})
                continue
            refs = [r["ref"] for r in data.get("results", [])][:PER_SEARCH]
            searches.append(row | {"status": "ok", "n": len(refs)})
            for ref in refs:
                found.setdefault(ref, query)

    items: list[dict[str, Any]] = []
    for ref, query in list(found.items())[:MAX_DESCRIBED]:
        try:
            card = await up.call("describe", ref=ref)
        except UpstreamError:
            continue
        files = card.get("files") or {}
        extensions = files.get("extensions") or {}
        items.append(
            {
                "ref": ref,
                "provider": card.get("provider"),
                "title": card.get("title"),
                "doi": card.get("doi"),
                "license": card.get("license"),
                "creators": (card.get("creators") or [])[:3],
                "total_bytes": card.get("total_bytes"),
                "n_files": files.get("n_files"),
                "extensions": extensions,
                # None when the card lists no files: unknown, which is not "documents only".
                "data_files": bool(set(extensions) - DOCUMENTS) if extensions else None,
                "found_by": query,
            }
        )
    scores = await rank([str(i["title"] or "") for i in items]) if items else []
    for item, score in zip(items, scores, strict=True):
        item["relevance"] = round(float(score), 4)
    # A record whose files are all documents describes data rather than holding it: counted, not
    # listed. The rest are ordered by closeness to the question.
    documents = [i for i in items if i["data_files"] is False]
    items = sorted(
        (i for i in items if i["data_files"] is not False), key=lambda i: -i["relevance"]
    )

    silent = sorted({s["provider"] for s in searches if s["status"] != "ok"})
    limits = []
    if silent:
        limits.append(
            f"Dataset catalogues that did not answer at least once: {', '.join(silent)}. "
            "Their silence is not an absence of datasets."
        )
    if len(found) > MAX_DESCRIBED:
        limits.append(
            f"{len(found) - MAX_DESCRIBED} further dataset record(s) were found but not read."
        )
    return {
        "queries": queries,
        "catalogues": [name for name, _ in searchable],
        "searches": searches,
        "found": len(found),
        "items": items[:MAX_LISTED],
        "documents_only": len(documents),
        "limits": limits,
    }


def _size(n: int | None) -> str:
    if not n:
        return "size not stated"
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} bytes"


def render_datasets(result: dict[str, Any]) -> list[str]:
    asked = "; ".join(f"“{q}”" for q in result["queries"])
    out = [
        f"Open dataset catalogues ({', '.join(result['catalogues']) or 'none available'}) were "
        f"searched for: {asked}. Each entry is the catalogue's own record. Nothing was downloaded, "
        "opened or assessed: whether a dataset can answer the question is for the reader to check.",
        "",
    ]
    if not result["items"]:
        out.append("- No dataset record with data files was found by these searches.")
    for i in result["items"]:
        kinds = ", ".join(f"{ext} ×{n}" for ext, n in sorted(i["extensions"].items())[:5])
        files = (
            f"{i['n_files']} file(s) ({kinds}), {_size(i['total_bytes'])}"
            if kinds
            else "files not listed by the catalogue (access may be restricted)"
        )
        stated = i["license"] if i["license"] and i["license"].lower() != "unknown" else None
        licence = stated or "no licence stated, which is not permission to reuse"
        who = ", ".join(i["creators"]) + (" et al." if len(i["creators"]) == 3 else "")
        out.append(
            f"- **{i['title'] or i['ref']}** — {i['provider']}; {licence}; {files}"
            + (f"; {who}" if who else "")
            + (f"; doi:{i['doi']}" if i["doi"] else "")
            + f". Ref `{i['ref']}`."
        )
    if result.get("documents_only"):
        out.append(
            f"- {result['documents_only']} further record(s) hold only documents (a PDF of an "
            "article, say) and are not listed: they describe data rather than hold it."
        )
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

REGISTRY: dict[str, Companion] = {c.key: c for c in (DATASETS,)}


def configured(gateway_config: Path) -> dict[str, Companion]:
    """The registered companions the gateway config has a server for."""
    return {k: c for k, c in REGISTRY.items() if spec_for(gateway_config, k, SECTION)}


def connect(gateway_config: Path, key: str) -> Upstream:
    return Upstream(gateway_config, key, SECTION)
