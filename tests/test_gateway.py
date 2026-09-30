"""The gateway end to end over an in-memory MCP session, with a real stdio upstream."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent
from paper_fetch import Library, LocalStore, OpenAlex
from paper_fetch.passages import PassageIndex

from research_mcp import gateway
from research_pipeline.papers import PaperLibrary, PapersError
from tests.conftest import OfflineHttp, hold, hold_w1

OPEN_TEXT = "Heart rate variability rose with paced breathing in 30 adults. " * 20
DEAD = "http://127.0.0.1:9"  # every request through this proxy is refused at once


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The real paper-fetch MCP server as the `papers` upstream, over a local store that holds
    W1 (indexed), an open paper not yet indexed, and a paper with no open copy. Nothing can reach
    the network: search has one provider and every proxy points at a closed port."""
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PIPELINE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    store_dir, index_file = tmp_path / "papers", tmp_path / "passages.sqlite"
    lib = Library(
        LocalStore(store_dir),
        OpenAlex(http=OfflineHttp(), api_key="", email=""),  # type: ignore[arg-type]
        providers=[],
        search_providers=[],
    )
    lib.passages = PassageIndex(index_file)  # lexical, as the upstream (no embedding model) is
    hold_w1(lib)
    lib.index_works()
    hold(lib, "W2", OPEN_TEXT, doi="10.1000/open", title="Paced breathing and HRV")
    hold(lib, "W3", "", doi="10.1000/closed", title="A closed trial", full_text=False)
    lib.passages.close()
    env = {
        "PAPER_FETCH_STORE": "local",
        "PAPER_FETCH_DATA_DIR": str(store_dir),
        "PAPER_FETCH_INDEX": str(index_file),
        "PAPER_FETCH_ENV_DIR": str(tmp_path / "no-env"),
        "PAPER_FETCH_PROFILE_DIR": str(tmp_path / "no-profiles"),
        "PAPER_FETCH_SEARCH_PROVIDERS": "doaj",
        "PAPER_FETCH_WEB_FALLBACK": "0",
        "PAPER_FETCH_EMBED_MODEL": "",
        "PAPER_FETCH_RERANK_MODEL": "",
        "HTTP_PROXY": DEAD,
        "HTTPS_PROXY": DEAD,
        "http_proxy": DEAD,
        "https_proxy": DEAD,
        "NO_PROXY": "",
        "no_proxy": "",
    }
    path = tmp_path / "gateway.json"
    path.write_text(
        json.dumps(
            {
                "upstreams": {
                    "papers": {
                        "command": sys.executable,
                        "args": ["-c", "from paper_fetch.mcp_server import main; main()"],
                        "env": env,
                        "write_tools": ["fetch"],
                        "open_world_tools": ["search", "fetch"],
                    },
                    "broken": {"command": str(tmp_path / "does-not-exist")},
                    "disabled": {"command": "unused", "enabled": False},
                }
            }
        )
    )
    return path


Session = Callable[[ClientSession], Coroutine[Any, Any, None]]


async def with_session(config: Path, tmp_path: Path, body: Session) -> None:
    server, _ = gateway.build_server(config, "127.0.0.1", 0, tmp_path / "token", http=False)
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        await body(session)


def payload(result: CallToolResult) -> dict[str, Any]:
    if result.structuredContent is not None:
        return result.structuredContent
    first = result.content[0]
    assert isinstance(first, TextContent)
    return json.loads(first.text)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_tools_are_listed_with_annotations(config: Path, tmp_path: Path) -> None:
    async def body(session: ClientSession) -> None:
        tools = {t.name: t for t in (await session.list_tools()).tools}
        assert {
            "gateway_status",
            "web__search",
            "rag__search",
            "rag__retrieve_evidence",
            "rag__check_citations",
            "rag__save_report",
            "rag__index_paper",
            "math__lookup",
            "math__bionumber",
            "papers__search",
            "papers__fetch",
        } <= set(tools)
        fetch = tools["papers__fetch"].annotations
        assert fetch is not None
        assert fetch.readOnlyHint is False
        assert fetch.openWorldHint is True

        library_tools = {n for n in tools if n.startswith("papers__")}
        assert {"papers__retrieve", "papers__recall", "papers__profiles"} <= library_tools
        status = payload(await session.call_tool("gateway_status", {}))
        assert status["connected"] == {"papers": len(library_tools)}
        assert set(status["errors"]) == {"broken"}
        assert status["total_tools"] == len(library_tools) + 9

    await with_session(config, tmp_path, body)


@pytest.mark.anyio
async def test_a_server_section_names_the_instance_and_can_drop_the_builtin_tools(
    config: Path, tmp_path: Path
) -> None:
    cfg = json.loads(config.read_text())
    cfg["server"] = {
        "name": "second-instance",
        "instructions": "Only the upstreams.",
        "builtin_tools": False,
    }
    other = tmp_path / "second.json"
    other.write_text(json.dumps(cfg))
    server, _ = gateway.build_server(other, "127.0.0.1", 0, tmp_path / "token", http=False)
    assert server.name == "second-instance"
    assert server.instructions == "Only the upstreams."

    async def body(session: ClientSession) -> None:
        names = {t.name for t in (await session.list_tools()).tools}
        assert "gateway_status" in names
        assert "papers__search" in names
        assert not any(n == "web__search" or n.startswith(("rag__", "math__")) for n in names)
        refused = await session.call_tool("rag__search", {"query": "x"})
        assert refused.isError
        status = payload(await session.call_tool("gateway_status", {}))
        assert status["total_tools"] == len({n for n in names if n.startswith("papers__")}) + 1

    await with_session(other, tmp_path, body)


@pytest.mark.anyio
async def test_rag_tools_round_trip(config: Path, tmp_path: Path) -> None:
    async def body(session: ClientSession) -> None:
        found = payload(
            await session.call_tool(
                "rag__retrieve_evidence", {"query": "lowered systolic pressure"}
            )
        )
        assert found["retrieval"] == "lexical"  # the upstream library has no embedding model
        hit = found["results"][0]
        claim = {
            "text": "Aerobic training lowered systolic pressure by 5 mmHg.",
            "evidence_ids": [hit["evidence_id"]],
            "quotes": {hit["evidence_id"]: hit["text"].split(". ")[0] + "."},
        }
        checked = payload(await session.call_tool("rag__check_citations", {"claims": [claim]}))
        assert checked["valid"] is True
        saved = payload(
            await session.call_tool(
                "rag__save_report",
                {
                    "title": "Report",
                    "markdown": "# R",
                    "evidence_ids": [hit["evidence_id"]],
                    "claims": [claim],
                },
            )
        )
        assert saved["claims_checked"] is True

        refused = await session.call_tool(
            "rag__save_report",
            {"title": "Report", "markdown": "# R", "evidence_ids": ["not-an-id"]},
        )
        assert refused.isError

    await with_session(config, tmp_path, body)


@pytest.mark.anyio
async def test_search_is_forwarded_and_errors_are_reported(config: Path, tmp_path: Path) -> None:
    async def body(session: ClientSession) -> None:
        found = payload(await session.call_tool("rag__search", {"query": "exercise"}))
        assert found["ok"] is True
        # The one provider cannot be reached: reported by name, never as "no results".
        assert found["data"]["providers"]["doaj"]["status"] == "unavailable"
        assert found["data"]["search_id"]  # and the library remembered the search
        web = await session.call_tool("web__search", {"query": "x"})
        assert web.isError
        unknown = await session.call_tool("nope__tool", {})
        assert unknown.isError

    await with_session(config, tmp_path, body)


@pytest.mark.anyio
async def test_the_pipeline_client_speaks_the_library_contract(config: Path) -> None:
    async with PaperLibrary(config, "papers") as library:
        found = await library.search("exercise")
        assert found["providers"]["doaj"]["status"] == "unavailable"
        held = await library.passages("W1", 0, 1)
        assert held[0].id == "W1#p0"
        assert held[0].paper["title"] == "Training and blood pressure"
        got = await library.retrieve(["lowered systolic pressure"], limit=1)
        assert got["results"][0]["passages"][0]["work"] == "W1"
        assert (await library.status())["passages"]["papers"] == 1
        assert await library.profile("no-such-field") is None
        assert (await library.profile("cardiovascular") or {})["terms"]
        with pytest.raises(PapersError) as refused:
            await library.passages("W999")
        assert refused.value.code == "not_found"


def test_a_missing_upstream_is_a_clear_error(config: Path) -> None:
    with pytest.raises(PapersError, match="no 'absent' upstream"):
        PaperLibrary(config, "absent")


def test_searxng_url_comes_from_the_environment_first(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = {"web_search": {"searxng_url": "http://config.example/"}}
    assert gateway._load_searxng(configured) == "http://config.example"
    monkeypatch.setenv("SEARXNG_URL", "http://env.example/")
    assert gateway._load_searxng(configured) == "http://env.example"


def test_the_token_file_is_created_private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    token_file = tmp_path / "token"
    token = gateway._token(token_file)
    assert len(token) >= 40
    assert token_file.stat().st_mode & 0o777 == 0o600
    assert gateway._token(token_file) == token


@pytest.mark.anyio
async def test_index_paper_makes_a_fetched_paper_citable(config: Path, tmp_path: Path) -> None:
    async def body(session: ClientSession) -> None:
        before = payload(
            await session.call_tool("rag__retrieve_evidence", {"query": "paced breathing"})
        )
        assert all(r["work"] != "W2" for r in before["results"])

        done = payload(
            await session.call_tool(
                "rag__index_paper", {"identifiers": ["10.1000/open", "10.1000/closed"]}
            )
        )
        status = {r["identifier"]: r["status"] for r in done["results"]}
        assert status == {"10.1000/open": "indexed", "10.1000/closed": "no_open_full_text"}
        assert done["index"]["papers"] == 2

        after = payload(
            await session.call_tool("rag__retrieve_evidence", {"query": "paced breathing"})
        )
        assert after["results"][0]["work"] == "W2"

        again = payload(
            await session.call_tool("rag__index_paper", {"identifiers": ["10.1000/open"]})
        )
        assert again["results"][0]["status"] == "already_indexed"

    await with_session(config, tmp_path, body)
