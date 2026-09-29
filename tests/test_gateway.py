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

from research_mcp import gateway
from research_pipeline.config import Settings
from research_pipeline.index import PassageIndex
from research_pipeline.papers import PaperLibrary, PapersError
from tests.conftest import PAPER, fake_embed

FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_papers_server.py"


class OfflineOllama:
    def __init__(self, _: Settings) -> None:
        pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        return fake_embed(texts)


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PIPELINE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("PIPELINE_EMBEDDING_MODEL", "fake-embed")
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    monkeypatch.setattr(gateway, "Ollama", OfflineOllama)
    settings = Settings()
    PassageIndex(settings.index_path, "fake-embed").add(
        {"work": "W1", "title": "Training and blood pressure"}, PAPER, fake_embed
    )
    path = tmp_path / "gateway.json"
    path.write_text(
        json.dumps(
            {
                "upstreams": {
                    "papers": {
                        "command": sys.executable,
                        "args": [str(FAKE_SERVER)],
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
            "papers__search",
            "papers__fetch",
        } <= set(tools)
        fetch = tools["papers__fetch"].annotations
        assert fetch is not None
        assert fetch.readOnlyHint is False
        assert fetch.openWorldHint is True

        status = payload(await session.call_tool("gateway_status", {}))
        assert status["connected"] == {"papers": 3}
        assert set(status["errors"]) == {"broken"}
        assert status["total_tools"] == 10

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
        assert not any(n == "web__search" or n.startswith("rag__") for n in names)
        refused = await session.call_tool("rag__search", {"query": "x"})
        assert refused.isError
        status = payload(await session.call_tool("gateway_status", {}))
        assert status["total_tools"] == 4

    await with_session(other, tmp_path, body)


@pytest.mark.anyio
async def test_rag_tools_round_trip(config: Path, tmp_path: Path) -> None:
    async def body(session: ClientSession) -> None:
        found = payload(
            await session.call_tool(
                "rag__retrieve_evidence", {"query": "lowered systolic pressure"}
            )
        )
        assert found["retrieval"] == "hybrid"
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
        assert found["data"]["hits"][0]["title"] == "A paper about exercise"
        web = await session.call_tool("web__search", {"query": "x"})
        assert web.isError
        unknown = await session.call_tool("nope__tool", {})
        assert unknown.isError

    await with_session(config, tmp_path, body)


@pytest.mark.anyio
async def test_the_pipeline_client_speaks_the_library_contract(config: Path) -> None:
    async with PaperLibrary(config, "papers") as library:
        found = await library.search("exercise")
        assert found["providers"] == {"fake": {"status": "ok"}}
        with pytest.raises(PapersError) as refused:
            await library.fetch("10.1000/missing")
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
        assert all(r["work"] != "W-open" for r in before["results"])

        done = payload(
            await session.call_tool(
                "rag__index_paper",
                {"identifiers": ["10.1000/open", "10.1000/closed", "10.1000/missing"]},
            )
        )
        status = {r["identifier"]: r["status"] for r in done["results"]}
        assert status == {
            "10.1000/open": "indexed",
            "10.1000/closed": "no_open_full_text",
            "10.1000/missing": "not_found",
        }
        assert done["index"]["papers"] == 2

        after = payload(
            await session.call_tool("rag__retrieve_evidence", {"query": "paced breathing"})
        )
        assert after["results"][0]["work"] == "W-open"

        again = payload(
            await session.call_tool("rag__index_paper", {"identifiers": ["10.1000/open"]})
        )
        assert again["results"][0]["status"] == "already_indexed"

    await with_session(config, tmp_path, body)
