"""Remote access: the single-user OAuth provider, and the whole flow through the HTTP gateway."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import sqlite3
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull
from starlette.testclient import TestClient

from research_mcp import gateway
from research_mcp.oauth import SingleUserOAuthProvider
from tests.test_gateway import OfflineOllama

KEY = "correct horse battery staple access key"
PUBLIC = "https://mcp.example.org/research"
REDIRECT = "https://client.example.org/callback"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def provider(tmp_path: Path) -> SingleUserOAuthProvider:
    return SingleUserOAuthProvider(PUBLIC, KEY, tmp_path / "oauth.sqlite", "")


def a_client(client_id: str = "client-1") -> OAuthClientInformationFull:
    return OAuthClientInformationFull.model_validate(
        {
            "client_id": client_id,
            "redirect_uris": [REDIRECT],
            "token_endpoint_auth_method": "none",
            "scope": "rag:use",
        }
    )


def params(resource: str = PUBLIC) -> AuthorizationParams:
    return AuthorizationParams.model_validate(
        {
            "state": "opaque",
            "scopes": ["rag:use"],
            "code_challenge": "a" * 43,
            "redirect_uri": REDIRECT,
            "redirect_uri_provided_explicitly": True,
            "resource": resource,
        }
    )


async def authorized(provider: SingleUserOAuthProvider) -> tuple[OAuthClientInformationFull, str]:
    client = a_client()
    await provider.register_client(client)
    url = await provider.authorize(client, params())
    assert url.startswith(f"{PUBLIC}/oauth/consent?")
    return client, parse_qs(urlparse(url).query)["request_id"][0]


class TestProvider:
    @pytest.mark.anyio
    async def test_code_token_refresh_and_revocation(
        self, provider: SingleUserOAuthProvider
    ) -> None:
        client, request_id = await authorized(provider)
        assert provider.redirect_origin(request_id) == "https://client.example.org"
        with pytest.raises(PermissionError):
            provider.approve(request_id, "wrong key")
        callback = parse_qs(urlparse(provider.approve(request_id, KEY)).query)
        assert callback["state"] == ["opaque"]

        code = await provider.load_authorization_code(client, callback["code"][0])
        assert code is not None
        token = await provider.exchange_authorization_code(client, code)
        access = await provider.load_access_token(token.access_token)
        assert access is not None
        assert (access.resource, access.scopes) == (PUBLIC, ["rag:use"])
        with pytest.raises(Exception, match="already used"):
            await provider.exchange_authorization_code(client, code)

        assert token.refresh_token is not None
        refresh = await provider.load_refresh_token(client, token.refresh_token)
        assert refresh is not None
        rotated = await provider.exchange_refresh_token(client, refresh, ["rag:use"])
        assert await provider.load_refresh_token(client, token.refresh_token) is None
        rotated_access = await provider.load_access_token(rotated.access_token)
        assert rotated_access is not None
        await provider.revoke_token(rotated_access)
        assert await provider.load_access_token(rotated.access_token) is None

    @pytest.mark.anyio
    async def test_the_state_file_holds_no_usable_secret(
        self, provider: SingleUserOAuthProvider, tmp_path: Path
    ) -> None:
        client, request_id = await authorized(provider)
        code_value = parse_qs(urlparse(provider.approve(request_id, KEY)).query)["code"][0]
        code = await provider.load_authorization_code(client, code_value)
        assert code is not None
        token = await provider.exchange_authorization_code(client, code)
        dump = "\n".join(
            r[0]
            for r in sqlite3.connect(tmp_path / "oauth.sqlite").execute(
                "select value from oauth_objects"
            )
        )
        assert token.access_token not in dump
        assert token.refresh_token is not None
        assert token.refresh_token not in dump
        assert (tmp_path / "oauth.sqlite").stat().st_mode & 0o777 == 0o600

    @pytest.mark.anyio
    async def test_a_code_belongs_to_the_client_it_was_issued_to(
        self, provider: SingleUserOAuthProvider
    ) -> None:
        _, request_id = await authorized(provider)
        code_value = parse_qs(urlparse(provider.approve(request_id, KEY)).query)["code"][0]
        other = a_client("client-2")
        assert await provider.load_authorization_code(other, code_value) is None

    @pytest.mark.anyio
    async def test_guessing_the_key_locks_the_request(
        self, provider: SingleUserOAuthProvider
    ) -> None:
        _, request_id = await authorized(provider)
        for _ in range(5):
            with pytest.raises(PermissionError):
                provider.approve(request_id, "guess")
        with pytest.raises(ValueError, match="locked"):
            provider.approve(request_id, KEY)

    @pytest.mark.anyio
    async def test_a_request_for_another_resource_is_refused(
        self, provider: SingleUserOAuthProvider
    ) -> None:
        with pytest.raises(Exception, match="resource must identify"):
            await provider.authorize(a_client(), params("https://evil.example/mcp"))

    def test_the_access_key_must_be_long(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least 32"):
            SingleUserOAuthProvider(PUBLIC, "too-short", tmp_path / "x.sqlite")


def test_the_whole_flow_through_the_http_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What ChatGPT does: discover, register, authorize with PKCE, consent, token, call a tool."""
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PIPELINE_EMBEDDING_MODEL", "fake-embed")
    monkeypatch.setenv("MCP_PUBLIC_URL", PUBLIC)
    monkeypatch.setenv("MCP_RESOURCE_PATH", "")
    monkeypatch.setenv("MCP_OAUTH_ACCESS_KEY", KEY)
    monkeypatch.setenv("MCP_OAUTH_STATE", str(tmp_path / "oauth.sqlite"))
    monkeypatch.setattr(gateway, "Ollama", OfflineOllama)
    config = tmp_path / "gateway.json"
    config.write_text(json.dumps({"upstreams": {}}))
    server, _ = gateway.build_server(config, "127.0.0.1", 0, tmp_path / "token", http=True)
    headers = {"Host": "mcp.example.org"}

    with TestClient(server.streamable_http_app(), base_url="https://mcp.example.org") as http:
        unauthorized = http.post("/", json={}, headers=headers)
        assert unauthorized.status_code == 401
        assert "oauth-protected-resource/research" in unauthorized.headers["www-authenticate"]
        prm = http.get("/.well-known/oauth-protected-resource/research", headers=headers).json()
        assert prm["resource"] == PUBLIC
        meta = http.get("/.well-known/oauth-authorization-server", headers=headers).json()
        assert meta["authorization_endpoint"] == f"{PUBLIC}/authorize"

        client = http.post(
            "/register",
            headers=headers,
            json={
                "redirect_uris": [REDIRECT],
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "scope": "rag:use",
            },
        ).json()
        verifier = secrets.token_urlsafe(48)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        consent_url = http.get(
            "/authorize",
            headers=headers,
            follow_redirects=False,
            params={
                "response_type": "code",
                "client_id": client["client_id"],
                "redirect_uri": REDIRECT,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "scope": "rag:use",
                "state": "s",
                "resource": PUBLIC,
            },
        ).headers["location"]
        request_id = parse_qs(urlparse(consent_url).query)["request_id"][0]

        page = http.get("/oauth/consent", params={"request_id": request_id}, headers=headers)
        assert "Access key" in page.text
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        wrong = http.post(
            "/oauth/consent",
            headers=headers,
            data={"request_id": request_id, "access_key": "wrong key"},
        )
        assert "Access key is incorrect." in wrong.text
        approved = http.post(
            "/oauth/consent",
            headers=headers,
            follow_redirects=False,
            data={"request_id": request_id, "access_key": KEY},
        )
        assert approved.status_code == 303
        code = parse_qs(urlparse(approved.headers["location"]).query)["code"][0]

        token = http.post(
            "/token",
            headers=headers,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT,
                "client_id": client["client_id"],
                "code_verifier": verifier,
                "resource": PUBLIC,
            },
        ).json()
        call = http.post(
            "/",
            headers=headers
            | {
                "Authorization": f"Bearer {token['access_token']}",
                "Accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "gateway_status", "arguments": {}},
            },
        )
        assert call.status_code == 200, call.text
        assert call.json()["result"]["structuredContent"]["total_tools"] == 9
