"""Single-user OAuth 2.1 for exposing the gateway to remote MCP clients (ChatGPT, Claude web).

A remote client registers itself (dynamic client registration), sends its user to /authorize with
PKCE, and the user unlocks a consent page with one private access key, kept by the operator (for
example in the macOS Keychain). The client then holds a one-hour access token and a rotating
30-day refresh token, both bound to this server's resource URL.

Tokens and codes are stored as SHA-256 hashes, so the state file alone cannot be replayed.
"""

from __future__ import annotations

import hashlib
import html
import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

SCOPE = "rag:use"
ACCESS_TOKEN_SECONDS = 60 * 60
REFRESH_TOKEN_SECONDS = 30 * 24 * 60 * 60
AUTHORIZATION_CODE_SECONDS = 5 * 60
PENDING_SECONDS = 10 * 60
MAX_KEY_ATTEMPTS = 5
MIN_ACCESS_KEY_CHARS = 32


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


class OAuthStore:
    """Clients, pending approvals, codes and tokens, in one private SQLite file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(path, check_same_thread=False)
        path.chmod(0o600)
        self.db.execute(
            "create table if not exists oauth_objects ("
            "kind text not null, key text not null, value text not null, "
            "primary key (kind, key))"
        )
        self.db.commit()

    def put(self, kind: str, key: str, value: dict[str, Any]) -> None:
        with self.db:
            self.db.execute(
                "insert or replace into oauth_objects values (?, ?, ?)",
                (kind, key, json.dumps(value)),
            )

    def get(self, kind: str, key: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "select value from oauth_objects where kind=? and key=?", (kind, key)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def delete(self, kind: str, key: str) -> None:
        with self.db:
            self.db.execute("delete from oauth_objects where kind=? and key=?", (kind, key))


class SingleUserOAuthProvider:
    """DCR + PKCE provider whose consent page is unlocked by one private access key."""

    def __init__(
        self, public_url: str, access_key: str, state_path: Path, resource_path: str = "/mcp"
    ):
        if len(access_key) < MIN_ACCESS_KEY_CHARS:
            raise ValueError(
                f"MCP_OAUTH_ACCESS_KEY must contain at least {MIN_ACCESS_KEY_CHARS} characters"
            )
        self.public_url = public_url.rstrip("/")
        self.resource_url = self.public_url + resource_path
        self.access_key = access_key
        self.store = OAuthStore(state_path)

    # -- clients -----------------------------------------------------------------------------
    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        value = self.store.get("client", client_id)
        return OAuthClientInformationFull.model_validate(value) if value else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        if not client_info.client_id:
            raise ValueError("registered client is missing client_id")
        self.store.put("client", client_info.client_id, client_info.model_dump(mode="json"))

    # -- authorization -----------------------------------------------------------------------
    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        if params.resource and params.resource.rstrip("/") != self.resource_url:
            raise AuthorizeError("invalid_request", "resource must identify this MCP endpoint")
        request_id = secrets.token_urlsafe(32)
        self.store.put(
            "pending",
            request_id,
            {
                "client_id": client.client_id,
                "params": params.model_dump(mode="json"),
                "expires_at": time.time() + PENDING_SECONDS,
                "attempts": 0,
            },
        )
        return f"{self.public_url}/oauth/consent?{urlencode({'request_id': request_id})}"

    def approve(self, request_id: str, supplied_key: str) -> str:
        """Check the access key; on success return the client's redirect with a fresh code."""
        pending = self.store.get("pending", request_id)
        if not pending or pending["expires_at"] < time.time():
            self.store.delete("pending", request_id)
            raise ValueError("authorization request is missing or expired")
        pending["attempts"] += 1
        if pending["attempts"] > MAX_KEY_ATTEMPTS:
            self.store.delete("pending", request_id)
            raise ValueError("authorization request is locked; start again from the client")
        if not secrets.compare_digest(supplied_key.encode(), self.access_key.encode()):
            self.store.put("pending", request_id, pending)
            raise PermissionError("access key is incorrect")

        params = AuthorizationParams.model_validate(pending["params"])
        code_value = secrets.token_urlsafe(32)
        code = AuthorizationCode(
            code=_hash(code_value),
            scopes=params.scopes or [SCOPE],
            expires_at=time.time() + AUTHORIZATION_CODE_SECONDS,
            client_id=pending["client_id"],
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource or self.resource_url,
        )
        self.store.put("code", _hash(code_value), code.model_dump(mode="json"))
        self.store.delete("pending", request_id)
        return construct_redirect_uri(str(params.redirect_uri), code=code_value, state=params.state)

    def redirect_origin(self, request_id: str) -> str | None:
        """The origin the consent form may post back to, for its Content-Security-Policy."""
        pending = self.store.get("pending", request_id)
        if not pending or pending["expires_at"] < time.time():
            return None
        redirect_uri = str(AuthorizationParams.model_validate(pending["params"]).redirect_uri)
        parsed = urlsplit(redirect_uri)
        return f"{parsed.scheme}://{parsed.netloc}"

    # -- codes and tokens --------------------------------------------------------------------
    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        value = self.store.get("code", _hash(authorization_code))
        if not value or value["client_id"] != client.client_id:
            return None
        return AuthorizationCode.model_validate(value | {"code": authorization_code})

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        key = _hash(authorization_code.code)
        if not self.store.get("code", key):
            raise TokenError("invalid_grant", "authorization code was already used")
        self.store.delete("code", key)
        return self._issue_tokens(
            client.client_id or "",
            authorization_code.scopes,
            authorization_code.resource or self.resource_url,
        )

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        value = self.store.get("refresh", _hash(refresh_token))
        if not value or value["client_id"] != client.client_id:
            return None
        if value.get("expires_at") and value["expires_at"] < time.time():
            self.store.delete("refresh", _hash(refresh_token))
            return None
        return RefreshToken.model_validate(value | {"token": refresh_token})

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        # Rotation: a refresh token works once.
        self.store.delete("refresh", _hash(refresh_token.token))
        return self._issue_tokens(
            client.client_id or "", scopes, refresh_token.resource or self.resource_url
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        value = self.store.get("access", _hash(token))
        if not value:
            return None
        if value.get("expires_at") and value["expires_at"] < time.time():
            self.store.delete("access", _hash(token))
            return None
        return AccessToken.model_validate(value | {"token": token})

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        kind = "access" if isinstance(token, AccessToken) else "refresh"
        self.store.delete(kind, _hash(token.token))

    def _issue_tokens(self, client_id: str, scopes: list[str], resource: str) -> OAuthToken:
        now = int(time.time())
        access_value = secrets.token_urlsafe(32)
        refresh_value = secrets.token_urlsafe(32)
        shared = {"client_id": client_id, "scopes": scopes, "resource": resource}
        access = AccessToken(
            token=_hash(access_value), expires_at=now + ACCESS_TOKEN_SECONDS, **shared
        )
        refresh = RefreshToken(
            token=_hash(refresh_value), expires_at=now + REFRESH_TOKEN_SECONDS, **shared
        )
        self.store.put("access", _hash(access_value), access.model_dump(mode="json"))
        self.store.put("refresh", _hash(refresh_value), refresh.model_dump(mode="json"))
        return OAuthToken(
            access_token=access_value,
            expires_in=ACCESS_TOKEN_SECONDS,
            refresh_token=refresh_value,
            scope=" ".join(scopes),
        )


async def consent(provider: SingleUserOAuthProvider, request: Request) -> Response:
    """GET shows the access-key form; POST checks the key and redirects back to the client."""
    message = ""
    if request.method == "POST":
        form = await request.form()
        request_id = str(form.get("request_id", ""))
        redirect_origin = provider.redirect_origin(request_id)
        try:
            return RedirectResponse(
                provider.approve(request_id, str(form.get("access_key", ""))), status_code=303
            )
        except PermissionError:
            message = "Access key is incorrect."
        except ValueError as exc:
            message = str(exc)
    else:
        request_id = request.query_params.get("request_id", "")
        redirect_origin = provider.redirect_origin(request_id)
    form_action = "'self'" + (f" {redirect_origin}" if redirect_origin else "")
    return HTMLResponse(
        _consent_page(request_id, message),
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                f"default-src 'none'; style-src 'unsafe-inline'; form-action {form_action}; "
                "frame-ancestors 'none'"
            ),
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
        },
    )


def _consent_page(request_id: str, message: str) -> str:
    safe_request_id = html.escape(request_id, quote=True)
    safe_message = html.escape(message)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Authorize Retrieval-Augmented Generation</title></head>
<body style="font-family:system-ui;max-width:34rem;margin:4rem auto;padding:1rem">
<h1>Authorize Retrieval-Augmented Generation</h1>
<p>Enter the private access key stored for this MCP server.</p>
<p style="color:#b42318">{safe_message}</p>
<form method="post">
<input type="hidden" name="request_id" value="{safe_request_id}">
<label>Access key<br><input name="access_key" type="password" required
autocomplete="current-password" style="width:100%;padding:.7rem;margin:.5rem 0"></label>
<button type="submit" style="padding:.7rem 1rem">Authorize</button>
</form></body></html>"""
