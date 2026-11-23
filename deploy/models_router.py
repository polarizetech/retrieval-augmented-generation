# /// script
# requires-python = ">=3.12"
# dependencies = ["fastapi[standard]", "httpx"]
# ///
"""The model service router on the droplet: an ordered list of providers, the Mac first.

The gateway, the pipeline and paper-fetch point at this router instead of a model service. It
keeps an ordered list of providers and sends each call to the first one that is not out. By
default the list is the Mac (`local`, the deploy/models_service.py over the tailnet) and then Modal
(`modal`). A provider that answers 402 or 429 (out of budget or limited) or cannot be reached is
marked out for a cooldown, and the next provider serves the call. When every provider is out, all
of them are still tried, in order. GET /router/status shows the order, the provider in use and
each provider's remaining cooldown.

Settings: `MODELS_PROVIDERS` is a comma list of names (default `local,modal`). Each name N needs
`MODELS_<N>_URL`, and may have `MODELS_<N>_PROXY`, an HTTP proxy for that provider's calls only,
such as tailscaled's outbound proxy on a host whose Tailscale runs in userspace mode. The older
settings still work and mean modal then local: `MODELS_PRIMARY_URL`, `MODELS_FALLBACK_URL` and
`MODELS_FALLBACK_PROXY`.
"""

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

OUT_STATUSES = (402, 429)
DEFAULT_PROVIDERS = "local,modal"


@dataclass
class Provider:
    """A model service the router can send calls to, and whether it is out."""

    name: str
    url: str
    proxy: str
    out_until: float = float("-inf")

    def __post_init__(self) -> None:
        self.url = self.url.rstrip("/")


class Providers:
    """The ordered providers, and which of them are out for the cooldown."""

    def __init__(
        self,
        providers: list[Provider],
        cooldown: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.providers = providers
        self.cooldown = cooldown
        self.clock = clock

    def out(self, p: Provider) -> bool:
        return self.clock() < p.out_until

    def order(self) -> list[Provider]:
        """The providers to try: those not out, in configured order, then those out."""
        up = [p for p in self.providers if not self.out(p)]
        down = [p for p in self.providers if self.out(p)]
        return up + down

    def mark_out(self, p: Provider) -> None:
        p.out_until = self.clock() + self.cooldown

    def status(self) -> dict[str, Any]:
        using = next((p for p in self.providers if not self.out(p)), self.providers[0])
        return {
            "order": [p.name for p in self.providers],
            "using": using.name,
            "out_for_seconds": {
                p.name: round(p.out_until - self.clock(), 1) for p in self.providers if self.out(p)
            },
        }


def create_app(providers: Providers) -> Any:
    """The FastAPI app: /router/status, and every other call forwarded with its body streamed."""
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.background import BackgroundTask

    api = FastAPI()
    # Connecting is bounded; a reply may stream as long as the service needs. Only a provider
    # with a proxy setting goes through that proxy (tailscaled's, to reach the tailnet); no
    # client takes proxy settings from the environment.
    timeout = httpx.Timeout(None, connect=10.0)
    clients = {
        p.name: httpx.AsyncClient(timeout=timeout, trust_env=False, proxy=p.proxy or None)
        for p in providers.providers
    }

    @api.get("/router/status")
    def status() -> dict:
        return providers.status()

    @api.api_route("/{path:path}", methods=["GET", "POST"])
    async def forward(path: str, request: Request):
        body = await request.body()
        keep = ("authorization", "content-type")
        headers = {k: v for k, v in request.headers.items() if k in keep}
        last = "nothing was tried"
        for p in providers.order():
            client = clients[p.name]
            outgoing = client.build_request(
                request.method, f"{p.url}/{path}", content=body, headers=headers
            )
            try:
                response = await client.send(outgoing, stream=True)
            except httpx.TransportError as exc:
                providers.mark_out(p)
                last = f"{p.name}: {type(exc).__name__}"
                print(f"models router: {last}; trying the next provider", flush=True)
                continue
            if response.status_code in OUT_STATUSES:
                await response.aclose()
                providers.mark_out(p)
                last = f"{p.name}: HTTP {response.status_code}"
                print(f"models router: {last}; the next provider serves for now", flush=True)
                continue
            return StreamingResponse(
                response.aiter_raw(),
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
                background=BackgroundTask(response.aclose),
            )
        return JSONResponse({"error": f"no model service answered ({last})"}, status_code=503)

    return api


def _named_providers(env: Mapping[str, str], names: list[str]) -> list[Provider]:
    providers = []
    for raw in names:
        name = raw.strip().lower()
        if not name:
            continue
        url_var = f"MODELS_{name.upper()}_URL"
        if url_var not in env:
            raise KeyError(url_var)
        proxy = env.get(f"MODELS_{name.upper()}_PROXY") or ""
        providers.append(Provider(name, env[url_var], proxy))
    return providers


def providers_from_env(env: Mapping[str, str]) -> list[Provider]:
    """The providers in the order the router tries them, from the settings in env."""
    listed = env.get("MODELS_PROVIDERS")
    if listed:
        return _named_providers(env, listed.split(","))
    if env.get("MODELS_PRIMARY_URL"):
        return [
            Provider("modal", env["MODELS_PRIMARY_URL"], ""),
            Provider("local", env["MODELS_FALLBACK_URL"], env.get("MODELS_FALLBACK_PROXY") or ""),
        ]
    return _named_providers(env, DEFAULT_PROVIDERS.split(","))


def main() -> None:
    import uvicorn

    providers = Providers(
        providers_from_env(os.environ),
        float(os.environ.get("MODELS_COOLDOWN_SECONDS", "900")),
    )
    port = int(os.environ.get("MODELS_ROUTER_PORT", "8895"))
    uvicorn.run(create_app(providers), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
