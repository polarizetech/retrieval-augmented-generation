# /// script
# requires-python = ">=3.12"
# dependencies = ["fastapi[standard]", "httpx"]
# ///
"""The model service router on the droplet: Modal first, the Mac when Modal is out.

The gateway, the pipeline and paper-fetch point at this router instead of a model service. It
sends each call to the primary (Modal). When the primary answers 402 or 429 (out of budget or
limited) or cannot be reached, the router marks it out for a cooldown and sends the call, and the
calls after it, to the fallback (the Mac's deploy/models_service.py over the tailnet). After the
cooldown it tries the primary again. GET /router/status says which one it is using.
MODELS_FALLBACK_PROXY (optional) sends only the fallback's calls through an HTTP proxy, such as
tailscaled's outbound proxy on a host whose Tailscale runs in userspace mode.
"""

import os
import time
from collections.abc import Callable
from typing import Any

OUT_STATUSES = (402, 429)


class Upstreams:
    """The primary and fallback model services, and whether the primary is out."""

    def __init__(
        self,
        primary: str,
        fallback: str,
        cooldown: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.primary = primary.rstrip("/")
        self.fallback = fallback.rstrip("/")
        self.cooldown = cooldown
        self.clock = clock
        self.out_until = float("-inf")

    def out(self) -> bool:
        return self.clock() < self.out_until

    def order(self) -> list[str]:
        """The services to try, in order: the fallback alone while the primary is out."""
        return [self.fallback] if self.out() else [self.primary, self.fallback]

    def mark_out(self, url: str) -> None:
        if url == self.primary:
            self.out_until = self.clock() + self.cooldown

    def status(self) -> dict[str, Any]:
        if self.out():
            return {
                "using": self.fallback,
                "primary_out_for_seconds": round(self.out_until - self.clock(), 1),
            }
        return {"using": self.primary, "primary_out_for_seconds": 0.0}


def create_app(upstreams: Upstreams, fallback_proxy: str = "") -> Any:
    """The FastAPI app: /router/status, and every other call forwarded with its body streamed."""
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.background import BackgroundTask

    api = FastAPI()
    # Connecting is bounded; a reply may stream as long as the service needs. Only the fallback
    # goes through the proxy (tailscaled's, to reach the tailnet); neither client takes proxy
    # settings from the environment.
    timeout = httpx.Timeout(None, connect=10.0)
    clients = {
        upstreams.primary: httpx.AsyncClient(timeout=timeout, trust_env=False),
        upstreams.fallback: httpx.AsyncClient(
            timeout=timeout, trust_env=False, proxy=fallback_proxy or None
        ),
    }

    @api.get("/router/status")
    def status() -> dict:
        return upstreams.status()

    @api.api_route("/{path:path}", methods=["GET", "POST"])
    async def forward(path: str, request: Request):
        body = await request.body()
        keep = ("authorization", "content-type")
        headers = {k: v for k, v in request.headers.items() if k in keep}
        last = "nothing was tried"
        for base in upstreams.order():
            outgoing = clients[base].build_request(
                request.method, f"{base}/{path}", content=body, headers=headers
            )
            try:
                response = await clients[base].send(outgoing, stream=True)
            except httpx.TransportError as exc:
                upstreams.mark_out(base)
                last = f"{base}: {type(exc).__name__}"
                print(f"models router: {last}; trying the next service", flush=True)
                continue
            if response.status_code in OUT_STATUSES and base == upstreams.primary:
                await response.aclose()
                upstreams.mark_out(base)
                last = f"{base}: HTTP {response.status_code}"
                print(f"models router: {last}; the fallback serves for now", flush=True)
                continue
            return StreamingResponse(
                response.aiter_raw(),
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
                background=BackgroundTask(response.aclose),
            )
        return JSONResponse({"error": f"no model service answered ({last})"}, status_code=503)

    return api


def main() -> None:
    import uvicorn

    upstreams = Upstreams(
        os.environ["MODELS_PRIMARY_URL"],
        os.environ["MODELS_FALLBACK_URL"],
        float(os.environ.get("MODELS_COOLDOWN_SECONDS", "900")),
    )
    port = int(os.environ.get("MODELS_ROUTER_PORT", "8895"))
    proxy = os.environ.get("MODELS_FALLBACK_PROXY", "")
    uvicorn.run(create_app(upstreams, proxy), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
