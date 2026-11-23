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

Per provider, `MODELS_<N>_KIND` is `ollama` (default) or `digitalocean`. `MODELS_<N>_KEY_FILE` names
a file holding that provider's key, sent in place of the caller's key. `MODELS_<N>_EMBED_MODEL` and
`MODELS_<N>_RERANK_MODEL` name the models a `digitalocean` provider uses (`bge-m3` and
`bge-reranker-v2-m3` by default). A `digitalocean` provider serves reranking and embeddings and
hands every other call (chat, tags, ps) to the next provider.
"""

import json
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OUT_STATUSES = (402, 429)
DO_OUT_STATUSES = (401, 402, 403, 429)
DEFAULT_PROVIDERS = "local,modal"
KINDS = ("ollama", "digitalocean")


@dataclass
class Provider:
    """A model service the router can send calls to, and whether it is out."""

    name: str
    url: str
    proxy: str
    kind: str = "ollama"
    key: str = ""
    embed_model: str = "bge-m3"
    rerank_model: str = "bge-reranker-v2-m3"
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


def digitalocean_request(
    method: str, path: str, body: bytes, provider: Provider
) -> tuple[str, dict, str] | None:
    """The DigitalOcean path, JSON body and reply kind for a call, or None when it doesn't serve it.

    DigitalOcean's serverless inference (`https://inference.do-ai.run`) serves bge-reranker-v2-m3
    and bge-m3, so `rerank`, `v1/embeddings` and Ollama's `api/embed` map onto its `v1/rerank` and
    `v1/embeddings`. Every other call (chat, tags, ps) is left to the next provider.
    """
    path = path.lstrip("/")
    if method != "POST":
        return None
    if path == "rerank":
        b = json.loads(body)
        return (
            "v1/rerank",
            {"model": provider.rerank_model, "query": b["query"], "documents": b["documents"]},
            "rerank",
        )
    if path == "v1/embeddings":
        b = json.loads(body)
        return ("v1/embeddings", {"model": provider.embed_model, "input": b["input"]}, "openai")
    if path == "api/embed":
        b = json.loads(body)
        text = b["input"]
        if isinstance(text, str):
            text = [text]
        return ("v1/embeddings", {"model": provider.embed_model, "input": text}, "ollama")
    return None


def _unit(vector: list[float]) -> list[float]:
    """The vector divided by its Euclidean norm; a zero vector is returned as it is."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return list(vector)
    return [x / norm for x in vector]


def digitalocean_response(kind: str, body: dict, model: str) -> dict:
    """The DigitalOcean reply reshaped into the shape the caller expects.

    For `rerank`, the scores come back one per document in document order, placed by each
    result's `index`; a document with no result gets -inf, so it can never outrank a real one.

    For `openai` and `ollama`, the embeddings come back in index order. DigitalOcean's embedding
    vectors are not unit length while Ollama's are, so the router scales them to unit length to
    match what the Mac and Modal return.
    """
    if kind == "rerank":
        results = body["results"]
        scores = [float("-inf")] * len(results)
        for r in results:
            scores[r["index"]] = r["relevance_score"]
        return {"scores": scores}
    if kind == "openai":
        data = sorted(body["data"], key=lambda d: d["index"])
        return {**body, "data": [{**d, "embedding": _unit(d["embedding"])} for d in data]}
    if kind == "ollama":
        data = sorted(body["data"], key=lambda d: d["index"])
        return {"model": model, "embeddings": [_unit(d["embedding"]) for d in data]}
    raise ValueError(f"no reply shaping for {kind!r} yet")


def create_app(providers: Providers, transports: Mapping[str, Any] | None = None) -> Any:
    """The FastAPI app: /router/status, and every other call forwarded with its body streamed."""
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    from starlette.background import BackgroundTask

    api = FastAPI()
    # Connecting is bounded; a reply may stream as long as the service needs. Only a provider
    # with a proxy setting goes through that proxy (tailscaled's, to reach the tailnet); no
    # client takes proxy settings from the environment.
    timeout = httpx.Timeout(None, connect=10.0)
    transports = transports or {}
    clients = {
        p.name: httpx.AsyncClient(
            timeout=timeout,
            trust_env=False,
            proxy=p.proxy or None,
            **({"transport": transports[p.name]} if p.name in transports else {}),
        )
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
            if p.kind == "digitalocean":
                sent = digitalocean_request(request.method, path, body, p)
                if sent is None:
                    last = f"{p.name}: does not serve {path}"
                    continue
                do_path, json_body, kind = sent
                do_headers = {"content-type": "application/json"}
                if p.key:
                    do_headers["authorization"] = f"Bearer {p.key}"
                try:
                    do_response = await client.post(
                        f"{p.url}/{do_path}", json=json_body, headers=do_headers
                    )
                except httpx.TransportError as exc:
                    providers.mark_out(p)
                    last = f"{p.name}: {type(exc).__name__}"
                    print(f"models router: {last}; trying the next provider", flush=True)
                    continue
                if do_response.status_code in DO_OUT_STATUSES:
                    providers.mark_out(p)
                    last = f"{p.name}: HTTP {do_response.status_code}"
                    print(f"models router: {last}; the next provider serves for now", flush=True)
                    continue
                if 200 <= do_response.status_code < 300:
                    return JSONResponse(
                        digitalocean_response(kind, do_response.json(), p.embed_model),
                        status_code=do_response.status_code,
                    )
                return Response(
                    content=do_response.content,
                    status_code=do_response.status_code,
                    media_type=do_response.headers.get("content-type"),
                )
            out_headers = dict(headers)
            if p.key:
                out_headers["authorization"] = f"Bearer {p.key}"
            outgoing = client.build_request(
                request.method, f"{p.url}/{path}", content=body, headers=out_headers
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
        upper = name.upper()
        url_var = f"MODELS_{upper}_URL"
        if url_var not in env:
            raise KeyError(url_var)
        proxy = env.get(f"MODELS_{upper}_PROXY") or ""
        kind_var = f"MODELS_{upper}_KIND"
        kind = (env.get(kind_var) or "ollama").strip().lower()
        if kind not in KINDS:
            raise ValueError(f"{kind_var} must be ollama or digitalocean, not {kind!r}")
        key_file = env.get(f"MODELS_{upper}_KEY_FILE") or ""
        key = Path(key_file).expanduser().read_text().strip() if key_file else ""
        providers.append(
            Provider(
                name,
                env[url_var],
                proxy,
                kind=kind,
                key=key,
                embed_model=env.get(f"MODELS_{upper}_EMBED_MODEL") or "bge-m3",
                rerank_model=env.get(f"MODELS_{upper}_RERANK_MODEL") or "bge-reranker-v2-m3",
            )
        )
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
