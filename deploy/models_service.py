# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fastapi[standard]",
#   "httpx",
#   "paper-fetch[rerank] @ git+https://github.com/polarizetech/paper-fetch@d733c25",
# ]
# ///
"""The research tool's model service: one app for every host.

It serves what the gateway, the pipeline and paper-fetch call:
- Ollama's own API (/api/*, /v1/*) for bge-m3 (embeddings) and bespoke-minicheck (claim checks),
  proxied unchanged;
- POST /rerank for bge-reranker-v2-m3 (onnx-community ONNX), scored by paper-fetch's OnnxReranker.
Every request needs `Authorization: Bearer <token>`.

Hosts: deploy/modal_models.py serves this app on a Modal GPU. On a machine with Ollama installed
(the Mac fallback), run this file with `uv run deploy/models_service.py`. docs/models-service.md has
the set-up.
"""

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

RERANK_REPO = "onnx-community/bge-reranker-v2-m3-ONNX"


def create_app(token: str, ollama_url: str, reranker: Callable[[], Any]) -> Any:
    """The FastAPI app: the bearer check, /rerank, and Ollama's API proxied unchanged."""
    from types import SimpleNamespace

    import httpx
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.background import BackgroundTask

    api = FastAPI()
    # No timeout: a claim check streams as long as it needs; the pipeline bounds each call.
    client = httpx.AsyncClient(base_url=ollama_url, timeout=None)  # noqa: S113

    @api.middleware("http")
    async def bearer(request: Request, call_next):
        if request.headers.get("authorization") != f"Bearer {token}":
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    @api.post("/rerank")
    def rerank(body: dict) -> dict:
        model = reranker()
        passages = [SimpleNamespace(text=text) for text in body["documents"]]
        return {
            "scores": model.score(body["query"], passages),
            "providers": model.session.get_providers(),
        }

    @api.api_route("/{path:path}", methods=["GET", "POST"])
    async def ollama(path: str, request: Request):
        if not path.startswith(("api/", "v1/")):
            raise HTTPException(status_code=404)
        upstream = client.build_request(
            request.method,
            "/" + path,
            content=await request.body(),
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
        response = await client.send(upstream, stream=True)
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
            background=BackgroundTask(response.aclose),
        )

    return api


def _onnx_reranker(repo: str, onnx_file: str) -> Any:
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from paper_fetch.rerank import OnnxReranker

    # Real files in one folder, not the hub cache's symlinks: onnxruntime refuses a weights file
    # (model.onnx_data) outside the model's own folder. Modal does the same (RERANK_DIR).
    folder = snapshot_download(
        repo,
        allow_patterns=[onnx_file, onnx_file + "_data", "tokenizer.json"],
        local_dir=Path.home() / ".cache" / "models-service" / repo.replace("/", "--"),
    )
    # paper-fetch's tokenizer and score(); its session is replaced by the model file above
    model = OnnxReranker(repo, "onnx/model_int8.onnx")
    model.session = ort.InferenceSession(
        str(Path(folder) / onnx_file), providers=["CPUExecutionProvider"]
    )
    model.inputs = {i.name for i in model.session.get_inputs()}
    model.name = f"onnx:{repo}/{onnx_file}"
    return model


class LazyReranker:
    """The reranker, loaded on first use and dropped after `idle_seconds` without a call, so a
    shared machine holds its memory only while the service is in use."""

    def __init__(
        self,
        repo: str,
        onnx_file: str,
        idle_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        loader: Callable[[str, str], Any] = _onnx_reranker,
    ) -> None:
        self.repo, self.onnx_file, self.idle = repo, onnx_file, idle_seconds
        self.clock, self.loader = clock, loader
        self.model: Any = None
        self.last = clock()
        self.lock = threading.Lock()

    def __call__(self) -> Any:
        with self.lock:
            if self.model is None:
                self.model = self.loader(self.repo, self.onnx_file)
            self.last = self.clock()
            return self.model

    def drop_if_idle(self) -> bool:
        with self.lock:
            if self.model is not None and self.clock() - self.last >= self.idle:
                self.model = None
                return True
            return False


def main() -> None:
    import uvicorn

    key_file = os.environ.get("MODELS_KEY_FILE", "~/.config/polarize/models.key")
    token = Path(key_file).expanduser().read_text().strip()
    lazy = LazyReranker(
        RERANK_REPO,
        os.environ.get("RERANK_FILE", "onnx/model.onnx"),
        float(os.environ.get("RERANK_IDLE_SECONDS", "600")),
    )

    def sweep() -> None:
        while True:
            time.sleep(60)
            lazy.drop_if_idle()

    threading.Thread(target=sweep, daemon=True).start()
    app = create_app(token, os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"), lazy)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("MODELS_PORT", "8896")))


if __name__ == "__main__":
    main()
