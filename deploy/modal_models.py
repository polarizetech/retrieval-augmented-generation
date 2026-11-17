# pyright: reportMissingImports=false
# (modal, fastapi, httpx and paper_fetch exist only in the Modal image, not the dev env.)
"""The research tool's local models on a Modal GPU, so they no longer run on the Mac.

Same models, same files, same scoring as before, on other hardware:
- Ollama's own API (/api/*, /v1/*) for bge-m3 (embeddings) and bespoke-minicheck:latest (claim
  checks), proxied unchanged.
- POST /rerank for bge-reranker-v2-m3 (onnx-community int8 ONNX), scored by paper-fetch's own
  OnnxReranker.score; only the onnxruntime session moves to the GPU.

Every request needs `Authorization: Bearer <token>`; the token is the Modal secret
`research-rag-models` (key MODELS_TOKEN). Containers start on demand and stop when idle, with
Modal's default idle window.

    modal run deploy/modal_models.py::pull     # once: the models into the volume
    modal deploy deploy/modal_models.py

Clients: PIPELINE_OLLAMA_URL and PIPELINE_OLLAMA_KEY_FILE (pipeline and gateway);
PAPER_FETCH_EMBED_URL=<url>/v1 and PAPER_FETCH_RERANK_URL=<url>, each with its _KEY_FILE
(paper-fetch).
"""

import modal

OLLAMA = "http://127.0.0.1:11434"
OLLAMA_MODELS = ("bge-m3:latest", "bespoke-minicheck:latest")
RERANK_REPO = "onnx-community/bge-reranker-v2-m3-ONNX"
RERANK_FILE = "onnx/model.onnx"
# real files for the model and its weights file, not cache symlinks
RERANK_DIR = "/models/rerank"

volume = modal.Volume.from_name("research-rag-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "zstd", "git")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install(
        "fastapi[standard]",
        "httpx",
        "paper-fetch[rerank] @ git+https://github.com/polarizetech/paper-fetch@d733c25",
    )
    .run_commands("pip uninstall -y onnxruntime", "pip install 'onnxruntime-gpu[cuda,cudnn]'")
    .env({"OLLAMA_MODELS": "/models/ollama", "HF_HOME": "/models/hf", "OLLAMA_KEEP_ALIVE": "-1"})
)
app = modal.App("research-rag-models", image=image)


def _start_ollama():
    import subprocess
    import time
    import urllib.request

    server = subprocess.Popen(["ollama", "serve"])  # noqa: S607
    for _ in range(120):
        try:
            urllib.request.urlopen(OLLAMA + "/api/version", timeout=1)
            return server
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("ollama did not start")


@app.function(volumes={"/models": volume}, timeout=1800)
def pull() -> None:
    """Fetch the models into the volume (once; again only to change a model)."""
    import subprocess

    from huggingface_hub import hf_hub_download

    server = _start_ollama()
    for name in OLLAMA_MODELS:
        subprocess.run(["ollama", "pull", name], check=True)  # noqa: S603, S607
    for name in (RERANK_FILE, RERANK_FILE + "_data", "tokenizer.json"):
        hf_hub_download(RERANK_REPO, name, local_dir=RERANK_DIR)
    server.terminate()
    volume.commit()


@app.cls(
    gpu=["L4", "A10", "T4"],
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("research-rag-models")],
    # stay up 10 idle minutes, so a session's calls find the models loaded
    scaledown_window=600,
)
# one container takes the gateway's overlapping calls instead of starting more
@modal.concurrent(max_inputs=8)
class Models:
    @modal.enter()
    def start(self) -> None:
        import httpx
        import onnxruntime as ort
        from paper_fetch.rerank import OnnxReranker

        self.ollama = _start_ollama()
        # load the models now, so a request never waits on a load from the volume
        httpx.post(
            f"{OLLAMA}/api/embed",
            json={"model": "bge-m3:latest", "input": "warm"},
            timeout=None,  # noqa: S113
        )
        httpx.post(
            f"{OLLAMA}/api/generate",
            json={"model": "bespoke-minicheck:latest", "prompt": "", "stream": False},
            timeout=None,  # noqa: S113
        )
        # paper-fetch's tokenizer and score(); the session below replaces its CPU one
        self.reranker = OnnxReranker(RERANK_REPO, "onnx/model_int8.onnx")
        # paper-fetch builds its session on the CPU; the same model file, on the GPU here.
        ort.preload_dlls()  # CUDA and cuDNN from the onnxruntime-gpu[cuda,cudnn] packages
        self.reranker.session = ort.InferenceSession(
            f"{RERANK_DIR}/{RERANK_FILE}",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.reranker.name = f"onnx:{RERANK_REPO}/{RERANK_FILE}"

    @modal.asgi_app()
    def web(self):
        import os
        from types import SimpleNamespace

        import httpx
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from starlette.background import BackgroundTask

        token = os.environ["MODELS_TOKEN"]
        api = FastAPI()
        # No timeout: a claim check streams as long as it needs; the pipeline bounds each call.
        client = httpx.AsyncClient(base_url=OLLAMA, timeout=None)  # noqa: S113

        @api.middleware("http")
        async def bearer(request: Request, call_next):
            if request.headers.get("authorization") != f"Bearer {token}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

        @api.post("/rerank")
        def rerank(body: dict) -> dict:
            passages = [SimpleNamespace(text=text) for text in body["documents"]]
            return {
                "scores": self.reranker.score(body["query"], passages),
                "providers": self.reranker.session.get_providers(),
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
