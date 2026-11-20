# The model service and its fallback

The models the research tool calls (bge-m3 embeddings, bespoke-minicheck claim checks and the
bge-reranker-v2-m3 reranker) are served by one app, `deploy/models_service.py`. Any host with
Ollama and the models can run it:

| host | how | role |
|---|---|---|
| Modal (GPU) | `modal deploy deploy/modal_models.py` | primary |
| the Mac (Ollama with Metal) | `uv run deploy/models_service.py` | fallback |

The droplet's gateway, pipeline and paper-fetch call `deploy/models_router.py` on
`127.0.0.1:8895`, not a model service. The router sends each call to Modal. When Modal answers 402
or 429 (out of budget or limited) or cannot be reached, the router uses the Mac for the next
`MODELS_COOLDOWN_SECONDS` (900 by default), then tries Modal again. `GET /router/status` on the
router says which service it is using; each switch is logged to the router's journal.

## The Mac

- Ollama with `bge-m3` and `bespoke-minicheck` pulled, serving on 127.0.0.1:11434 (the
  `tech.polarize.ollama-serve` LaunchAgent). Ollama unloads a model a few minutes after its last call.
- `uv run deploy/models_service.py` as a LaunchAgent, on 127.0.0.1:8896, with the same key as the
  droplet (`~/.config/polarize/models.key`). The reranker loads on the first /rerank and is dropped
  after `RERANK_IDLE_SECONDS` (600) without one; `RERANK_FILE` picks the ONNX file (default
  `onnx/model.onnx`, the file Modal serves).
- `RERANK_BACKEND=torch` runs the reranker in PyTorch on the Mac's GPU (`RERANK_DEVICE`, default
  `mps`), from `BAAI/bge-reranker-v2-m3` (`RERANK_TORCH_REPO`). Its scores match `onnx/model.onnx`;
  onnxruntime on the Mac's CPU was too slow for the 40 s retrieval limit (about 36 s for 50 passages
  with int8, 5 s on the GPU). The Mac's LaunchAgent sets it.
- Published to the tailnet only: `tailscale serve --bg --https=8896 http://localhost:8896`.

The Mac has to be awake and on the tailnet for the fallback to answer. When neither service
answers, the router returns 503 and the gateway degrades as before.

## The droplet

- `research-rag-models-router.service` runs `uv run deploy/models_router.py` with
  `MODELS_PRIMARY_URL` (the Modal URL) and `MODELS_FALLBACK_URL` (the Mac's tailnet URL on 8896).
- The droplet's Tailscale runs in userspace mode (`--tun=userspace-networking`), so its programs
  have no route to the tailnet. tailscaled also runs its outbound HTTP proxy on loopback only
  (`--outbound-http-proxy-listen=127.0.0.1:1055` in `/etc/default/tailscaled`), and the router
  sets `MODELS_FALLBACK_PROXY=http://127.0.0.1:1055`: only the router's calls to the Mac use it.
  Never bind that proxy to a public address. Paper and web fetches don't use it, so a bad link
  can't reach the tailnet.
- `config/rag.env`: `PIPELINE_OLLAMA_URL=http://127.0.0.1:8895`.
- `~/.config/paper-fetch/retrieval.env`: `PAPER_FETCH_EMBED_URL=http://127.0.0.1:8895/v1` and
  `PAPER_FETCH_RERANK_URL=http://127.0.0.1:8895`.
- The key files stay as they are: the router passes the key through, and both services use it.

## Later

The same app can be packaged as a Docker image (Ollama plus `models_service.py`) to run on the
droplet or anywhere else. On the Mac, Docker has no Metal GPU, so the Mac runs it with `uv` and the
native Ollama instead.
