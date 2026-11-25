# The model service and its fallback

The models the research tool calls (bge-m3 embeddings, bespoke-minicheck claim checks and the
bge-reranker-v2-m3 reranker) are served by one app, `deploy/models_service.py`. Any host with
Ollama and the models can run it:

| host | how | role |
|---|---|---|
| the Mac (Ollama with Metal) | `uv run deploy/models_service.py` | first (default) |
| Modal (GPU) | `modal deploy deploy/modal_models.py` | next |
| DigitalOcean (serverless inference) | `MODELS_<N>_KIND=digitalocean`, `https://inference.do-ai.run` | reranking and embeddings only |

The droplet's gateway, pipeline and paper-fetch call `deploy/models_router.py` on
`127.0.0.1:8895`, not a model service. The router tries providers in the order `MODELS_PROVIDERS`
gives (default `local,modal`). A provider that answers 402 or 429, or can't be reached, sits out for
`MODELS_COOLDOWN_SECONDS` (900 by default) while the next one serves. When all are out, all are
still tried in order. `GET /router/status` shows the order, the provider in use and each provider's
remaining cooldown; each switch is logged to the router's journal.

A call can choose its provider with the `X-Models-Route` header: `mac` (the `local` provider),
`modal`, or a provider's name. Only that provider is tried; when it does not answer the call
gets 503, not the next provider. paper-fetch sends the header when `ask` is given
`providers={"embed": "modal"|"mac", "rerank": "modal"|"mac"}`. `rerank: "droplet"` runs the
library's ONNX cross-encoder in paper-fetch's own process (`PAPER_FETCH_RERANK_LOCAL_MODEL`),
`"off"` skips reranking, and `"llm"` has the pipeline's text model score the passages
(research_mcp/llm_rerank.py). There is no droplet embedding model. These two steps apply to a
quick or code ask's retrieval only; a research run ignores them. To keep reranking off by default
while allowing it per request, set the rerank variables in `~/.config/paper-fetch/retrieval.env`
and add `PAPER_FETCH_RERANK_DEFAULT=off`.

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
- Memory: on a 16 GB Mac, bespoke-minicheck (about 7 GB in Ollama) and the GPU reranker (about
  6 GB) together left 8% free and pushed the Mac into swap. Keep `RERANK_IDLE_SECONDS` at its 600
  default rather than `inf`, and `ollama stop bespoke-minicheck` frees the claim checker at once.
- Published to the tailnet only: `tailscale serve --bg --https=8896 http://localhost:8896`.

The Mac has to be awake and on the tailnet for the fallback to answer. When neither service
answers, the router returns 503 and the gateway degrades as before.

## DigitalOcean

- The router translates `/rerank`, `/v1/embeddings` and Ollama's `/api/embed` into DigitalOcean's
  `/v1/rerank` and `/v1/embeddings` (models `bge-reranker-v2-m3` and `bge-m3`). It scales its
  embedding vectors to unit length, to match Ollama's.
- Every other call (the claim checker's `/api/chat`, `/api/tags`, `/api/ps`) goes to the next
  provider.
- 401, 402, 403 and 429 put it out for the cooldown.
- Its key is a DigitalOcean model access key in `MODELS_<N>_KEY_FILE`, sent in place of the caller's
  key.
- The droplet's order would be `MODELS_PROVIDERS=digitalocean,local,modal`.
- Checked by us on 2026-11-23 on 100 passages from the paper index: the reranker matched the Mac's
  full-precision reranker to within 1.1e-5, with the same order and top 5; bge-m3 matched Ollama's
  at a cosine similarity of at least 0.9996.

## The droplet

- `research-rag-models-router.service` runs `uv run deploy/models_router.py` with
  `MODELS_PROVIDERS`, `MODELS_<NAME>_URL` and `MODELS_<NAME>_PROXY` (for example `MODELS_LOCAL_URL`
  set to the Mac's tailnet URL on 8896, `MODELS_MODAL_URL` set to the Modal URL, and
  `MODELS_LOCAL_PROXY=http://127.0.0.1:1055`). The old `MODELS_PRIMARY_URL`, `MODELS_FALLBACK_URL`
  and `MODELS_FALLBACK_PROXY` still work and mean modal then local.
- The droplet's Tailscale runs in userspace mode (`--tun=userspace-networking`), so its programs
  have no route to the tailnet. tailscaled also runs its outbound HTTP proxy on loopback only
  (`--outbound-http-proxy-listen=127.0.0.1:1055` in `/etc/default/tailscaled`), and the router
  sets `MODELS_LOCAL_PROXY=http://127.0.0.1:1055`: only the router's calls to the Mac use it.
  Never bind that proxy to a public address. Paper and web fetches don't use it, so a bad link
  can't reach the tailnet.
- `config/rag.env`: `PIPELINE_OLLAMA_URL=http://127.0.0.1:8895`.
- `~/.config/paper-fetch/retrieval.env`: `PAPER_FETCH_EMBED_URL=http://127.0.0.1:8895/v1` and
  `PAPER_FETCH_RERANK_URL=http://127.0.0.1:8895`.
- The key files stay as they are: the router passes the key through, and both services use it.

## Cost

Checked 2026-11-21 on modal.com/pricing and digitalocean.com/pricing/gpu-droplets:

- Modal Starter, the plan in use: $0 a month plus compute, with $30 of free compute a month. Team is
  $250 a month plus compute ($100 credit). An L4 is $0.000222 a second, so the credit buys about 37
  L4-hours; with `scaledown_window=300`, each cold wake costs at least about 7 cents.
- In November 2026 every ask and every session start woke the GPU, so it ran about 20 hours a day
  and spent about $40 before Modal started answering 429; since then the router has used the Mac.
  PR #64 wakes it only for asks that use the models.
- DigitalOcean's cheapest GPU Droplet (RTX 4000 Ada) is $0.76 an hour, about $511 a month at its
  672-hour monthly cap, and a powered-off Droplet is still billed, so it was not used.
- DigitalOcean serverless inference (its pricing docs, checked 2026-11-23, last verified by
  DigitalOcean on 18 Nov 2026) charges $0.01 per million tokens for bge-reranker-v2-m3 and $0.02 per
  million for bge-m3, prepaid only. Reranking 20 passages took about 6,000 tokens in our check.

## Later

The same app can be packaged as a Docker image (Ollama plus `models_service.py`) to run on the
droplet or anywhere else. On the Mac, Docker has no Metal GPU, so the Mac runs it with `uv` and the
native Ollama instead.
