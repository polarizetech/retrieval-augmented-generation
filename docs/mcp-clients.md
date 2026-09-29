# MCP clients

One gateway process (`research-rag-gateway`) exposes:

| Tools | Source | Uses a model? |
|---|---|---|
| `rag__search`, `rag__retrieve_evidence`, `rag__check_citations`, `rag__save_report` | this repository | no (embedding only, for hybrid retrieval) |
| `papers__*` | the paper library upstream | no |
| `pipeline__research_start`, `_continue`, `_status` | the pipeline upstream | yes: by default the calling model; optionally a local Ollama model |
| `web__search` | an optional SearXNG instance | no |
| `gateway_status` | the gateway | no; lists connected upstreams and startup errors |

Upstreams are stdio MCP servers listed in `config/mcp-gateway.json`. The gateway spawns them once,
namespaces their tools as `<upstream>__<tool>`, forwards calls unchanged, and adds MCP tool
annotations (`readOnlyHint`, `destructiveHint`, `openWorldHint`) from the config so that clients
can decide what needs approval. One broken upstream is reported by `gateway_status` and does not
take the others down.

## Claude Desktop and Claude Code

Use stdio: nothing listens on a port and no token is involved.

```json
{
  "mcpServers": {
    "research-rag": {
      "command": "/absolute/path/to/scientific-research-rag/.venv/bin/research-rag-gateway",
      "args": ["--config", "/absolute/path/to/scientific-research-rag/config/mcp-gateway.json"]
    }
  }
}
```

An app bundle does not inherit a login shell's `PATH`. If an upstream is started by name (for
example `paper-fetch-mcp`), give its absolute path in the gateway config or add an `env.PATH` entry.

## The full pipeline, with the client as its model

`pipeline__research_start(question)` starts a run and returns its first step. Code runs every
stage: planning, searching, fetching, retrieval, reranking, quote and number checks, grading, the
retraction check and the run log. Whenever a stage needs a model, the result has status `tasks`:

```json
{"status": "tasks", "instructions": "...", "prompts": {"extract": "..."},
 "tasks": [{"task_id": "t4", "task": "extract", "input": "...", "output_schema": {...}}]}
```

The client answers every task from its input alone, as JSON matching `output_schema`, and sends
them back with `pipeline__research_continue(run_id, answers=[{task_id, result}])`. Each answer is
validated against its schema before the run uses it; a refused answer is returned with the error
and asked again (three invalid answers end the run). Status `working` means the run is searching
or fetching: call `research_continue(run_id, [])` to wait. Status `done` carries the answer.

Independent calls are batched, at most `PIPELINE_CLIENT_BATCH` (12) per turn. A typical question
takes about a dozen turns: plan, extraction (with paper classification), synthesis, the
falsification pass, critique, verification and the summary check.

- The client's model is the text model and a verifier. Nothing can see which model a client runs,
  so the run log records the client's self-reported name, not a model digest.
- It still checks its own claims, and every answer says so. Set `PIPELINE_VERIFIER_MODEL` to a
  local model to add an independent verifier that must also accept each claim.
- Ollama still computes embeddings (a chat model cannot produce the index's vectors).
- `PIPELINE_MCP_LLM=ollama` makes the local model answer instead; the same two tools then simply
  wait for the run.

## The evidence tools on their own

1. `rag__search` to find papers; `papers__fetch` to add open-access full text to the library, then
   `research-pipeline index` (or the pipeline itself) to index it.
2. `rag__retrieve_evidence` for verbatim passages. Only these passages may ground a claim.
3. Write each claim with the evidence ids it relies on and a verbatim quote from at least one of
   them.
4. `rag__check_citations` on every claim. Remove or fix any claim it rejects.
5. `rag__save_report` with the claims, so the check is stored with the report.

These tools check evidence ids, quotes and numbers. They do not check that a passage entails a
claim, and a client relaying their results should say so. For that, use the pipeline.

## Why the pipeline is turn-based

Code, not the model, decides the order of steps: a model asked to chain `papers__search`,
`papers__fetch` and `papers__text` itself may skip the falsification searches or the checks, and a
small model does so reliably (see the [design document](research-pipeline.md#the-problem)). A run
also takes minutes, longer than a client will hold one call open, so no call waits more than
`PIPELINE_CLIENT_TURN_WAIT` (40) seconds. MCP "sampling", where a server asks the client's model
for a completion mid-call, would avoid the turns, but the clients this targets do not offer it.

## HTTP transport

```bash
./scripts/mcp-start.sh      # streamable HTTP on 127.0.0.1:$MCP_PORT, path /<token>/mcp
./scripts/mcp-url.sh        # print the URL
./scripts/mcp-stop.sh
```

The gateway binds to loopback and must stay there: it fronts the paper library's write tool and
any tools other upstreams add. `mcp-start.sh` refuses a non-loopback `MCP_HOST` unless
`MCP_ALLOW_PUBLIC_BIND=1` is set. The path token (generated into `config/.mcp-token`, mode 0600)
is not authentication in any strong sense: without OAuth (below), keep the endpoint local.

DNS-rebinding protection stays on. A front end that presents a different `Host` header (a Docker
bridge, a reverse proxy) is allowlisted under `network.allowed_hosts` in the gateway config.

If the gateway process is killed hard, its stdio children can outlive it. The HTTP gateway records
their PIDs in `run/mcp-children.pid`, and `mcp-start.sh` reaps them on the next start.

## Remote clients: ChatGPT, Claude web and mobile

Remote clients need a public HTTPS URL and OAuth. Set `MCP_PUBLIC_URL` and the HTTP gateway
becomes a single-user OAuth 2.1 server (`research_mcp/oauth.py`): clients register themselves,
authorize with PKCE, and the owner approves each connection on a consent page by entering one
private access key (`MCP_OAUTH_ACCESS_KEY`, at least 32 characters). Clients then hold one-hour
access tokens and rotating 30-day refresh tokens, bound to this server's URL; tokens are stored
hashed in `MCP_OAUTH_STATE` (mode 0600). Deleting that file revokes every connection.

| Variable | Meaning |
|---|---|
| `MCP_PUBLIC_URL` | the public base URL, e.g. `https://mcp.example.org/scientific-research-rag` |
| `MCP_RESOURCE_PATH` | where the MCP endpoint sits under it: `/mcp` (default), or empty when a front end mounts the server under a path and strips the prefix |
| `MCP_OAUTH_ACCESS_KEY` | the consent key; keep it in a secret store, not in a file in the repo |
| `MCP_OAUTH_STATE` | the token store (default `run/oauth.sqlite`) |

The gateway still binds to loopback. Put a TLS front end in front of it, for example Tailscale
Funnel. When the server is mounted under a path, OAuth discovery also needs two root-level paths:

| Public path | Forward to the gateway's |
|---|---|
| `/<path>` | `/` (prefix stripped) |
| `/.well-known/oauth-protected-resource/<path>` | same path |
| `/.well-known/oauth-authorization-server/<path>` | `/.well-known/oauth-authorization-server` |

In ChatGPT (developer mode) or Claude, add a custom connector with the public URL and OAuth
authentication, and leave the client ID and secret empty.
