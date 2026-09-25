# MCP clients

One gateway process (`research-rag-gateway`) exposes:

| Tools | Source | Uses a model? |
|---|---|---|
| `rag__search`, `rag__retrieve_evidence`, `rag__check_citations`, `rag__save_report` | this repository | no (embedding only, for hybrid retrieval) |
| `papers__*` | the paper library upstream | no |
| `pipeline__research_start`, `_status`, `_result` | the optional pipeline upstream | yes: the local text and verifier models |
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
      "command": "/absolute/path/to/kit-scientific-research-rag/.venv/bin/research-rag-gateway",
      "args": ["--config", "/absolute/path/to/kit-scientific-research-rag/config/mcp-gateway.json"]
    }
  }
}
```

An app bundle does not inherit a login shell's `PATH`. If an upstream is started by name (for
example `paperlib-mcp`), give its absolute path in the gateway config or add an `env.PATH` entry.

## The intended workflow for a client model

1. `rag__search` to find papers; `papers__fetch` to add open-access full text to the library, then
   `research-pipeline index` (or the pipeline itself) to index it.
2. `rag__retrieve_evidence` for verbatim passages. Only these passages may ground a claim.
3. Write each claim with the evidence ids it relies on and a verbatim quote from at least one of
   them.
4. `rag__check_citations` on every claim. Remove or fix any claim it rejects.
5. `rag__save_report` with the claims, so the check is stored with the report.

These tools check evidence ids, quotes and numbers. They do not check that a passage entails a
claim, and a client relaying their results should say so. For that, use the pipeline.

## Why the pipeline is start/poll/collect

Small local models format a single tool call well and sequence several badly (see the
[design document](research-pipeline.md#the-problem)). A local model should not answer a literature
question by chaining `papers__search`, `papers__fetch` and `papers__text` itself. With the
`pipeline` upstream enabled it calls `pipeline__research_start` once, polls
`pipeline__research_status`, and reads `pipeline__research_result`; the multi-step work runs in
code. A run takes minutes, longer than a client will hold one call open.

## HTTP transport

```bash
./scripts/mcp-start.sh      # streamable HTTP on 127.0.0.1:$MCP_PORT, path /<token>/mcp
./scripts/mcp-url.sh        # print the URL
./scripts/mcp-stop.sh
```

The gateway binds to loopback and must stay there: it fronts the paper library's write tool and
any tools other upstreams add. `mcp-start.sh` refuses a non-loopback `MCP_HOST` unless
`MCP_ALLOW_PUBLIC_BIND=1` is set. The path token (generated into `config/.mcp-token`, mode 0600)
is not authentication in any strong sense; for remote access, put an authenticating proxy in
front of loopback rather than widening the bind.

DNS-rebinding protection stays on. A front end that presents a different `Host` header (a Docker
bridge, a reverse proxy) is allowlisted under `network.allowed_hosts` in the gateway config.

If the gateway process is killed hard, its stdio children can outlive it. The HTTP gateway records
their PIDs in `run/mcp-children.pid`, and `mcp-start.sh` reaps them on the next start.
