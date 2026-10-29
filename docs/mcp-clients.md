# MCP clients

One gateway process (`research-rag-gateway`) exposes:

| Tools | Source | Uses a model? |
|---|---|---|
| `ask` | this repository: routes a request and runs only what it needs | no for routing and retrieval; the research route starts the pipeline |
| `rag__search`, `rag__retrieve_evidence`, `rag__check_citations`, `rag__save_report` | this repository | no (embedding only, for hybrid retrieval) |
| `papers__*` | the paper library upstream | no |
| `pipeline__research_start`, `_continue`, `_status` | the pipeline upstream | yes: your configured endpoint when set, else the calling model; `model="client"` hands one run to the caller |
| `datasets__*`, `patents__*` | dataset-fetch and patent-fetch adapters | no |
| `session__read_capsule`, `session__propose` | session store and research PR gate | no |
| `research__*` | trusted research-repository reads and proposals | no |
| `repos__*` | allowlisted GitHub repository search, reads and PR proposals | no |
| `web__search` | an optional SearXNG instance; listed only when `SEARXNG_URL` is set | no |
| `gateway_status` | the gateway | no; lists connected upstreams and startup errors |

Upstreams are stdio MCP servers listed in `config/mcp-gateway.json`. The gateway spawns them once,
namespaces their tools as `<upstream>__<tool>`, forwards calls unchanged, and adds MCP tool
annotations (`readOnlyHint`, `destructiveHint`, `openWorldHint`) from the config so that clients
can decide what needs approval. One broken upstream is reported by `gateway_status` and does not
take the others down.

**Web search without SearXNG.** The scholarly indexes rarely miss an open copy (1.2% of searches
across 265 runs, October 2026), so no web search engine is required. When `papers__search` wants
the web fallback and no SearXNG is configured, its result carries `ask_the_web`: a client that can
search the web does so itself and passes the URLs it finds to `papers__leads`, which parses paper
identifiers out of them; `papers__fetch` then lets the open-access providers decide what may be
kept. Nothing is downloaded from a web page. Runs whose model has no web access (your endpoint,
Ollama) skip the fallback and say so in their Limits section.

## A second instance for other upstreams

The same gateway binary can front a different set of stdio servers under its own name, with its
own OAuth state, port and public URL. Add a `server` section to that instance's config:

```json
{"server": {"name": "scientific-research-news", "instructions": "...", "builtin_tools": false},
 "upstreams": {"news": {"command": "/abs/path/python", "args": ["/abs/path/serve.py"]}}}
```

`builtin_tools: false` drops `web__search` and the `rag__*` tools, so the instance lists only
`gateway_status` and its upstreams. Omitting the section keeps today's behaviour.

## One entry point: `ask`

Every request can go through `ask(question, repo?)`, and the server instructions tell clients to start
there. It decides which parts of the server run, using trigger words rather than a model:

| Route | When | What runs |
|---|---|---|
| `quick` (default) | a question with no research or code trigger | one passage retrieval (five passages) |
| `research` | an evidence question ("evidence", "studies", "is it true", ...) | the research pipeline, started on the gateway |
| `code` | a task on code ("fix the bug", "implement", "add a flag to the cli") or a code review | three passages for context; the assistant then edits through `repos__` tools or its checkout |

A question never recruits the coder, even a question about code ("how does the hook work?"). A
prefix decides outright: `quick:`, `research:`, `deep:`, `code:`, `review:`. The specialist brief
is chosen by `pipeline__profile_route`, or by the operator's own words. The words are in
`config/routing.default.toml`; put your own in the gitignored `config/routing.toml` in the same
shape. Its lists are added to the defaults, and a `[profiles]` entry names the specialist its
words pick. The file is read on every call, so edits apply without a restart.

`ask` returns `relay`, a Markdown block for the client to show unchanged. It says what ran and why,
lists the passages with their evidence ids, and ends with a Limits line. Anything the client adds
goes after it, labelled as its own. The gateway does not make code changes itself yet: on the code
route it gives context and names the tools, and the assistant makes the change.

## The hub: ledger, failures and usage

Phase 1 of the [hub plan](hub-plan.md) is built. `ask` takes three more optional arguments:
`session` and `project` (else inferred: the stdio bridge sends its own id and folder, a direct
HTTP client is known by its MCP client name, and the ledger marks the session as inferred) and
`decisions`, described below.

**Ledger.** Every `ask` gets a `request_id` and a row in `data/pipeline/hub.sqlite` (`HUB_LEDGER`
moves it): client, session, project, repository, route, specialist, question fingerprint, the
files it names, status, timings, problems, and for a research run its provider, tokens and cost
once the run's calls pass through the gateway. `hub__status` lists what is running and each
component's health and circuit breaker; `hub__board` summarises per project; `hub__result(id)`
returns a request's result, including a research run's answer after the call that started it.

**One error contract.** Every non-success reply carries `status` (`queued`, `degraded`, `refused`,
`failed` or `needs_decision`), `component`, `what_failed`, `impact`, `fallback_used`, `options`,
`recommendation` and `retry_after`. The relay puts it first in plain words; tool errors carry the
same JSON after their text, and upstream errors keep their own text first.

**Failures and fallbacks.** Each component (reranker, embeddings, paper library, model provider,
verifier, upstream process, GitHub, research corpus) has a check and a circuit breaker: three
failures in a row and it is skipped for two minutes, then probed. The first time a component
fails in a session, `ask` stops and asks (`needs_decision`) whether to `fallback` or `stop`; the
client passes the answer back as `decisions={"<component>": "fallback"}`, and it holds for the
rest of that session. Fallbacks follow the plan's table. Two need a paper-library change first:
switching off only the reranker or only the embeddings (the library's `retrieve` tool would need
`rerank` and `dense` flags; the gateway sends them as soon as the tool lists them). Until then
those two fall back to scholarly search results, and the relay says so. A part the library dropped
by itself (no reranker, lexical only) is labelled as detected. An upstream that is down fails
with the restart command; restarting it with backoff is phase 3.

**Usage.** `hub__usage` reads OpenRouter (`/api/v1/key` and `/api/v1/credits`: balance, spend
today, this week and this month, burn per day and projected run-out), NEAR AI (needs the
organisation id and a reporting token in `~/.config/polarize/nearai-reporting.env`; the API key is
refused), local Ollama (up, installed, loaded), and says which plans it cannot read (Claude,
ChatGPT). Readings are cached for five minutes, and each relay ends with a one-line readout from
the cache, so a reply never waits on a provider.

**Provider ladders.** `config/hub.default.toml` (your own values in the gitignored
`config/hub.toml`) gives each pipeline step its providers in order: planning, extraction and
writing go OpenRouter Qwen3.8 27B (zero retention), then NEAR AI direct, then the local
`qwen3:4b-instruct-2507`. A provider counts as low at USD 1 left or 95 % of a key limit used.
Checking stays on the local verifier and is not laddered: when it is down, runs drop to explore
strictness and claims are marked unverified. Inside a run, a failing or low provider moves the
step down at once and the run log's notes say so; before a run starts, `ask` checks the ladder
and asks the session once, as above.

**Reply layout.** How replies look is set by `config/replies.default.md`: the quick `ask` reply,
the reply while a request is queued or working, and the research pipeline's finished answer. To
change one, copy the file to the gitignored `config/replies.md` and edit it; a template left out
of your copy keeps the default. `{{name}}` places a block (passages, steps, limits, sources and so
on; the file lists them), and `{{#name}}...{{/name}}` shows its inside only when that block has
something in it. Code still writes every block, so quotes, evidence ids and sources are never
retyped; the template only picks order, headings and what appears. Edits show on the next reply,
with no restart.

**Queues and deadlines (phase 2).** Every `ask` answers within 45 s (`[deadlines]` in the hub
config). Work that takes longer carries on and replies `working`, or `queued` with its position and
an estimate; its result lands in the ledger for `hub__result`. Retrieval runs two at a time (the
reranker shares the CPU). A research request the pipeline turns away (`PIPELINE_MAX_RUNS`) waits in
line and starts when a slot frees. Model calls are limited per provider across every run (the local
model one at a time). Retrieval and research wait while less than 10 % of memory is free.
`hub__status` shows the queues.

**Duplicates (phase 2).** A research question that matches one on the go (the same words, or a
bge-m3 similarity of 0.92 or more) attaches to it instead of starting a second run; its
`hub__result` returns that run's answer. From 0.80 it runs with a warning naming the other request.
Without embeddings only the same words attach. `hub__board` also lists stale processes (stdio
gateways holding their own library copy, old versions, orphaned upstreams), reported and never
killed.

Decisions taken 2026-10-28: code work by two sessions on the same repository is neither blocked
nor held; each works in its own git worktree, and `ask` mentions the other session's recent
work. Fallbacks ask once per session per component. Sessions pass their name or the gateway
infers it. The ladders and thresholds are as above.

The full plan, with its later phases, is in [hub-plan.md](hub-plan.md); where things can run is
in [hosting-options.md](hosting-options.md).

## Claude Desktop and Claude Code

Use stdio: nothing listens on a port and no token is involved.

```json
{
  "mcpServers": {
    "research-rag": {
      "command": "/absolute/path/to/retrieval-augmented-generation/.venv/bin/research-rag-gateway",
      "args": ["--config", "/absolute/path/to/retrieval-augmented-generation/config/mcp-gateway.json"]
    }
  }
}
```

An app bundle does not inherit a login shell's `PATH`. If an upstream is started by name (for
example `paper-fetch-mcp`), give its absolute path in the gateway config or add an `env.PATH` entry.

### Many sessions: share one gateway

Each stdio gateway starts its own upstreams, and the paper library alone holds its passage vectors
and a reranker (well over a gigabyte) in every process. With many Claude sessions open, that is
one copy per session, and retrieval slows into timeouts as the machine swaps. When an HTTP gateway
is already running on this machine (as for remote access below), add `--connect` and each session
becomes a small proxy to it instead:

```json
"args": ["--config", "/absolute/path/to/config/mcp-gateway.json",
         "--connect", "http://127.0.0.1:8891/"]
```

The bridge passes tool names, schemas and instructions through unchanged, so clients see the same
tools. It authenticates with the local token in `config/.mcp-token`. A gateway with OAuth (with
`MCP_PUBLIC_URL` set) also accepts that token, but only on requests addressed to `127.0.0.1` or
`localhost` that Tailscale Funnel has not marked; through the public URL the token is refused. A call
made while the gateway restarts fails on its own, and the next one works. If the gateway is not
running when a session starts, the bridge waits about 30 seconds and then reports how to start it.

## The full pipeline

`pipeline__research_start(question)` starts a run and returns its first step. Code runs every
stage: planning, searching, fetching, retrieval, reranking, quote and number checks, grading, the
retraction check and the run log.

**Who does the model steps.** By default, the provider configured on the server
(`PIPELINE_MCP_LLM=auto`): your own endpoint when `PIPELINE_ENDPOINT_*` is set (see
[Your own endpoint](research-pipeline.md#your-own-endpoint-for-the-model-steps)), otherwise the
calling model. With a provider, `research_start` and `research_continue` simply return `working`
until the run is `done`. To have the calling assistant do the writing for one run, for example
when you ask Claude to run the pipeline itself, it passes `model="client"`; `"endpoint"` and
`"ollama"` force those. Every reply carries `model_backend`.

### With the client as its model

Whenever a stage needs a model, the result has status `tasks`:

```json
{"status": "tasks", "instructions": "...", "prompts": {"extract": "..."},
 "tasks": [{"task_id": "t4", "task": "extract", "input": "...", "output_schema": {...}}]}
```

The client answers every task from its input alone, as JSON matching `output_schema`, and sends
them back with `pipeline__research_continue(run_id, answers=[{task_id, result}])`. Each answer is
validated against its schema before the run uses it; a refused answer is returned with the error
and asked again (three invalid answers end the run). Status `working` means the run is searching
or fetching: call `research_continue(run_id, [])` to wait. Status `done` carries the answer.

Pass `strictness="explore"`, `"grounded"` or `"publication"`. Publication mode requires an
independent verifier; `PIPELINE_VERIFIER_MODEL` may name a local verifier even when the client is
the drafting model. The pipeline deterministically routes the question through the versioned
scientific agent briefs. An optional domain policy is applied only when its package is installed;
the suggested route and any generic-policy fallback are recorded.

Set `PIPELINE_AGENT_PROFILES_DIR` to the local scientific-research-agents checkout and, when you
have a private specialist overlay, set `PIPELINE_AGENT_EXTENSIONS_DIR` to that checkout in the
gitignored `config/rag.env`. The router reads `agents/*.md` from both, selects a matching
`extends: researcher` brief, and records each selected file's hash and repository commit. Only
the brief's bounded `## Pipeline guidance` section shapes search planning and claim synthesis;
the pipeline's evidence and verification rules still control the result. Use
`pipeline__profile_route(question)` to inspect selection before a run. A specialist without that
section is still routed and recorded but adds no model guidance.

Independent calls are batched, at most `PIPELINE_CLIENT_BATCH` (12) per turn. A typical question
takes about a dozen turns: plan, extraction (with paper classification), synthesis, the
falsification pass, critique, verification and the summary check.

- The client's model is the text model and a verifier. Nothing can see which model a client runs,
  so the run log records the client's self-reported name, not a model digest.
- It still checks its own claims, and every answer says so. Set `PIPELINE_VERIFIER_MODEL` to a
  local model to add an independent verifier that must also accept each claim.
- Ollama still computes embeddings (a chat model cannot produce the index's vectors).
- `PIPELINE_MCP_LLM=client` makes the calling model the default even with an endpoint configured;
  `PIPELINE_MCP_LLM=ollama` makes the local model answer instead.

Every completed run stores `answer.md`, `run.json`, `session.json` and `capsule.md`. Use
`session__read_capsule` for the compact checked record. `session__propose` can place that capsule
on a branch and open a PR in the research repository; it never promotes content directly.

## Repository boundary

`repos__list_repositories`, `repos__search_code` and `repos__read_file` operate only inside
`GITHUB_MCP_ORGS`. Authentication is GitHub CLI's login, or, with `GITHUB_MCP_TOKEN_FILE`, a token
of the tool's own: a fine-grained token limited to the repositories it may change (Contents and
Pull requests read/write, Metadata read, no Workflows), kept outside any repository. With
`GITHUB_MCP_ALLOW_WRITE=1`, `repos__propose_files` accepts complete text replacements, creates a
new `agent/*` branch and opens a pull request. There is deliberately no merge, deletion, settings,
or default-branch-write tool.

A proposal is written by a model that may have read untrusted text, so `propose_files` refuses:

- any path with a `.git` component, since git's own files can make the next git command run a
  program on the host;
- `.github/workflows/` and `.github/actions/`, since a workflow on a pushed branch runs with the
  repository's secrets before anyone reviews the PR;
- writing through a symbolic link.

Paths are compared case-insensitively, and every git command runs with hooks disabled. A
workflow that already runs on every push still runs the proposed code with whatever secrets it
can reach; keep deploy secrets in GitHub environments that require approval.

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
| `MCP_PUBLIC_URL` | the public base URL, e.g. `https://mcp.example.org/retrieval-augmented-generation` |
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
