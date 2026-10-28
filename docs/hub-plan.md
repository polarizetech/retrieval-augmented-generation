# The gateway as the hub: plan

Status: phases 1 and 2 built (2026-10-28); phases 3 and 4 are plans. Update this file as phases land,
and record decisions where they are made. How the built parts work: [MCP clients](mcp-clients.md#the-hub-ledger-failures-and-usage).

## Goal

All work, research and code, goes through the gateway (`ask`, merged in #17). The chat model of choice
relays what the gateway did. The gateway should therefore:

1. take several requests at once, organise them, and keep track of what is on the go;
2. catch redundant or conflicting work across sessions and projects;
3. keep work moving when parts break, telling the session plainly what failed, what was skipped, what that
   means for the work, and what to do next;
4. know how much credit or plan allowance each model provider has left, show it, and fall back down a
   ladder of providers to a small local model when everything else is exhausted;
5. keep sessions small, so long projects stay coherent without resending ever-growing context.

## Where it stands (2026-10-28)

- `ask` routes each request to quick retrieval, the research pipeline or the coder by trigger words
  (`config/routing.default.toml` plus a gitignored `config/routing.toml`) and returns a relay block.
- No queue or admission control beyond `PIPELINE_MAX_RUNS`, which refuses rather than queues.
- No timeouts on upstream calls; a slow reranker (about 0.3 s per candidate on CPU) surfaces as a client
  timeout.
- An upstream that dies mid-session stays dead until the gateway restarts; research runs do not survive a
  restart.
- Errors are free text: nothing says what broke, what was skipped or what to do next.
- Sessions that start their own stdio gateway each hold a copy of the paper library (about 1.7 to 3 GB). The
  bridge (`--connect`) avoids this, but client configs can lose the flag (Claude Desktop's did on
  2026-10-28).

## 1. Request ledger: what is on the go

Every request gets an id and a row in a small SQLite database on the gateway: client and session, project
and repository, route, specialist, a fingerprint of the question, the files or units it touches, status,
timings, provider per step, tokens and cost, errors. It survives restarts.

- `hub__status`: what is running or queued now.
- `hub__board`: a readable summary per project.
- `hub__result(id)`: the result of a request that outlived its call.

Because work flows through the gateway, the board maintains itself; no transcripts are read.

## 2. Taking many requests at once

- One limit and queue per scarce resource: retrieval (two at a time; the reranker shares the CPU), research
  runs, model calls per provider, and one code task per repository branch.
- A queued request answers at once with its position and an estimate, never a hang.
- Every call answers within a deadline under the client's MCP timeout. Longer work returns `working` with an
  id for `hub__result`.
- Memory pressure is a resource too: heavy requests queue when the machine is short of memory.

## 3. Overlap and conflict

- **Duplicate research:** a question that matches a running one (normalised text, route and an embedding
  similarity on bge-m3, which the gateway already has) attaches to that run instead of starting another.
- **Code conflicts:** a code task on the same repository and files, or the same app version, is held. The
  reply names the other session and what it is doing.
- **Near overlaps:** run, with a warning.
- **Stale processes:** old app versions and orphaned gateways are reported on the board, never killed by
  the hub.

## 4. Health, circuit breakers and fallbacks

Each component has a health check and a circuit breaker: after repeated failures it is skipped for a
cool-down, then probed. Nothing falls back silently. A degraded result says so in the relay block, the run
log and the capsule.

| Component down | Fallback | What it means for the work |
|---|---|---|
| reranker | retrieval without reranking | faster, less precise passages |
| embeddings (Ollama) | lexical retrieval | misses paraphrases |
| paper library | scholarly search results only | no verbatim passages; claims marked not citable |
| model provider | next provider on the step's ladder (section 6) | weaker model, or the client checking itself; logged |
| verifier | explore strictness; claims marked unverified | no publication-level answers until it is back |
| an upstream process | restart with backoff; resume or fail with a resume hint | a pause |
| GitHub access | code route edits the local checkout only | no PR until it is back |
| research corpus | work continues; evidence propagation queued as pending | cross-study updates later |
| the whole gateway | the bridge reports it and how to restart it | files can still be read; no grounded claims |

## 5. One error contract

Every non-success reply carries: `status` (queued, degraded, refused, failed), `component`, `what_failed`,
`impact`, `fallback_used`, `options`, `recommendation` and `retry_after`. The relay block puts it first in
plain words, so it reaches the person whichever chat model relays it.

## 6. Usage meters and provider ladders

**Meters.** The gateway reads each provider on a timer and logs every model call it makes (provider,
tokens, cost), so it can report the balance, the burn rate per day and week, and the projected run-out.

| Provider | How it is read | Notes |
|---|---|---|
| OpenRouter | `GET /api/v1/key` (usage today, week, month) and `GET /api/v1/credits` (total and used) | works with the existing key; also covers the zero-retention Qwen and Kimi runs |
| NEAR AI, direct | `GET /v1/organizations/{org_id}/usage/balance` and the Usage Reporting API | the API key is refused (401); needs the organisation id and a reporting token from the dashboard, kept in `~/.config/polarize/` |
| Claude plan | only a Claude session can read it (the app's usage reading: 5-hour and weekly windows, extra usage) | the gateway cannot; Claude sessions add their own line |
| ChatGPT plan | no usage API for a consumer plan | shown as not readable; the gateway counts only what it does for ChatGPT |
| Local Ollama | free; whether it is up and which model is loaded | always the last rung |

- `hub__usage` answers "how much have I used this week", "what is my burn rate", "when does OpenRouter run
  out".
- A one-line usage readout under each `ask` relay block. A small panel in the Claude desktop app (a mod)
  can show the same without spending tokens on every reply.

**Provider ladders.** Each pipeline step (planning, extraction, writing, checking) has its own ordered list
of providers, for example writing: OpenRouter Qwen3.8 27B (zero retention), then NEAR AI direct, then the
local model; checking: the local verifier first, since it is free. The gateway moves down a ladder when a
provider fails, runs low (an operator threshold, such as USD 1 left or 95 % used) or is exhausted, and
back up when it recovers or resets. The relay block names the provider each step used.

**Last resort.** `qwen3:4b-instruct-2507` (2.5 GB, already installed) with the local verifier
`bespoke-minicheck` and `bge-m3`. With retrieval supplying the passages it keeps work grounded; its writing
and reasoning are weaker, and its output is labelled. When everything is exhausted the readout says which
provider resets first, and when.

## 7. Keeping sessions small

Long sessions resend their whole history on every turn, which is where most of the cost goes. The hub
should keep each session to what the current task needs.

- **Project capsule.** The gateway keeps one capsule per project: goal and scope, decisions, conventions
  in force, open items, links to PRs and runs. It is updated at each wrap-up (from the ledger and the
  session's own handoff). Any new session, in any chat model, starts from the capsule plus the repository's
  `AGENTS.md`, not from an old transcript. This is the "threads" idea: `hub__resume(project)`.
- **Rules loaded on demand.** Specialist briefs and project rules reach the model only when the router
  picks them (the route's guidance, not every brief). Relay blocks stay compact: passage excerpts and ids,
  with full text fetched by id when needed.
- **A size watcher.** A small scheduled Claude session (cheap model) reads each open session's context
  size with the app's usage reading. Around half full it asks the session to write its capsule update;
  near the auto-compact point it compacts or hands the task to a fresh session seeded with the capsule,
  never in the middle of a task. It reports what it did. The gateway cannot do this itself, because
  compaction and hand-off are app actions; sessions in other chat models rely on the capsule and the rules.
- **Measure the saving.** The ledger records tokens per request, so the board can show context per session
  and the cost before and after.

## Phases

1. **Built** (PR #20). Ledger, error contract, health checks with circuit breakers, labelled fallbacks,
   usage meters, provider ladders with the local last rung. Not yet: the reranker-only and
   embeddings-only fallbacks need `rerank` and `dense` flags on paper-fetch's `retrieve` tool; until then
   they fall back to scholarly search, labelled.
2. **Built.** Queues and limits (retrieval two at a time; research waits in line when the pipeline is
   full; model calls per provider, the local model one at a time), memory pressure, a 45 s reply deadline
   with `hub__result`, duplicate and near-duplicate research, stale processes on the board. Code
   conflicts are not held (decision below).
3. Upstream restart with backoff; research runs that survive a gateway restart.
4. Board views and a daily digest; project capsules and `hub__resume`; the size watcher; checks for
   sessions outside the hub.

Each phase ships with fault-injection tests: every component is broken on purpose, and the fallback and
its message are checked.

## Decisions

Taken 2026-10-28:

- **Code conflicts:** neither blocked nor held. Each session works in its own git worktree; `ask`
  mentions another session's recent code work on the same repository.
- **Degraded fallbacks:** ask once per session per component. The first failure replies
  `needs_decision`; the answer (`fallback` or `stop`) holds for the rest of that session.
- **Session identity:** sessions pass their name and project; otherwise the gateway infers them and
  marks them as inferred.
- **Ladders and thresholds:** planning, extraction and writing go OpenRouter Qwen3.8 27B (zero
  retention), then NEAR AI direct, then local `qwen3:4b-instruct-2507`; checking stays on the local
  verifier. A provider is low at USD 1 left or 95 % of a key limit used.

Waiting:

- What the size watcher may do on its own (capsule update, compact, hand-off) and what it asks first.

## Related

- [Hosting options](hosting-options.md): where things run, and what was decided about paid hosting.
- [MCP clients](mcp-clients.md): `ask`, the bridge and the tools as they are today.
