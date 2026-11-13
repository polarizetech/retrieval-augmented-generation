# Hosting options

A running record of where this stack and the projects around it could run, what each option costs and
what it would mean. Update it when prices, limits or decisions change. Each entry says when it was last
checked; treat anything older than a few months as stale.

## Decision so far (2026-10-28)

- **Develop locally by default.** Projects, the gateway and the paper library run on the operator's
  machine. A private URL for a phone comes from `tailscale serve` on the tailnet, never Funnel.
- **No paid hosting for development.** A cloud droplet (DigitalOcean, about USD 48 a month for 4 vCPU and
  8 GB) was planned and dropped on cost.
- **Offload option, only on request:** GitHub Codespaces on a personal free allowance (below).

## GitHub Codespaces (checked 2026-10-28)

| | |
|---|---|
| Free allowance | 120 core-hours (about 60 hours on 2 cores) and 15 GB storage a month on a personal account; free use is not included for organisation accounts, so organisation repos must bill to the person |
| Cost control | a USD 0 spending limit stops use when the allowance runs out |
| Private URLs | forwarded ports are private by default, behind the GitHub login |
| Always on? | no: a codespace stops after 30 minutes idle (default), and its URLs work only while it runs |
| Secrets | Codespaces secrets in the account, scoped to repositories, never in files |
| Assistants | Claude Code in the devcontainer; the desktop app can reach it through `gh codespace ssh --config` (untested) |

Sources: [Codespaces pricing](https://docs.github.com/ja/billing/managing-billing-for-github-codespaces/about-billing-for-codespaces),
[organisation costs](https://docs.github.com/en/codespaces/managing-codespaces-for-your-organization/managing-the-cost-of-github-codespaces-in-your-organization),
[port forwarding](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace).

## Hugging Face (checked 2026-10-28)

Used today only to download models (the ONNX reranker). Not used for hosting.

| Option | Cost | Fits | Catch |
|---|---|---|---|
| **Spaces** | free tier: 2 vCPU, 16 GB RAM | hosting a project app at a URL; offloading the reranker | sleeps when idle (slow first request); private Spaces on the free tier unconfirmed |
| **Inference Endpoints** | per minute; CPU about USD 0.03/h, small GPU about USD 0.50/h; scale to zero | the reranker on a GPU (retrieval from about 10 s to about 1 s) | paid; cold starts can return errors; sources disagree on whether an idle endpoint still bills |
| **Inference Providers** | USD 0.10 a month free credits (USD 2 with Pro) | little: OpenRouter already routes models with zero-retention routing | small allowance |

What it would mean here:

- **Project apps:** the most promising use. A free Space could serve an app version at a URL without the
  operator's machine running it. Confirm a private Space works on the free tier before relying on it.
- **Reranker:** free CPU is likely no faster than a recent laptop; the gain is memory (about 0.6 GB). Real
  speed needs a paid GPU endpoint.
- **The gateway:** not a fit. It needs the local paper library (several GB), the GitHub login and local
  repositories.
- **Privacy:** requests and data leave the machine, and Hugging Face is not a zero-data-retention host.

Sources: [pricing](https://huggingface.co/pricing), [Spaces](https://huggingface.co/docs/hub/spaces-overview),
[Inference Providers pricing](https://huggingface.co/docs/inference-providers/pricing),
[Inference Endpoints pricing, 2026](https://www.spheron.network/blog/hugging-face-inference-endpoints-pricing-2026/),
[how Endpoints bill](https://lyceum.technology/magazine/hugging-face-inference-endpoints-cost-vs-serverless-gpu/).

Open checks: a private Space on the free tier (does it stay private, how long is the wake-up); whether
an idle Inference Endpoint at zero replicas bills.

## Claude Code cloud sessions (checked 2026-11-13)

Claude Code sessions on Anthropic's machines, started from claude.ai/code, the desktop app or
`claude --cloud`. A coding workspace, not hosting.

| | |
|---|---|
| Cost | no separate charge for the machine: each session counts against the Claude plan's usage limits (Pro, Max, Ultra, Team, Enterprise), shared with the rest of Claude; parallel sessions use them up faster |
| Machine | a new Ubuntu 24.04 VM per session, about 4 vCPU, 16 GB RAM and 30 GB disk; `uv`, Python, pytest, ruff and Docker installed |
| Lifetime | pauses after a few minutes idle with files kept; a paused VM can be reclaimed, and reopening then starts a new one (the conversation returns, background commands do not) |
| Ports | none: no shell for the person and no port exposure, so it cannot serve the gateway or an app |
| Network | None, Limited (default: GitHub, package registries, Anthropic) or Full; other hosts such as `openrouter.ai` are added to the environment's list |
| Secrets | environment variables are readable by anyone using the environment; "network secrets" are added by Anthropic's proxy for listed hosts and never enter the VM |
| GitHub | the Claude GitHub App on the organisation (private repos, Auto-fix), or `/web-setup`, which stores the `gh` token with Anthropic; the GitHub proxy blocks GraphQL, so `gh pr` and `gh issue` fail and `gh api` works |
| Retention | transcripts kept: 30 days, or 5 years on a consumer account that allows training; not zero retention |
| Unpublished | how many sessions can run at once, and the longest a session can run |

What it would mean here:

- **Setup:** install the Claude GitHub App on polarizetech; create an environment (Limited network, plus
  `openrouter.ai` if runs should call models; keys as network secrets); a setup script that runs `uv sync`,
  under about five minutes so it is cached.
- **What carries over:** the repo's `CLAUDE.md`, `.claude/settings.json`, `.mcp.json`, skills and agents.
  The person's own `~/.claude/CLAUDE.md`, user-level MCP servers and plugins do not, so the
  retrieval-first rule must be in the repo to apply there.
- **This machine is out of reach:** the VM cannot see localhost, the tailnet or local disks, so SearXNG, the
  paper library, Ollama and the polarize checkouts are unavailable. Code changes and the tests fit; research
  runs do not.
- **research-rag still works** as a claude.ai connector (it is on the public MCP host, and connectors go
  through Anthropic's servers), while this machine is awake.
- **Remote Control** is the alternative that keeps everything local: a session runs on this machine and is
  driven from a phone or the web, on the same plan, with outbound connections only.

Sources: [cloud sessions](https://code.claude.com/docs/en/claude-code-on-the-web.md),
[cloud environments](https://code.claude.com/docs/en/cloud-environments.md),
[quickstart](https://code.claude.com/docs/en/web-quickstart.md),
[Remote Control](https://code.claude.com/docs/en/remote-control.md),
[data usage](https://code.claude.com/docs/en/data-usage.md), [pricing](https://claude.com/pricing).

## Model providers

How the model steps are paid for and where they run is in [the hub plan](hub-plan.md#usage-meters-and-provider-ladders).
