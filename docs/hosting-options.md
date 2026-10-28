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

## Model providers

How the model steps are paid for and where they run is in [the hub plan](hub-plan.md#usage-meters-and-provider-ladders).
