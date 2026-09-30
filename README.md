# retrieval-augmented-generation

Evidence-first literature research with small local language models. Code controls search,
evidence, verification, abstention and the run log; the model only fills in narrow,
schema-constrained forms. Every claim in an answer cites a verbatim passage from a full-text paper,
and claims that fail a check are removed and counted rather than softened.

The method is described in
[A RAG protocol that makes the model show its work](https://polarize.tech/blog/a-rag-protocol-that-makes-the-model-show-its-work/).
The [design document](docs/research-pipeline.md) gives the rationale, the literature behind each
rule, and a table mapping every rule to the code that enforces it and the test that checks it.

> [!IMPORTANT]
> This does not make a language model truthful. It narrows the paths by which unsupported claims
> reach an answer and makes the remaining uncertainty visible. Read the
> [known limits](docs/research-pipeline.md#known-limits) before relying on an answer.

## What it does

There are two ways to use it:

- **The pipeline** (`research-pipeline ask`) answers a question end to end with a local model:
  plan, search, fetch open-access full text, retrieve and rerank passages, extract evidence,
  synthesise claims, search again for null results, verify, grade, and write `answer.md` plus a
  `run.json` that records every search, passage, model digest and verdict.
- **The evidence tools** (`rag__*` over MCP) let a stronger client model (Claude, ChatGPT) write the
  answer while this repository holds it to the same deterministic rules: evidence only from the
  local index, quotes that must occur in the cited passage, and numbers that must occur in the
  cited evidence. These tools do not judge entailment; the pipeline's verifier models do.

What each rule guarantees, and what it does not:

| Rule | Enforced by |
|---|---|
| A claim can only cite evidence retrieved in this run | synthesis schema enum, then checked in code |
| A quote must occur in the stored passage | `verify.anchor_quote`: exact match, or a near-identical repair that may not change a negation or a number |
| A number in a claim must occur in its cited evidence, with a matching unit where the field declares units | `verify.unsupported_numbers`; a failing claim is removed |
| Verifier disagreement is shown, not averaged | `verify.settle` → `DISPUTED` |
| A passage describing another paper is a pointer, not evidence | model judgement plus a citation-marker heuristic that can only demote |
| Retrieved text addressing an AI reader is excluded | an English pattern list: catches careless cases, not a determined adversary |
| Retracted sources support nothing | OpenAlex flag at fetch, Crossref/Retraction Watch at answer time |
| The bibliography is printed from stored records | `render.py`; the model never types an author, year or DOI |

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) with a text model and an embedding model. Defaults:
  `qwen3:4b-instruct-2507` (about 2.5 GB) and `bge-m3` (about 1.2 GB). Sized for a machine with
  16 GB of memory running one model at a time.
- For online runs, a paper library over MCP:
  [paper-fetch](https://github.com/polarizetech/paper-fetch), which searches
  open-access providers and stores legal full text with provenance.

## Install

```bash
git clone https://github.com/polarizetech/retrieval-augmented-generation.git
cd retrieval-augmented-generation
uv sync                       # add --extra rerank for the ONNX cross-encoder (no torch)
ollama pull qwen3:4b-instruct-2507
ollama pull bge-m3
cp config/rag.env.example config/rag.env
cp config/mcp-gateway.example.json config/mcp-gateway.json
```

Install paper-fetch so that `paper-fetch-mcp` is on your `PATH`, or put its absolute path in
`config/mcp-gateway.json`.

## Use

```bash
uv run research-pipeline index        # index the full texts the paper library holds
uv run research-pipeline ask "Does exercise training lower resting blood pressure in adults?"
uv run research-pipeline ask --offline "..."   # indexed corpus only: no search, no downloads
uv run research-pipeline status
```

Each run writes `runs/<timestamp>-<slug>/answer.md` and `run.json`. An online run fetches
open-access papers into the paper library; `PIPELINE_MAX_FETCH` bounds how many. With the
`[rerank]` extra, the first run downloads a 571 MB int8 cross-encoder.

Set `PIPELINE_VERIFIER_MODEL` to a model from a different family than the text model. Without it
the text model checks its own claims, and every answer says so.

### Novelty probe: has someone already done this?

```bash
uv run research-pipeline novelty --id CAND-0007 "The candidate claim, written out in full" \
    --established "the closest established terminology" \
    --queries "phrasing A" "phrasing B" "phrasing C" "phrasing D" \
    --research-dir ../research        # or set RESEARCH_REPO
```

An adversarial prior-art search. Its null hypothesis is that the claim is already published, and
it tries to prove that:

1. It searches every index the paper library federates, once per phrasing. At least five
   phrasings are required, one of them in the field's established terms.
2. It fetches the nearest open-access papers in full.
3. It puts each retrieved passage to the verifier with the candidate as the claim.

A search hit is not prior art; a passage that states the claim is. There are four verdicts:

| Verdict | Meaning |
|---|---|
| `PRIOR_ART` | Every verifier accepts a passage, its quote is found in the stored text, and every number in the claim is present in it. |
| `PARTLY_KNOWN` | A passage states a weaker or narrower version, or the verifiers disagree. |
| `INCONCLUSIVE` | An index refused, the run was offline, or fewer than three of the nearest works could be read. |
| `CANDIDATE` | None of the above. It describes only the works read, not the literature. |

The dossier (`prior-art.md`, plus the full run log under `runs/`) is written into the **research
repository** at `novelty/<ID>/`, next to the corpus it concerns. The command refuses any directory
that does not carry that repository's marker files. The dossier never announces a discovery. It
does quote the statement as given, and reports what the earlier any-hit rule would have said,
alongside what the reading found.

### As an MCP server

For Claude Desktop or Claude Code, run the gateway over stdio (no port, no token):

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

It exposes `rag__search`, `rag__retrieve_evidence`, `rag__check_citations`, `rag__save_report`,
the paper library as `papers__*`, and the whole pipeline as `pipeline__*`. Through MCP the pipeline
uses the calling model (Claude, ChatGPT) for its model steps by default: code runs every stage and
hands the client batches of tasks with a JSON schema to answer, and Ollama is only needed for
embeddings. See [MCP clients](docs/mcp-clients.md) for the workflow and the HTTP transport.

## Domains

The engine is domain-neutral. What a field knows (which study designs count and in what order,
which databases are authoritative, what a unit means, what may not be generalised) lives in a
separately versioned package under `domains/`, discovered through an entry point. The engine
never imports one; a policy is data and is written into `run.json`.

```bash
uv pip install -e domains/cardiovascular
uv run research-pipeline ask --domain cardiovascular "Does higher baroreflex sensitivity predict lower cardiovascular mortality?"
```

| Domain | What it adds |
|---|---|
| `cardiovascular` | Cardiac, vascular and autonomic evidence; a surrogate is not an outcome. |
| `respiratory` | Breathing, gas exchange and the chemoreflex; lung function is not a clinical endpoint. |
| `vestibular` | Balance and dizziness, graded by diagnostic accuracy rather than by trial. |
| `neuroscience` | The nervous system, with the species gap and reverse inference as failure modes. |

Writing your own: [domains](docs/DOMAINS.md).

## Evaluation

```bash
uv run python benchmarks/research/verifier_bench.py   # does the verifier reject what it should?
uv run python benchmarks/research/eval_pipeline.py    # end-to-end cases, scored from run logs
```

Both are scored without an LLM judge. The seed sets are small (7 verifier pairs, 4 questions):
they are smoke tests and templates, not accuracy estimates. See [benchmarking](docs/benchmarking.md).

## Development

```bash
uv sync --group dev
uv run pytest --cov          # offline: no Ollama, no network
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

CI runs the same checks on Python 3.11 to 3.13.

## Security

Ollama and the gateway's HTTP transport have no meaningful authentication. Keep them on loopback
(`scripts/mcp-start.sh` refuses any other bind unless `MCP_ALLOW_PUBLIC_BIND=1` is set) and put an
authenticating proxy in front if remote access is needed. Retrieved papers are untrusted input: the
pipeline wraps them as data and holds no write tool except the library's own `fetch`.

## License

[MIT](LICENSE)
