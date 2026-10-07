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
  paper library's passage index, quotes that must occur in the cited passage, and numbers that must occur in the
  cited evidence. These tools do not judge entailment; the pipeline's verifier models do.
- **The quantity tools** (`math__*` over MCP) give a client model constants and equations to cite
  instead of recall: `math__lookup` searches the research corpus's calculator records
  (`projects/*/calculators/*.md`, found through `$RESEARCH_CORPUS` or a sibling `research/` checkout)
  and returns their sections verbatim with the corpus commit; `math__bionumber` returns a BioNumbers
  entry by BNID, verbatim, fetched once and cached. No model is involved in either.

Finding and indexing papers is not done here. That is the paper library,
[paper-fetch](https://github.com/polarizetech/paper-fetch), reached over MCP: it searches the
open-access providers, knows how each scientific discipline is indexed (a *profile*: its MeSH
terms and synonyms), remembers past searches, keeps project collections, stores legal full text,
and owns the passage index that retrieval reads. This repository tells it the field and the
concepts, then does everything that happens after a passage is found: reading it, judging it,
and checking every claim made from it.

| paper-fetch (search and indexing) | this repository (evidence) |
|---|---|
| provider search, discipline profiles, search memory, collections | question planning (with the profile's vocabulary) |
| open-access fetching, storage, provenance | extraction, study-design classification |
| passage index: chunking, BM25, embeddings, optional cross-encoder | synthesis, verification, grading, critique |
| fetch ordering by relevance to sub-questions | quote, number and retraction checks; the answer and run log |

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
- The paper library, [paper-fetch](https://github.com/polarizetech/paper-fetch), over MCP.
  Install it with its `mcp` and `retrieval` extras, and give it an embedding model for hybrid
  retrieval (`PAPER_FETCH_EMBED_MODEL=bge-m3`, about 1.2 GB in Ollama); without one, retrieval is
  lexical.
- For local-model runs, [Ollama](https://ollama.com) with a text model (default
  `qwen3:4b-instruct-2507`, about 2.5 GB). Sized for a machine with 16 GB of memory running one
  model at a time. Behind MCP the calling model does this work instead.

## Install

```bash
git clone https://github.com/polarizetech/retrieval-augmented-generation.git
cd retrieval-augmented-generation
uv sync
ollama pull qwen3:4b-instruct-2507
cp config/rag.env.example config/rag.env
cp config/mcp-gateway.example.json config/mcp-gateway.json

uv tool install "paper-fetch[mcp,retrieval] @ git+https://github.com/polarizetech/paper-fetch"
ollama pull bge-m3
echo "PAPER_FETCH_EMBED_MODEL=bge-m3" >> ~/.config/paper-fetch/retrieval.env
```

`paper-fetch-mcp` must be on your `PATH`, or its absolute path in `config/mcp-gateway.json`.
Reranking is paper-fetch's cross-encoder: install `paper-fetch[rerank]` and set
`PAPER_FETCH_RERANK_MODEL` (see its README); without it passages are read in retrieval order.

## Use

```bash
uv run research-pipeline index        # have the library index the full texts it holds
uv run research-pipeline ask "Does exercise training lower resting blood pressure in adults?"
uv run research-pipeline ask --domain cardiovascular "..."   # the field's rules and vocabulary
uv run research-pipeline ask --collection my-review "..."    # search, fetch and read in a collection
uv run research-pipeline ask --offline "..."   # indexed corpus only: no search, no downloads
uv run research-pipeline status
```

Each run writes `runs/<timestamp>-<slug>/answer.md` and `run.json`. An online run fetches
open-access papers into the paper library; `PIPELINE_MAX_FETCH` bounds how many. Every search is
remembered by the library, so a later run on the same concept is told what earlier ones found.

Set `PIPELINE_VERIFIER_MODEL` to a model from a different family than the text model. Without it
the text model checks its own claims, and every answer says so.

### Companion tools: datasets and patents

A run can loop in another tool when the question calls for it. The planner is told what each
configured companion is for and may request it, with queries, in the plan it already writes; code
then calls the tool and prints what its own records say. No model writes that section, and it
never counts as evidence for a claim.

Two companions ship. Both are off until enabled under `companions` in
`config/mcp-gateway.json`, and each is asked only when the planner judges that the question calls
for it; most questions call for neither.

- `datasets` asks [dataset-fetch](https://github.com/polarizetech/dataset-fetch)'s `recommend` (it
  needs a dataset-fetch with that tool) for datasets that could test the question, and adds a
  "Datasets that could test this" section: title, catalogue, licence, files, subjects, DOI, a
  pinned reference and why it was listed. Nothing is downloaded or assessed.
- `patents` runs [patent-fetch](https://github.com/polarizetech/patent-fetch)'s
  `patent-fetch --json search` when the question is about a device, a method or another
  invention, and adds an "Existing patents on this" section: title, publication number, year,
  applicants or inventors, classes and a link, ordered by how close each title sits to the
  question. Nothing is fetched or read, and it is not a legal opinion: the section says so, and
  the limits name every patent service that was not asked or did not answer. Without a patent
  service account, patent-fetch can only search life-science patents up to 2012; the answer says
  that too.

Adding another tool is one entry in `research_pipeline/companions.py`: an MCP server or a command
line that prints JSON.

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
