# Benchmarking

Benchmarks here answer one question: does a change to the pipeline, a model or a prompt make the
answers more or less trustworthy? They are scored from run logs by code, never by an LLM judge.

## The two suites

| Suite | What it measures | Cases |
|---|---|---|
| `benchmarks/research/verifier_bench.py` | whether a verifier model rejects claims it should: false accepts (unsupported claims passed as SUPPORTED) matter most, because those reach the reader | `verifier_cases.jsonl`: 7 synthetic passage/claim pairs, one per failure mode |
| `benchmarks/research/eval_pipeline.py` | end to end: source recall, claims kept, whether a known conflict is shown, whether an unanswerable question is abstained on, forbidden phrasing | `pipeline_cases.jsonl`: 4 questions with expected open-access sources |

The verifier cases use synthetic passages written for the benchmark, so that no copyrighted text
is redistributed and the expected label is unambiguous. The pipeline cases name real open-access
papers; index them before an offline run.

## Reading the numbers

The seed sets are too small to estimate a rate: with n = 10, a proportion's 95% interval is about
±30 points. Use them as smoke tests and as templates. Before comparing two versions, grow each
slice to about 50 cases; unanswerable cases can be made by removing the gold paper from the index.
Larger external sets: LitSearch (retrieval), SciFact and LLM-AggreFact (verification), LitQA2
(end-to-end answers with abstention).

## Recording a result

Raw results go to `benchmarks/results/` (gitignored). A result worth committing records:

- the date, hardware, OS and memory pressure (a swapping run is not comparable to one that is not);
- the Ollama version and each model's digest, quantisation and context size (all in `run.json`);
- the prompt version (`prompt_version` in `run.json`) and the case-file revision;
- the number of runs and whether the model was already loaded.

Model calls use temperature 0 and a fixed seed. That makes a run repeatable on one machine and
runtime version, not across hardware, so compare results collected on the same setup.

## Adding cases

One case tests one observable behaviour. Prefer checks that code can score: an expected DOI among
the evidence, a known conflict, abstention, or a `must_not_claim` pattern such as causal language
for an observational association. Every `expect_dois` entry must be a real, open-access paper,
checked against Crossref before it is added.
