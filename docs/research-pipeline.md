# Design and rationale

This document explains why the pipeline is built the way it is, which rule each design decision
enforces, where in the code it is enforced, and which test checks it. It also states what the
design does not achieve. The references at the end were checked for existence, authors and venue
in September 2026; figures quoted from them are as their abstracts or documentation report.

## The problem

Retrieval-augmented generation gives a model more text. It does not make the model's answer
accountable to that text. With a small local model (about 4B parameters, one model resident in
16 GB of memory) the usual failure modes are sharper:

1. **Small models cannot be the agent.** On the Berkeley Function Calling Leaderboard (V4, data of
   2026-04-12) `Qwen3-4B-Instruct-2507` scores about 88% on single, well-formed calls, 22% on
   multi-turn sequences and 3% on web-search tasks [22]. It is a reliable component and an
   unreliable orchestrator.
2. **Self-verification does not verify.** Intrinsic self-correction does not improve reasoning
   without external feedback [4, 5]; model judges favour their own output [6]; small judges are
   lenient [7].
3. **References get fabricated, and real ones get misused.** Unaided LLMs fabricate a
   substantial share of bibliographic citations [13], and even retrieval-backed answers often
   cite sources that do not support the statement they are attached to [14].
4. **Citations get laundered.** A substantial share of quotations in the medical literature do not
   accurately reflect the cited source [15], and claims acquire authority through citation alone
   [16]. A passage describing another paper's result is a pointer, not a source.
5. **Retraction flags are unreliable.** OpenAlex's `is_retracted` has a documented error record
   [17]. Crossref has carried the Retraction Watch database, in `updated-by`, since January 2025 [23].
6. **Retrieved papers are untrusted input.** Documents can carry instructions aimed at the model
   that reads them [18], and preprints have been found carrying hidden instructions aimed at AI
   reviewers.
7. **Parsing noise cascades.** Errors in PDF extraction propagate through retrieval into
   generation [19], and no parser handles scientific PDFs well [20]. Structured full text (JATS
   XML) is preferred where the paper library can obtain it.
8. **LLM rerankers are slow and biased.** They carry a recency bias and flip a sizeable share of
   equal-relevance pairs [12]. A small cross-encoder is faster and has no view of dates or venues.
9. **Guessing is rewarded.** Accuracy-only evaluation rewards a model for guessing over
   abstaining [3]. Here "insufficient evidence" is a normal, first-class output.

## Architecture

Code orchestrates; the model fills in forms. Every model call is one turn, decoded under a JSON
schema through Ollama's native API with the context size set explicitly. No call can choose a tool.

The stages marked *library* are the paper library's
([paper-fetch](https://github.com/polarizetech/paper-fetch), over MCP), which owns everything
about finding and indexing papers: provider search, discipline profiles, search memory,
collections, fetching and the passage index. Nothing it does uses a language model.

```text
question
  |  plan         model: mode, core question, <=4 sub-questions, queries, falsification queries,
  |               given the field's indexed terms from the library's discipline profile
  v
discover         library `search` for every query, with the field's profile: a synonym is also
  |               searched as its indexed term; a provider that did not answer is logged; every
  |               search is remembered, and one on a concept searched before says so
  |  normalise    merge by DOI/PMID/PMCID; fold a preprint and its published version into one source
  v
acquire          library `relevance` ranks candidates against the sub-questions; `fetch` the top N
  |               (open access only)
  |  index        library: paragraph chunks with character offsets; reference list excluded;
  |               hidden characters stripped and counted; dense vectors when it has an embedder
  v
retrieve         library: SQLite FTS5 (BM25) + exact dense search, fused by reciprocal rank
  |               (k = 60) [11], optionally re-scored by its cross-encoder
  |  filter       drop retracted works and passages flagged by the safety scan
  v
select           the library's order is kept (no model here re-scores passages); <=2 passages
  |               per paper, <=6 per sub-question, because long contexts are used unevenly [9]
  v
extract          model per passage: relevant? direction, finding, verbatim quote, population
  |  anchor       the quote must occur in the stored passage, or the evidence is dropped
  |  second-hand  a citation marker in the quote demotes it to a pointer (role O, not E)
  v
synthesise       model per sub-question; may cite only evidence ids from this run
  v
critique         computed problems + model proposes NEW SEARCHES; it never re-rates the draft
  |               -> discover .. synthesise again, up to PIPELINE_MAX_ROUNDS
  v
verify           number check (code), then verifier model(s) that see only claim and passage
  |  integrity    Crossref / Retraction Watch for every DOI that supports a claim
  |  grade        label computed from independent first-hand sources, conflict and preprint status
  v
render           assembled in code from verified records; bibliography from stored metadata
  v
runs/<timestamp>-<slug>/answer.md and run.json
```

`run.json` records the domain policy, models by digest, prompts by hash, every query and each
provider's status, every candidate and what happened to it, every evidence record with the full
passage text it came from, every claim with every verifier check, the retraction lookups, and the
answer with its SHA-256.

### Companion tools

Some questions are also served by something papers cannot give, such as a list of open datasets
that could test the claim. A *companion* (`research_pipeline/companions.py`) is a tool that
provides that, used on the same terms as every other step:

1. **The model decides whether, and what to ask.** With a companion configured, the plan's schema
   gains one field, `tools`: a list of `{tool, queries, why}` restricted to the configured keys.
   The planner is told what each is for and that most questions need none. Without a configured
   companion the plan prompt and schema are exactly as before.
2. **Code calls it.** At most once per tool per run, with the planner's queries and the bounds
   the registry sets. A companion that fails is recorded as not having answered; it never ends a
   run. Offline runs call none.
3. **Code prints it.** The section comes from the tool's own records. No model writes it, and it
   never enters the evidence table: it is a pointer for the reader, not support for a claim.

The `datasets` companion speaks dataset-fetch's MCP contract: it asks which catalogues take a
free-text query, searches each (Zenodo restricted to records typed as datasets), reads the data
cards, drops records that hold only documents (counted, not listed), and orders the rest by the
paper library's `relevance` to the question. A catalogue that did not answer is named in the
answer's limits. Measured: two queries over three catalogues, ten cards read, 14 s.

To add a companion, write `run` (call the tool, return plain records) and `render` (print them)
and register a `Companion`. Nothing in the pipeline changes.

## Rules, where they are enforced, and how they are tested

| Rule | Enforced in | Tested by |
|---|---|---|
| The model cannot cite evidence that was not retrieved | evidence ids are an `enum` in the synthesis schema, then checked in `Pipeline.synthesise` because a decoding constraint is the runtime's promise, not this code's | `test_pipeline.py::test_an_unretrieved_evidence_id_never_becomes_a_claim` |
| A quote must occur in the stored passage | `verify.anchor_quote`. An exact match (ignoring whitespace and case) is accepted. A near-identical paraphrase (similarity >= 0.85) is replaced by the source's own sentence, unless the two differ in negation or in any number. Quotes under five words are rejected. | `test_verify.py::TestQuoteAnchoring` |
| A number in a claim must occur in its cited evidence | `verify.unsupported_numbers`. Spellings are canonicalised (`.05` = `0.05`, `10-Hz` = `10 Hz`). Where the domain declares units, the number must appear with a unit of the same measure. A failing claim gets verdict `NUMBER_NOT_IN_SOURCE` and is removed, whatever the verifiers said. | `test_verify.py::TestNumbers`, `test_pipeline.py::test_a_wrong_number_removes_the_claim_even_when_the_verifier_accepts_it` |
| Existence and support are separate checks | existence is structural (evidence only from library records); support is quote, number and verifier checks | the two rows above |
| Verifier disagreement is reported, not averaged | `verify.settle`: SUPPORTED needs every verifier to accept the same passage; a split is `DISPUTED` and is shown as such | `test_verify.py::TestSettling` |
| Failed claims are removed and counted | `render.py`, "Limits of this answer" | `test_pipeline.py` (wrong number, retracted source) |
| A passage describing another paper is a pointer | the model's `secondhand` judgement, overridden (never relaxed) by `verify.cites_other_work`; only role `E` counts toward a label or toward "sources disagree" | `test_verify.py::TestCitationMarkers`, `test_grading.py::test_second_hand_reports_are_weak` |
| Disagreement is computed, not asserted | each passage has a direction relative to the question; "sources disagree" needs first-hand results from different papers on each side | `test_pipeline.py::test_the_null_result_is_reported_as_opposing_evidence`, `test_surfaces.py::TestEvalScoring` |
| Falsification searches always run | a falsification sub-question is added to every plan and searched with its own queries | `test_pipeline.py::test_critique_produces_new_searches` |
| The critic searches; it does not rewrite | `Pipeline.critique` returns queries, which go back through discovery | `test_pipeline.py::test_critique_produces_new_searches` |
| A silent provider is not zero results | `Pipeline.discover` logs `did_not_answer`; the answer lists those providers | `test_pipeline.py::test_silent_providers_are_recorded_not_read_as_empty` |
| "No opposing result" is a statement about the search | printed with the searches that were run | rendered in every answer without opposing evidence |
| Retracted sources support nothing | OpenAlex flag at fetch and retrieval; Crossref at answer time sets `RETRACTED_SOURCE`. A failed lookup is "unchecked", and a clean lookup is "no notice found", never "clean". | `test_pipeline.py::test_a_retracted_source_supports_nothing`, `test_integrity.py` |
| Retrieved text is data | passages are wrapped and delimited; `safety.scan` drops passages addressed to an AI reader, including the text used to classify a paper; hidden characters are counted on the raw text before they are stripped | `test_grading.py::TestSafety`, `test_rag.py::test_injected_and_retracted_passages_are_excluded`, `test_pipeline.py::test_hidden_characters_are_noted_before_they_are_stripped` |
| A run cannot promote its own output into the corpus | the pipeline calls only the library's read tools and `fetch`; nothing it writes is indexed | structural: `papers.py` has no other write |
| The bibliography is never typed by the model | `render.py` prints authors, year, title and DOI from stored records | `test_pipeline.py::test_a_correct_claim_is_kept_with_its_quote` |
| Self-verification is disclosed | with no separate verifier model configured, every answer says so | `test_pipeline.py::test_self_verification_is_disclosed` |
| A run is reproducible from its log | see the `run.json` contents above | `test_pipeline.py::test_the_run_log_holds_passages_models_and_the_answer` |

Labels (strong, moderate, weak, insufficient) are computed from the evidence set by
`grading.label` and describe what was retrieved and machine-checked in one run. They are not
evidence grades in the GRADE sense [21], which rates a body of evidence per outcome on risk of
bias, inconsistency, indirectness, imprecision and publication bias. Nobody has read these papers.

## The client-mediated path (`rag__*`)

When a stronger model writes the answer, the `rag__*` MCP tools apply the deterministic subset of
the rules above:

- `rag__retrieve_evidence` returns verbatim passages under ids of the form `<work>#p<n>.<hash>`.
  The position survives re-indexing; the hash of the passage text makes an id stale if the text
  changes. SQLite reuses freed row ids, so a row id alone could silently name a different passage
  (`test_rag.py::test_row_ids_alone_would_not_be_safe_evidence_ids`). Flagged and retracted
  passages are excluded and listed. Retrieval is hybrid when the embedding model answers and
  reports when it fell back to lexical search.
- `rag__check_citations` accepts a claim only if every id resolves, at least one quote occurs in its
  cited passage, no quote fails, and every number in the claim occurs in the cited passages.
- `rag__save_report` stores the report with the exact passages it cites, and records whether its
  claims were checked.

**What this path cannot check is entailment**: whether the passage, read in full, supports the
claim. That needs a verifier model, which is what the pipeline adds. Tool descriptions and the
gateway's instructions say so, so that a client model can relay it.

## Models

Behind the MCP server the calling model fills in the forms by default
([MCP clients](mcp-clients.md#the-full-pipeline-with-the-client-as-its-model)): every stage, prompt,
schema and check is the same; only who answers changes. The command line uses a local model:

| Role | Default | Note |
|---|---|---|
| Text | `qwen3:4b-instruct-2507` | An 8B model would be a real upgrade in multi-turn reliability [22]; the pipeline does not need it, because it never asks the model to sequence calls. |
| Embedding | `bge-m3`, the paper library's (`PAPER_FETCH_EMBED_MODEL`) | Dense only; BM25 comes from SQLite FTS5, because Ollama exposes no sparse output. |
| Reranker | the library's cross-encoder (`PAPER_FETCH_RERANK_MODEL`, e.g. int8 `bge-reranker-v2-m3`, 571 MB) | Scores a passage in a fraction of a second, sees passage text only, and writes nothing. A chat model no longer reranks: the local 4B model spent 4.9 of a 5.7-hour run doing it. Without the setting, passages are read in retrieval order and the answer's log says so. |
| Verifier | the text model, **with a printed caveat** | The right second verifier is a trained grounding classifier from another family, such as Bespoke-MiniCheck [8] (7B, CC BY-NC). `verify.judge` supports its documented `Document:/Claim:` format. |

Set the context size explicitly (`PIPELINE_NUM_CTX`). Ollama's default can be about 4K tokens and
overflow is truncated from the start of the prompt, which removes the system prompt first.

Where a local run spends its time (Qwen3-4B q4_K_M, Ollama 0.34, M2 Pro 16 GB, six runs of
September 2026): 9 to 13 minutes per question, of which 50-70% is model time. Generation runs at
about 50 tokens/s with every layer on the GPU. With a JSON schema, Ollama lets this model write
several hundred tokens of hidden reasoning before the constrained answer, and ignores
`think: false`: a verification call that returns 272 characters generates about 500 tokens (about
10 s). Suppressing that reasoning made verification four times faster but got 5 of the 7 verifier
cases right instead of 7, so it stays on. Models are kept loaded for 30 minutes
(`PIPELINE_OLLAMA_KEEP_ALIVE`) so that a run's network-bound stages do not unload them.

### Planned: your own endpoint for the model steps

The calling model is the default behind MCP (`PIPELINE_MCP_LLM=client`), and stays so. The other
backend today is Ollama on the same machine, which exists for one reason: keeping a question and
its intermediate work off a hosted assistant. That backend should become "any OpenAI-compatible
endpoint" (one setting each for URL, key and model), so the same option covers local Ollama, a
hosted open-weight model with zero data retention (OpenRouter with ZDR-only routing, DigitalOcean
serverless inference), or a self-hosted GPU server. Sizing: a run without reranking is about
42,000 input and 10,000 output tokens, which is cents on hosted open-weight models. Note that a
run started from a chat client still shows that client the question and the final answer; a
fully private run starts from the command line. Not built yet; nothing else needs to change for
it, since every model step already goes through one `ChatModel` interface.

## Alternatives considered (as of September 2026)

- **An agent framework (PaperQA2 and similar).** These depend on the model sequencing tool calls,
  which is exactly what a 4B model does badly (problem 1). PaperQA2's own documentation warns
  against 7B models. Its per-passage contextual extraction with a relevance judgement is a good
  idea at this scale, and `extract` reimplements it.
- **A tool aggregator (ToolUniverse and similar).** Useful for breadth, but a compact tool mode
  still requires multi-hop tool use, and the literature providers it wraps are covered by the
  paper library with open-access rules and provenance.
- **Prompt optimisation (DSPy).** Deferred: optimisers such as GEPA and MIPROv2 need a reflection
  model stronger than the one being optimised.
- **OpenScholar** [1] and **SciRAG** [2] are the strongest published systems for this task. They
  are reference points; OpenScholar's reranker is a candidate for comparison.

## Known limits

- **Text only.** Results that live in figures, tables or supplements are not seen.
- **Open access only.** Relevant closed papers are listed in each answer as unread.
- **Entailment is a model's judgement**, and by default the same model's. Deterministic checks
  catch wrong quotes and wrong numbers, not a claim that overstates a correctly quoted passage.
- **A client model is trusted to follow its instructions.** Behind MCP, answers are checked for
  shape (JSON schema), quotes and numbers, but nothing can confirm that the client used only the
  text it was given, or tell which model it runs. Its claims still pass the same verification.
- **Study type is a small model's classification** of a paper's opening text, cached per paper.
- **The second-hand and injection checks are heuristics.** The citation-marker pattern misses some
  styles and can be fooled by unusual gene names; the injection scan is an English pattern list.
- **No section labels.** The library returns plain text, so "introduction versus results" is
  inferred from citation markers rather than read from JATS structure.
- **Preprint and published versions are folded by title.** A retitled pair is counted twice;
  Crossref's `is-preprint-of` relation would catch it.
- **Determinism is limited.** Model calls use temperature 0 and a fixed seed, which makes a run
  repeatable on one machine and runtime version, not across hardware.
- **The evaluation sets are seeds** (7 verifier pairs, 4 end-to-end questions). With 10 cases a
  proportion's 95% interval is about ±30 points. Larger external sets: LitSearch [10] for
  retrieval, SciFact and LLM-AggreFact for verification, LitQA2 for end-to-end answers with
  abstention. No benchmark results are committed yet.

## Domains

The engine is domain-neutral. What a field knows lives in a separately versioned domain package
that declares a `DomainPolicy` and is found through the `rag.domains` entry point. The
policy is data, not behaviour: it serialises into `run.json`, and its prompt fragments are appended
to the engine's, never substituted, so a domain can tighten a rule but cannot remove one. See
[domains](DOMAINS.md).

## References

Peer-reviewed unless marked.

1. Asai A, et al. Synthesizing scientific literature with retrieval-augmented language models. *Nature* 650:857-863 (2026). doi:10.1038/s41586-025-10072-4
2. Ding H, et al. SciRAG. EACL 2026, pp 6440-6460. https://aclanthology.org/2026.eacl-long.303/
3. Kalai AT, Nachum O, Vempala SS, Zhang E. Evaluating large language models for accuracy incentivizes hallucinations. *Nature* 653:1047-1051 (2026). doi:10.1038/s41586-026-10549-w
4. Huang J, et al. Large Language Models Cannot Self-Correct Reasoning Yet. ICLR 2024.
5. Kamoi R, et al. When Can LLMs Actually Correct Their Own Mistakes? *TACL* 2024. https://aclanthology.org/2024.tacl-1.78/
6. Panickssery A, Bowman S, Feng S. LLM Evaluators Recognize and Favor Their Own Generations. NeurIPS 2024.
7. Thakur AS, et al. Judging the Judges. GEM 2025. https://aclanthology.org/2025.gem-1.33/
8. Tang L, Laban P, Durrett G. MiniCheck. EMNLP 2024. https://aclanthology.org/2024.emnlp-main.499/
9. Liu NF, et al. Lost in the Middle. *TACL* 2024. https://aclanthology.org/2024.tacl-1.9/
10. Ajith A, et al. LitSearch. EMNLP 2024. https://aclanthology.org/2024.emnlp-main.840/
11. Cormack GV, Clarke CLA, Buettcher S. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. SIGIR 2009.
12. Fang H, et al. Recency bias in LLM rerankers. SIGIR-AP 2025. arXiv:2509.11353
13. Walters WH, Wilder EI. Fabrication and errors in the bibliographic citations generated by ChatGPT. *Sci Rep* 13:14045 (2023).
14. Wu K, et al. SourceCheckup. *Nat Commun* 16:3615 (2025).
15. Jergas H, Baethge C. Quotation accuracy in medical journal articles: a systematic review and meta-analysis. *PeerJ* 3:e1364 (2015).
16. Greenberg SA. How citation distortions create unfounded authority. *BMJ* 339:b2680 (2009).
17. Hauschke C, Nazarovets S. (Non-)retracted academic papers in OpenAlex. *J Inf Sci* (2025). arXiv:2403.13339
18. Greshake K, et al. Not what you've signed up for: indirect prompt injection. AISec@CCS 2023.
19. Zhang J, et al. OCR Hinders RAG (OHRBench). ICCV 2025.
20. Meuschke N, et al. A benchmark of PDF information extraction tools. iConference 2023.
21. Guyatt GH, et al. GRADE: an emerging consensus on rating quality of evidence and strength of recommendations. *BMJ* 336:924 (2008).
22. Berkeley Function Calling Leaderboard V4 (data of 2026-04-12). https://gorilla.cs.berkeley.edu/leaderboard.html (not peer-reviewed)
23. Crossref. Retraction Watch data in the REST API. https://www.crossref.org/documentation/retrieve-metadata/retraction-watch/ (documentation)
