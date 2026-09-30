# Writing a domain extension

The engine in `research_pipeline/` knows how to plan a search, retrieve passages, extract
findings, synthesise claims, verify them against their sources and grade what survives. It does
not know what a study design is in your field, which databases are authoritative for it, what a
number with a unit means, or when a result may not be generalised.

That knowledge is a **domain**: a separately versioned package that declares a `DomainPolicy` and
is found through an entry point. The engine never imports it. The pattern is the usual one for
plugin ecosystems: a distribution-name prefix, import-time declaration, a version gate,
entry-point discovery, and no core-side list of extensions.

## The one rule

**A domain is data, not behaviour.** You describe your field; the engine applies its own rules
using your description. It never calls back into your package to make a decision.

This is not stylistic. A `DomainPolicy` serialises, so it goes into `run.json` and a run stays
reproducible from its log. Behaviour would not serialise, and a run whose grading depended on
code that has since changed could not be audited. It also keeps the dependency arrow pointing one
way, which is what lets a domain be installed, removed or versioned without touching the engine.

The second consequence is worth stating plainly: **domain prompt fragments are appended, never
substituted.** A domain can tighten an engine rule. It cannot talk the engine out of one.

## Anatomy

```
domains/<slug>/
├── pyproject.toml            # distribution + the entry point that makes it discoverable
└── rag_<slug>/
    └── __init__.py           # the policy, and one declare() call at the bottom
```

Copy `docs/domain-template/` and rename. Then add a row to `domains/catalog.json`, which is an
informational index — nothing loads it at runtime, but `tests/test_domains.py` checks it against
the directories, so a domain cannot be added or renamed without the index noticing.

## The policy, piece by piece

### `Taxonomy` — what counts as a study here

The study designs your field produces, ranked the way your field ranks them. This is the piece
most likely to differ between domains and the reason the taxonomy is not in the engine.

`primary=True` marks a design that reports its own first-hand observations; only primary evidence
can lift a claim above "weak", because a review restating a result is a pointer to the study, not
a second study. `rank` is what a GRADE-style rubric sorts on. An `unclear` design is mandatory —
the classifier needs somewhere to put a paper whose text does not say.

Compare `vestibular` with `cardiovascular`. Vestibular science has comparatively little
randomised-trial literature; its load-bearing designs are diagnostic accuracy against a reference
standard, psychophysics and posturography. Grading it with a trial-shaped rubric would mark most
of its real evidence down, so it ranks a well-conducted accuracy study near the top. Neither
ordering is correct in general. That is the point.

Definitions are read by a small model inside the classifier prompt, so keep each to one clause of
plain language.

### `Anchor` — what your field calls things

Anchors seed query expansion with the term a database actually indexes under, and they are
written into the run log so a run can state which vocabulary it searched.

**A `mesh` anchor must carry a real descriptor id.** The dataclass rejects anything that is not a
`D` followed by six to nine digits, and you should not satisfy it by guessing. An invented id is
worse than no id: it would be logged as provenance for a search that never happened, and it looks
exactly like a real one to anyone auditing the run afterwards.

Plenty of real terms have no descriptor. Declare those as `free-text` with their synonyms. This
is not a lesser anchor, it is an accurate one — and for a term with no descriptor the synonyms
are doing all the retrieval work anyway.

An example of why this matters: "High-Intensity Interval Training" has a MeSH descriptor
(`D000072696`); "exercise snacking" (brief bouts of vigorous activity through the day) does not.
A pipeline that anchors both to MeSH retrieves far more of the first literature than the second,
and then reports that imbalance as though it were a finding about the field rather than an
artefact of indexing. Declaring the second as free-text, with its synonyms, keeps the comparison
fair.

To check a descriptor before you use it:

```bash
curl -s 'https://id.nlm.nih.gov/mesh/lookup/descriptor?label=Hypertension&match=exact&limit=1'
```

### `Measure` — numbers that carry units

The engine checks every number in a claim against its cited passages. That check is blind to
units by default, so a claim's "40 mmHg" counts as supported by a passage that only ever said
"40 ms". The same holds for spelling: "40-mmHg", "40mmHg" and "40 mmHg" are one quantity. Declaring your measures upgrades it: a number with a unit must now be found in the
source with a unit the same measure allows.

Units form de-facto equivalence classes. Two measures that share a unit collapse together, which
is the correct "same unit family" semantic — SDNN and RMSSD are both in `ms` and are both
legitimately checked against a passage reporting `ms`.

### `Source` — where the evidence lives, and why

The engine does not fetch from these; acquisition belongs to the paper library. It records them
in the run log and tells the planner which vocabularies its queries will be matched against,
which is what actually changes the queries a small model writes. Fill in `why` — it is the part a
reader uses to judge whether the search was aimed at the right place.

### `CritiqueRule` — a weakness your field recognises

Stated as a test on study design: it fires when **every** design behind a claim falls inside the
set, and the planner is then asked for a search that would fix it. This is how "animal or
in-vitro evidence only" stopped being an assumption baked into the engine and became something a
field declares.

Note it deliberately does not fire on an empty evidence set. No evidence at all is a different
problem, reported elsewhere; a critique rule claiming it would be misattributing the failure.

### Prompt fragments

`search_guidance`, `reporting_rule`, `generalisation_rule` and `gap_guidance` are appended to the
corresponding engine prompts. Write them as instructions to a small model: concrete, imperative,
and about this field's specific failure modes rather than good practice in general.

### `min_primary_for_strong`

How many independent first-hand primary sources a claim needs before your field calls its support
strong. The default is 2. A field whose studies are typically small, whose effect sizes are
unstable, or whose central constructs are measured by instruments that disagree with each other
should set 3: two concordant studies there mean less than two concordant trials elsewhere.

## Declaring it

```python
DOMAIN = declare(
    name="rag-domain-<slug>",  # the prefix is enforced
    module=__name__,
    version=__version__,
    core_requires=">=0.2.0,<0.3",  # checked on import, not at grading time
    job="One line: the single thing this domain adds that the engine could not know.",
    policy=POLICY,
)
```

`declare()` refuses a name without the prefix, an empty `job`, a version that is not dotted
integers, an engine outside `core_requires`, an `extends` target that is missing or out of range,
two packages claiming one name, and two domains claiming one slug.

### `extends`

Only for a real dependency: a domain that spans others, and imports their `ANCHORS` to build
queries across them. A cardiopulmonary domain, for example, would extend `cardiovascular` and
`respiratory` rather than copy their vocabularies, because copies drift apart silently.
`declare()` checks that every extended domain is installed and in range.

If you are not importing from a domain, do not extend it.

## Installing and checking

```bash
uv pip install -e domains/<slug>
research-pipeline domains          # discovery, with a status row per domain
research-pipeline ask --domain <slug> "..."
```

Set `PIPELINE_DOMAIN=<slug>` in `config/rag.env` to make it the default.

A broken or incompatible domain is **reported with a status row, never silently skipped**. A run
that quietly lost its field's grading rules would still produce an answer, and that answer would
be wrong in a way nobody could see from the output.

With no domain installed or selected, the engine uses the `GENERIC` policy: the engine's own rules,
with no field's vocabulary, units or grading.

## Testing

`tests/test_domains.py` covers the contract and this repo's domains. Two tests are load-bearing:

- `test_core_imports_no_domain` — the engine must never import a domain package. The whole
  architecture rests on this, and it is invisible at runtime until someone adds a convenience
  import and quietly teaches the engine one field's rules.
- `test_declared_metadata_matches_the_catalogue` — parses each `declare()` call from source
  rather than importing it, so the check still runs when the domain packages are not installed.

## Current domains

| slug | what it adds |
|---|---|
| `cardiovascular` | Cardiac, vascular and autonomic evidence; a surrogate is not an outcome. |
| `respiratory` | Breathing, gas exchange and the chemoreflex; lung function is not a clinical endpoint. |
| `vestibular` | Balance and dizziness, graded by diagnostic accuracy rather than by trial. |
| `neuroscience` | The nervous system, with the species gap and reverse inference as first-class failure modes. |
