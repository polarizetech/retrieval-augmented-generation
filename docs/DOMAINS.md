# Writing a domain extension

The engine in `research_pipeline/` knows how to plan a search, retrieve passages, extract
findings, synthesise claims, verify them against their sources and grade what survives. It does
not know what a study design is in your field, what a number with a unit means, or when a result
may not be generalised.

That knowledge is a **domain**, and it has two halves that live in two places:

- **The evidence half**, here: a separately versioned package that declares a `DomainPolicy`
  (study designs and how they rank, measures and their units, critique rules, prompt rules for
  reading and generalising), found through an entry point.
- **The search half**, in the paper library ([paper-fetch](https://github.com/polarizetech/paper-fetch)):
  a *discipline profile*, a small TOML file with the terms the field is indexed under, their
  synonyms, its authoritative sources and advice for writing its queries. The paper library uses
  it deterministically (it also searches each synonym's indexed term, and ranks hits that name the
  field's terms first), and the planner here reads it. A policy names its profile in
  `DomainPolicy.profile`, by default its own slug.

The evidence package is found through an entry point. The engine never imports it. The pattern is the usual one for
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

~/.config/paper-fetch/profiles/<slug>.toml   # the search half, read by the paper library
```

Copy `docs/domain-template/` and rename. Its `profile.toml` is the search half: copy it to
`~/.config/paper-fetch/profiles/<slug>.toml` (or `$PAPER_FETCH_PROFILE_DIR`), or contribute it to
paper-fetch's built-in profiles, where the four domains in this repository have theirs. Then add a row to `domains/catalog.json`, which is an
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

### `Measure` — numbers that carry units

The engine checks every number in a claim against its cited passages. That check is blind to
units by default, so a claim's "40 mmHg" counts as supported by a passage that only ever said
"40 ms". The same holds for spelling: "40-mmHg", "40mmHg" and "40 mmHg" are one quantity. Declaring your measures upgrades it: a number with a unit must now be found in the
source with a unit the same measure allows.

Units form de-facto equivalence classes. Two measures that share a unit collapse together, which
is the correct "same unit family" semantic — SDNN and RMSSD are both in `ms` and are both
legitimately checked against a passage reporting `ms`.

### `CritiqueRule` — a weakness your field recognises

Stated as a test on study design: it fires when **every** design behind a claim falls inside the
set, and the planner is then asked for a search that would fix it. This is how "animal or
in-vitro evidence only" stopped being an assumption baked into the engine and became something a
field declares.

Note it deliberately does not fire on an empty evidence set. No evidence at all is a different
problem, reported elsewhere; a critique rule claiming it would be misattributing the failure.

### Prompt fragments

`reporting_rule`, `generalisation_rule` and `gap_guidance` are appended to the corresponding
engine prompts; the planner's search advice comes from the profile's `search_guidance`. Write them as instructions to a small model: concrete, imperative,
and about this field's specific failure modes rather than good practice in general.

### `min_primary_for_strong`

How many independent first-hand primary sources a claim needs before your field calls its support
strong. The default is 2. A field whose studies are typically small, whose effect sizes are
unstable, or whose central constructs are measured by instruments that disagree with each other
should set 3: two concordant studies there mean less than two concordant trials elsewhere.

## The search half: a discipline profile in the paper library

A profile is data the paper library reads; nothing in it runs. The fields:

```toml
slug = "my-domain"            # the same slug as the policy (or set DomainPolicy.profile)
label = "My domain"
scope = "One sentence."
search_guidance = """How to write a query that this field's databases will actually match."""
providers = []                # optional: search providers to use instead of the default set

[[anchors]]                   # a term the field is indexed under
system = "mesh"
label = "Hypertension"
id = "D006973"
synonyms = ["high blood pressure"]

[[measures]]                  # quantities the planner may name in queries
label = "Systolic blood pressure"
units = ["mmHg"]

[[sources]]                   # recorded, never fetched from
key = "pubmed"
label = "PubMed / MEDLINE"
why = "Why this source matters for your field."
```

A query that uses a synonym is also run with its anchor's label, never the reverse: the synonym
is how people write, the label is what the database indexes. So put the indexed term in `label`
and what people type in `synonyms`.

**A `mesh` anchor must carry a real descriptor id.** paper-fetch rejects anything that is not a
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

Only for a real dependency: a domain that spans others, and imports their `MEASURES` or
taxonomy. A cardiopulmonary domain, for example, would extend `cardiovascular` and `respiratory`
rather than copy their units and designs, because copies drift apart silently.
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
with no field's vocabulary, units or grading. If a policy names a profile the paper library does
not have, the run proceeds without the field's vocabulary and its notes say so.

## Testing

`tests/test_domains.py` covers the contract and this repo's domains. Two tests are load-bearing:

- `test_core_imports_no_domain` — the engine must never import a domain package. The whole
  architecture rests on this, and it is invisible at runtime until someone adds a convenience
  import and quietly teaches the engine one field's rules.
- `test_declared_metadata_matches_the_catalogue` — parses each `declare()` call from source
  rather than importing it, so the check still runs when the domain packages are not installed.
- `test_every_domain_has_a_profile_in_the_paper_library` — each catalogued domain's search half
  exists in paper-fetch, so no run plans its queries without the field's indexed terms.

## Current domains

| slug | what it adds |
|---|---|
| `cardiovascular` | Cardiac, vascular and autonomic evidence; a surrogate is not an outcome. |
| `respiratory` | Breathing, gas exchange and the chemoreflex; lung function is not a clinical endpoint. |
| `vestibular` | Balance and dizziness, graded by diagnostic accuracy rather than by trial. |
| `neuroscience` | The nervous system, with the species gap and reverse inference as first-class failure modes. |
