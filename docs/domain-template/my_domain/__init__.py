"""A domain extension, reduced to the smallest thing that works. Copy this directory and edit.

Replace `my_domain` with your field's slug everywhere: this directory name, the package directory
name (`rag_<slug>`), the distribution name in pyproject.toml, and the entry-point key.

The one rule worth reading before you start: a domain is DATA, not behaviour. The engine never
calls back into your package to make a decision. You describe your field; the engine applies its
own rules using your description. That is what keeps a run reproducible from its log, because a
policy serialises into run.json and behaviour would not.
"""

from __future__ import annotations

from research_pipeline.domains import (
    CritiqueRule, DomainPolicy, Measure, StudyDesign, Taxonomy, declare,
)

__version__ = "0.1.0"

# The engine versions this package was written against. Checked at import, so an incompatible
# pairing fails at discovery with a clear message rather than silently grading by the wrong rubric.
CORE_REQUIRES = ">=0.2.0,<0.3"

# The kinds of study your field produces, ranked the way your field ranks them. `primary` means
# the design reports its own first-hand observations; only primary evidence can push a claim above
# "weak". An `unclear` design is mandatory — the classifier needs somewhere to put a paper whose
# text does not say. Definitions are read by a small model inside a prompt, so keep each to one
# clause of plain language.
TAXONOMY = Taxonomy((
    StudyDesign("meta_analysis", "Systematic review or meta-analysis",
                "pooled quantitative synthesis of other studies", primary=True, rank=90,
                human=True),
    StudyDesign("primary_human", "Human study", "new data from human participants",
                primary=True, rank=70, human=True),
    StudyDesign("primary_animal", "Animal study", "new data from live animals (in vivo)",
                primary=True, rank=50),
    StudyDesign("review", "Review", "narrative summary of other papers", rank=15),
    StudyDesign("unclear", "Unclear", "the text does not say", rank=0),
))

# What your field calls things, where its evidence lives and how to write its queries is the
# SEARCH half of the field. It is not declared here: it is a discipline profile in the paper
# library (see ../profile.toml, and docs/DOMAINS.md). This package is the EVIDENCE half.

# Quantities your field reports, with the units they are legitimately reported in. Declaring these
# upgrades the engine's number check: a claim's "40 mmHg" will no longer count as supported by a
# passage that only ever said "40 ms".
MEASURES = (
    Measure("example", "Example quantity", ("unit",), aliases=("what people call it",)),
)

POLICY = DomainPolicy(
    slug="my-domain",
    label="My domain",
    scope="One sentence. A reader is shown this in the answer, so it must say what was searched.",
    taxonomy=TAXONOMY,
    measures=MEASURES,
    # The paper library's profile for this field; defaults to the slug.
    profile="my-domain",

    # Prompt fragments. Each is APPENDED to an engine prompt; none replaces one. You can tighten
    # the engine's rules, never talk it out of one.
    reporting_rule="What a finding must keep to stay meaningful here (population, conditions...).",
    generalisation_rule="What may NOT be carried across populations, preparations or measures.",
    gap_guidance="What the critic should notice is missing.",

    # A weakness your field recognises, stated as a test on study design. It fires when EVERY
    # design behind a claim falls inside the set, and the planner is then asked for a search that
    # would fix it.
    critique_rules=(
        CritiqueRule("preclinical_only", frozenset({"primary_animal"}),
                     "animal evidence only; no human data was retrieved."),
    ),

    # How many independent first-hand primary sources a claim needs before this field calls its
    # support strong. Raise it for small-sample fields with unstable effects.
    min_primary_for_strong=2,

    examples=("A question this domain is built to answer.",),
)

DOMAIN = declare(
    name="rag-domain-my-domain",  # must start with rag-domain-
    module=__name__,
    version=__version__,
    core_requires=CORE_REQUIRES,
    job="One line: the single thing this domain adds that the engine could not know.",
    policy=POLICY,
    # extends={"rag-domain-other": ">=0.1.0,<0.2"},  # only for a real dependency
)
