"""Vestibular science: balance, the vestibulo-ocular reflex, and dizziness.

This field's evidence looks different from the other two organ systems, and the difference is the
whole reason it is its own domain. There is comparatively little randomised-trial literature; the
load-bearing designs are diagnostic accuracy against a reference standard, psychophysics, and
posturography. Grading it with a trial-shaped rubric would mark most of the real evidence down,
so the taxonomy here puts a well-conducted accuracy study near the top instead.
"""

from __future__ import annotations

from research_pipeline.domains import (
    Anchor,
    CritiqueRule,
    DomainPolicy,
    Measure,
    Source,
    StudyDesign,
    Taxonomy,
    declare,
)

__version__ = "0.1.0"
CORE_REQUIRES = ">=0.2.0,<0.3"

TAXONOMY = Taxonomy(
    (
        StudyDesign(
            "meta_analysis",
            "Systematic review or meta-analysis",
            "pooled quantitative synthesis of other studies",
            primary=True,
            rank=90,
            human=True,
        ),
        StudyDesign(
            "consensus_criteria",
            "Consensus diagnostic criteria",
            "classification or diagnostic criteria agreed by a body such as the Barany Society",
            rank=85,
            human=True,
        ),
        StudyDesign(
            "diagnostic_accuracy",
            "Diagnostic accuracy study",
            "a vestibular test compared against a reference standard, reporting sensitivity, "
            "specificity or likelihood ratios",
            primary=True,
            rank=80,
            human=True,
        ),
        StudyDesign(
            "randomised_trial",
            "Randomised controlled trial",
            "participants randomly assigned to a treatment, manoeuvre or rehabilitation "
            "programme and a comparator",
            primary=True,
            rank=78,
            human=True,
        ),
        StudyDesign(
            "observational_cohort",
            "Cohort or registry study",
            "human participants followed over time without random assignment",
            primary=True,
            rank=60,
            human=True,
        ),
        StudyDesign(
            "psychophysics",
            "Psychophysical or posturographic study",
            "thresholds, perceptual reports, eye movements or postural sway measured in people",
            primary=True,
            rank=55,
            human=True,
        ),
        StudyDesign(
            "case_series",
            "Case series or case report",
            "a described series of patients with no comparison group",
            primary=True,
            rank=35,
            human=True,
        ),
        StudyDesign(
            "primary_animal",
            "Animal study",
            "new data from live animals, including afferent and vestibular-nucleus recordings",
            primary=True,
            rank=40,
        ),
        StudyDesign(
            "in_vitro",
            "In vitro or ex vivo study",
            "isolated labyrinth, hair-cell or brainstem-slice preparations",
            primary=True,
            rank=30,
        ),
        StudyDesign(
            "computational_model",
            "Computational or simulation model",
            "sensory-integration or biomechanical simulation without new measurement",
            rank=25,
        ),
        StudyDesign("review", "Review", "narrative summary of other papers", rank=15),
        StudyDesign("methods", "Methods paper", "a test, device or protocol paper", rank=10),
        StudyDesign("unclear", "Unclear", "the text does not say", rank=0),
    )
)

ANCHORS = (
    Anchor(
        "mesh",
        "D015837",
        "Vestibular Diseases",
        tree="C09.218.568.900",
        synonyms=("vestibular disorder", "vestibulopathy", "dizziness", "vertigo"),
    ),
    Anchor(
        "mesh",
        "D014722",
        "Vestibule, Labyrinth",
        tree="A09.246.300.909",
        synonyms=(
            "vestibular apparatus",
            "otolith organs",
            "utricle",
            "saccule",
            "semicircular canals",
        ),
    ),
    Anchor(
        "free-text",
        "",
        "vestibulo-ocular reflex",
        synonyms=("VOR", "VOR gain", "video head impulse test", "vHIT", "caloric testing"),
    ),
    Anchor(
        "free-text", "", "vestibular evoked myogenic potential", synonyms=("VEMP", "cVEMP", "oVEMP")
    ),
    Anchor(
        "free-text",
        "",
        "vestibulo-sympathetic reflex",
        synonyms=(
            "vestibulo-autonomic",
            "vestibular autonomic interaction",
            "orthostatic vestibular response",
        ),
    ),
    Anchor(
        "free-text",
        "",
        "persistent postural-perceptual dizziness",
        synonyms=(
            "PPPD",
            "vestibular migraine",
            "benign paroxysmal positional vertigo",
            "BPPV",
            "Meniere disease",
        ),
    ),
)

MEASURES = (
    Measure(
        "angular",
        "Angles, tilts and angular velocity",
        ("deg", "degrees", "deg/s", "°/s"),
        aliases=("subjective visual vertical", "slow-phase velocity", "sway angle"),
    ),
    Measure(
        "gain",
        "Reflex gain",
        ("gain",),
        aliases=("VOR gain",),
        note="VOR gain is a dimensionless ratio; a gain without its test is not comparable.",
    ),
    Measure(
        "level",
        "Sound and stimulus levels",
        ("dB", "dB nHL", "dB SPL"),
        aliases=("VEMP threshold", "stimulus level"),
    ),
    Measure(
        "asymmetry",
        "Asymmetries and proportions",
        ("%", "percent"),
        aliases=("canal paresis", "caloric asymmetry", "sensitivity", "specificity"),
    ),
    Measure("latency", "Response latency", ("ms", "msec"), aliases=("p13 latency", "n23 latency")),
)

SOURCES = (
    Source(
        "pubmed",
        "PubMed / MEDLINE",
        "literature",
        "https://pubmed.ncbi.nlm.nih.gov/",
        "MeSH-indexed; the C09 otorhinolaryngologic tree.",
    ),
    Source(
        "europepmc",
        "Europe PMC",
        "literature",
        "https://www.ebi.ac.uk/europepmc/webservices/rest/",
        "Open-access full text plus MeSH terms in one record.",
    ),
    Source(
        "openalex",
        "OpenAlex",
        "literature",
        "https://api.openalex.org/works",
        "Venue and open-access metadata, and the citation graph.",
    ),
    Source(
        "barany",
        "Barany Society classification documents",
        "guideline",
        "https://www.jvr-web.org",
        "This field's consensus diagnostic criteria (BPPV, vestibular migraine, PPPD, "
        "Meniere disease) are published as papers in the Journal of Vestibular Research, "
        "the society's venue since 2015. The society has no dedicated API or stable domain "
        "of its own that resolved from this environment; the documents are reached through "
        "PubMed/Europe PMC like any other paper, and jvr-web.org is recorded here as the "
        "journal of record rather than as a scrape target.",
    ),
)

POLICY = DomainPolicy(
    slug="vestibular",
    label="Vestibular science",
    scope=(
        "Balance, the vestibular end organs and their central pathways, oculomotor and "
        "postural responses, and the disorders of dizziness and vertigo."
    ),
    taxonomy=TAXONOMY,
    sources=SOURCES,
    anchors=ANCHORS,
    measures=MEASURES,
    search_guidance=(
        "Write queries suited to PubMed/MEDLINE, Europe PMC and OpenAlex. Name the test rather "
        "than the symptom: vHIT, caloric irrigation, VEMP, posturography and subjective visual "
        "vertical each have their own literature, and a query for 'dizziness' returns almost none "
        "of it. Use the consensus diagnostic label (BPPV, vestibular migraine, PPPD) where one "
        "exists, and its expansion alongside the abbreviation."
    ),
    reporting_rule=(
        "Keep the population (patient group, age, and whether it was a specialist or unselected "
        "sample), the test and its stimulus parameters, the reference standard, and any hedging."
    ),
    generalisation_rule=(
        "Do not generalise animal or isolated-preparation results to patients. A test result is "
        "not a diagnosis, and an abnormal test in a specialist clinic does not carry its "
        "accuracy into primary care: sensitivity and specificity depend on the sample that was "
        "tested. A threshold or reflex gain measured with one stimulus is not comparable to one "
        "measured with another. A symptom improving is not the lesion recovering."
    ),
    gap_guidance=(
        "Look for: an accuracy claim whose source gives no reference standard or no clear "
        "recruitment; a single specialist-clinic sample generalised to the population; a reflex "
        "gain or threshold quoted without its test and stimulus; a treatment claim resting on a "
        "case series where a controlled comparison exists; missing null results."
    ),
    critique_rules=(
        CritiqueRule(
            "preclinical_only",
            frozenset({"primary_animal", "in_vitro", "computational_model"}),
            "preclinical evidence only; no human data was retrieved for a claim about people.",
        ),
        CritiqueRule(
            "uncontrolled_only",
            frozenset({"case_series"}),
            "case series only; with no comparison group this cannot separate the "
            "treatment from the natural course.",
        ),
        CritiqueRule(
            "laboratory_only",
            frozenset({"psychophysics", "diagnostic_accuracy"}),
            "laboratory measurement only; no symptom or functional outcome was retrieved.",
        ),
    ),
    min_primary_for_strong=2,
    examples=(
        "Does video head impulse test gain distinguish vestibular neuritis from stroke?",
        "Does vestibular stimulation alter blood pressure through the vestibulo-sympathetic "
        "reflex?",
        "Does vestibular rehabilitation reduce dizziness handicap in chronic unilateral "
        "vestibular loss?",
    ),
)

DOMAIN = declare(
    name="rag-domain-vestibular",
    module=__name__,
    version=__version__,
    core_requires=CORE_REQUIRES,
    job=(
        "Research balance and dizziness by a diagnostic-accuracy evidence hierarchy, not a "
        "trial one."
    ),
    policy=POLICY,
)
