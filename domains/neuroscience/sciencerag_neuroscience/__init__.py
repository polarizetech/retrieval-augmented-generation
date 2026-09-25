"""Neuroscience: the nervous system, from ion channels to behaviour, stated as a policy.

The study-design list, the "animal or in-vitro only" critique and the species gap in the
generalisation rule are this field's judgements, so they live here and not in the engine, where
no other field could disagree with them.

Neuroscience's characteristic failure modes (analytic flexibility, circular analysis, cluster
inference, reverse inference, underpowered samples) are declared here as gap guidance and
critique rules, because they belong to this field rather than to science in general.
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

# The original STUDY_TYPES list, in the same order, with the ranks and definitions that were
# previously implied by the classifier prompt and the grading code.
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
            "primary_human",
            "Human study",
            "new data from human participants or patients",
            primary=True,
            rank=70,
            human=True,
        ),
        StudyDesign(
            "primary_animal",
            "Animal study",
            "new data from live animals (in vivo)",
            primary=True,
            rank=50,
        ),
        StudyDesign(
            "in_vitro",
            "In vitro study",
            "cells, slices, cultures or organoids",
            primary=True,
            rank=40,
        ),
        StudyDesign(
            "computational_model",
            "Computational model",
            "simulation or theory with no new measurement",
            rank=30,
        ),
        StudyDesign("review", "Review", "narrative summary of other papers", rank=15),
        StudyDesign(
            "methods", "Methods paper", "a technique, tool or analysis-method paper", rank=10
        ),
        StudyDesign("unclear", "Unclear", "the text does not say", rank=0),
    )
)

ANCHORS = (
    Anchor(
        "mesh",
        "D001479",
        "Basal Ganglia",
        tree="A08.186.211.200.885.287.249",
        synonyms=("striatum", "caudate nucleus", "putamen"),
    ),
    Anchor(
        "mesh",
        "D006624",
        "Hippocampus",
        tree="A08.186.211.180.405",
        synonyms=("CA1", "CA3", "dentate gyrus", "hippocampal formation"),
    ),
    Anchor(
        "mesh",
        "D002540",
        "Cerebral Cortex",
        tree="A08.186.211.200.885.287.500",
        synonyms=("neocortex", "cortical"),
    ),
    Anchor(
        "mesh",
        "D009415",
        "Nerve Net",
        tree="A08.511",
        synonyms=("neural circuit", "neuronal network", "microcircuit"),
    ),
    Anchor(
        "mesh",
        "D017981",
        "Receptors, Neurotransmitter",
        tree="D12.776.543.750.720",
        synonyms=("neurotransmitter receptor", "ion channel", "synaptic receptor"),
    ),
    Anchor(
        "free-text",
        "",
        "neural oscillations",
        synonyms=(
            "delta waves",
            "theta rhythm",
            "alpha power",
            "beta band",
            "local field potential",
            "LFP",
        ),
    ),
    Anchor(
        "free-text",
        "",
        "synaptic plasticity",
        synonyms=(
            "long-term potentiation",
            "LTP",
            "long-term depression",
            "LTD",
            "spike-timing-dependent plasticity",
        ),
    ),
    Anchor(
        "free-text",
        "",
        "optogenetics",
        synonyms=("channelrhodopsin", "chemogenetics", "DREADD", "closed-loop stimulation"),
    ),
)

MEASURES = (
    Measure(
        "frequency",
        "Oscillation and firing frequency",
        ("Hz", "hertz", "kHz"),
        aliases=("oscillation frequency", "stimulation frequency", "firing rate", "spikes/s"),
        note="A result at one frequency band does not license a claim about another.",
    ),
    Measure(
        "time",
        "Latency and duration",
        ("ms", "msec", "s", "min"),
        aliases=("response latency", "inter-spike interval", "time constant"),
    ),
    Measure(
        "potential",
        "Membrane and field potential",
        ("mV", "uV", "µV"),
        aliases=("membrane potential", "EPSP amplitude", "LFP amplitude"),
    ),
    Measure(
        "current",
        "Current and conductance",
        ("pA", "nA", "nS", "pS"),
        aliases=("synaptic current", "conductance"),
    ),
    Measure(
        "concentration",
        "Drug and ion concentration",
        ("uM", "µM", "nM", "mM"),
        aliases=("bath concentration", "dose"),
    ),
    Measure(
        "distance",
        "Anatomical distance and coordinates",
        ("mm", "um", "µm"),
        aliases=("MNI coordinate", "electrode depth", "spine size"),
    ),
    Measure(
        "proportion",
        "Proportions and changes",
        ("%", "percent", "fold"),
        aliases=("percent change", "fold increase", "proportion of neurons"),
    ),
)

SOURCES = (
    Source(
        "pubmed",
        "PubMed / MEDLINE",
        "literature",
        "https://pubmed.ncbi.nlm.nih.gov/",
        "MeSH-indexed; the A08 nervous-system anatomy tree and the F02 psychological "
        "phenomena tree.",
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
        "biorxiv",
        "bioRxiv",
        "literature",
        "https://api.biorxiv.org/",
        "The Neuroscience category is among the largest on the server; much of the field "
        "appears here first. Preprints are labelled, never excluded.",
    ),
    Source(
        "openneuro",
        "OpenNeuro",
        "dataset",
        "https://openneuro.org/",
        "Whether the data behind an imaging claim are available at all.",
    ),
    Source(
        "neurosynth",
        "Neurosynth / NeuroQuery",
        "dataset",
        "https://neurosynth.org/",
        "Coordinate-based meta-analytic maps, as a check on a claimed localisation.",
    ),
)

POLICY = DomainPolicy(
    slug="neuroscience",
    label="Neuroscience",
    scope=(
        "Nervous system structure and function from ion channels to behaviour: circuits, "
        "oscillations, plasticity, neuroimaging and computational models."
    ),
    taxonomy=TAXONOMY,
    sources=SOURCES,
    anchors=ANCHORS,
    measures=MEASURES,
    search_guidance=(
        "Write queries suited to PubMed and OpenAlex. Name the structure, the signal and the "
        "preparation separately (hippocampus, theta rhythm, freely moving rats) rather than "
        "asking the question in prose. Expand abbreviations alongside themselves — LTP, LFP and "
        "DREADD are each ambiguous on their own — and include one older or alternative term for "
        "the structure, since the anatomical nomenclature has changed more than once."
    ),
    reporting_rule=(
        "Keep the species and preparation, the brain region, the recording or imaging modality, "
        "the stimulus or manipulation and its parameters, the sample size, and any hedging."
    ),
    generalisation_rule=(
        "Do not generalise animal, slice or simulated results to humans; the species gap is the "
        "single most common overreach in this field. Do not transfer across modality or "
        "parameter: a result in one frequency band does not license a claim about another, and "
        "an optogenetic result does not license a claim about a drug. Do not infer a mental state "
        "from where activation appeared. A correlation between a neural signal and a behaviour "
        "is not evidence that the signal causes the behaviour."
    ),
    gap_guidance=(
        "Look for: a claim resting on one small sample; an animal or slice result carried across "
        "to people; a functional claim inferred backwards from a region's activation; an imaging "
        "result reported without its correction for multiple comparisons or its cluster-forming "
        "threshold; a result whose analysis choices appear to have been made after seeing the "
        "data, or where the selection of neurons, voxels or trials was made using the same data "
        "the effect was then measured in; missing null results or failed replications."
    ),
    critique_rules=(
        CritiqueRule(
            "preclinical_only",
            frozenset({"primary_animal", "in_vitro"}),
            "animal or in-vitro evidence only; no human data was retrieved.",
        ),
        CritiqueRule(
            "simulation_only",
            frozenset({"computational_model"}),
            "model evidence only; no empirical measurement was retrieved.",
        ),
        CritiqueRule(
            "secondhand_only",
            frozenset({"review", "methods"}),
            "no first-hand study was retrieved; every source restates other work.",
        ),
    ),
    min_primary_for_strong=2,
    examples=(
        "Does repetitive transcranial magnetic stimulation of motor cortex change motor-evoked "
        "potential amplitude?",
        "Is long-term potentiation necessary for spatial memory formation?",
        "Do place cells remap when the environment changes shape?",
    ),
)

DOMAIN = declare(
    name="science-rag-domain-neuroscience",
    module=__name__,
    version=__version__,
    core_requires=CORE_REQUIRES,
    job="Research the nervous system with the species gap and reverse inference treated as "
    "first-class failure modes.",
    policy=POLICY,
)
