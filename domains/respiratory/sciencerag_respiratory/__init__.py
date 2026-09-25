"""Respiratory science: breathing, gas exchange and the chemoreflex.

Kept separate from cardiovascular even though the two share a brainstem. MeSH puts their diseases
in disjoint trees (C08 against C14), the societies and journals are different, and the endpoints
are different in kind: a spirometry index is not an ejection fraction, and a ventilator setting is
not a drug dose. A single merged domain would have to describe both badly. A question that spans
them can be answered by a domain that `extends` both (see docs/DOMAINS.md).
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
            "pooled quantitative synthesis of other trials or cohorts",
            primary=True,
            rank=90,
            human=True,
        ),
        StudyDesign(
            "guideline",
            "Official ATS/ERS statement or guideline",
            "an official statement, guideline or technical standard from a body such as the "
            "ATS or ERS",
            rank=85,
            human=True,
        ),
        StudyDesign(
            "randomised_trial",
            "Randomised controlled trial",
            "participants randomly assigned to an intervention and a comparator",
            primary=True,
            rank=80,
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
            "case_control",
            "Case-control or cross-sectional study",
            "human participants compared by outcome or measured at one time point",
            primary=True,
            rank=50,
            human=True,
        ),
        StudyDesign(
            "lung_function",
            "Lung function or gas-exchange study",
            "new spirometry, plethysmography, gas-exchange or ventilatory-response "
            "measurement in people",
            primary=True,
            rank=45,
            human=True,
        ),
        StudyDesign(
            "primary_animal",
            "Animal study",
            "new data from live animals, including carotid-body and ventilatory recordings",
            primary=True,
            rank=35,
        ),
        StudyDesign(
            "in_vitro",
            "In vitro or ex vivo study",
            "isolated airway, alveolar, carotid-body or brainstem-slice preparations",
            primary=True,
            rank=30,
        ),
        StudyDesign(
            "computational_model",
            "Computational or simulation model",
            "respiratory mechanics, gas-exchange or control-of-breathing simulation without "
            "new measurement",
            rank=25,
        ),
        StudyDesign("review", "Review", "narrative summary of other papers", rank=15),
        StudyDesign("methods", "Methods paper", "a device, measurement or protocol paper", rank=10),
        StudyDesign("unclear", "Unclear", "the text does not say", rank=0),
    )
)

ANCHORS = (
    Anchor(
        "mesh",
        "D012140",
        "Respiratory Tract Diseases",
        tree="C08",
        synonyms=("respiratory disease", "pulmonary disease"),
    ),
    Anchor(
        "mesh",
        "D002628",
        "Chemoreceptor Cells",
        tree="A08.675.650.915.500",
        synonyms=(
            "carotid body",
            "peripheral chemoreceptors",
            "central chemoreceptors",
            "chemoreflex",
            "hypoxic ventilatory response",
        ),
    ),
    Anchor(
        "mesh",
        "D065809",
        "Respiratory Sinus Arrhythmia",
        tree="G09.330.380.500.715",
        synonyms=("cardiorespiratory coupling",),
    ),
    Anchor(
        "free-text",
        "",
        "control of breathing",
        synonyms=(
            "respiratory rhythm generation",
            "pre-Botzinger complex",
            "ventilatory drive",
            "hypercapnic ventilatory response",
        ),
    ),
    Anchor(
        "free-text",
        "",
        "pulmonary function testing",
        synonyms=(
            "spirometry",
            "FEV1",
            "FVC",
            "diffusing capacity",
            "DLCO",
            "body plethysmography",
        ),
    ),
    Anchor(
        "free-text",
        "",
        "dyspnoea",
        synonyms=("dyspnea", "breathlessness", "air hunger", "respiratory discomfort"),
    ),
)

MEASURES = (
    Measure(
        "volume",
        "Lung volumes and flows",
        ("L", "mL", "L/min", "L/s"),
        aliases=("FEV1", "FVC", "tidal volume", "minute ventilation"),
    ),
    Measure(
        "gas_tension",
        "Blood gas tensions",
        ("mmHg", "kPa"),
        aliases=("PaO2", "PaCO2", "end-tidal CO2"),
    ),
    Measure(
        "saturation",
        "Saturations and predicted fractions",
        ("%", "percent"),
        aliases=("SpO2", "SaO2", "percent predicted", "FEV1/FVC ratio"),
    ),
    Measure(
        "breath_rate",
        "Respiratory rate",
        ("breaths/min", "bpm"),
        aliases=("respiratory rate", "breathing frequency"),
    ),
    Measure(
        "airway_pressure",
        "Airway and ventilator pressures",
        ("cmH2O", "cm H2O"),
        aliases=("PEEP", "plateau pressure", "driving pressure"),
    ),
)

SOURCES = (
    Source(
        "pubmed",
        "PubMed / MEDLINE",
        "literature",
        "https://pubmed.ncbi.nlm.nih.gov/",
        "MeSH-indexed; the C08 disease tree and the chemoreception descriptors.",
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
        "clinicaltrials",
        "ClinicalTrials.gov",
        "registry",
        "https://clinicaltrials.gov/api/v2/studies",
        "Registration and pre-specified endpoints for ventilation and inhaled-therapy trials.",
    ),
    Source(
        "ats_ers",
        "ATS / ERS official documents",
        "guideline",
        "https://site.thoracic.org/about-us/news/official-ats-documents",
        "Joint statements set the technical standards this field reports against.",
    ),
    Source(
        "medrxiv",
        "medRxiv",
        "literature",
        "https://api.biorxiv.org/",
        "Preprints are labelled, never excluded.",
    ),
)

POLICY = DomainPolicy(
    slug="respiratory",
    label="Respiratory science",
    scope=(
        "Breathing, gas exchange, airway and lung disease, and the chemoreflex control of "
        "ventilation, from carotid-body recordings to multicentre ventilation trials."
    ),
    taxonomy=TAXONOMY,
    sources=SOURCES,
    anchors=ANCHORS,
    measures=MEASURES,
    search_guidance=(
        "Write queries suited to PubMed/MEDLINE, Europe PMC and OpenAlex. Prefer the MeSH term "
        "for the condition and name the ventilatory or gas-exchange variable that was actually "
        "measured (FEV1, hypercapnic ventilatory response, PaCO2, minute ventilation) rather "
        "than the disease alone. Control-of-breathing work is indexed under chemoreception, not "
        "under the lung."
    ),
    reporting_rule=(
        "Keep the population (species, patient group, disease severity), the respiratory "
        "challenge or intervention, the variable measured, the follow-up time, and any hedging."
    ),
    generalisation_rule=(
        "Do not generalise animal, ex-vivo or simulated results to patients. A lung-function or "
        "gas-exchange index is not a clinical outcome: a change in FEV1, PaCO2 or ventilatory "
        "response is not a change in exacerbations, ventilator-free days or mortality, and must "
        "not be restated as one. A result under an experimental gas challenge is not a result "
        "during spontaneous breathing. A result in intensive care is not a result in the clinic."
    ),
    gap_guidance=(
        "Look for: a claim resting on one trial or one cohort; a lung-function surrogate standing "
        "in for a clinical outcome; an animal chemoreflex result carried across to people; a "
        "ventilation claim whose source does not state the mode, the settings or the sedation; "
        "missing null results or failed replications."
    ),
    critique_rules=(
        CritiqueRule(
            "preclinical_only",
            frozenset({"primary_animal", "in_vitro", "computational_model"}),
            "preclinical evidence only; no human data was retrieved for a claim about people.",
        ),
        CritiqueRule(
            "observational_only",
            frozenset({"observational_cohort", "case_control"}),
            "observational evidence only; confounding by severity is not excluded.",
        ),
        CritiqueRule(
            "physiology_only",
            frozenset({"lung_function"}),
            "lung-function measurement only; no clinical outcome was retrieved.",
        ),
    ),
    min_primary_for_strong=2,
    examples=(
        "Does carotid body denervation reduce sympathetic activity in hypertension?",
        "Does slow breathing at six breaths per minute alter the hypercapnic ventilatory response?",
        "Is respiratory sinus arrhythmia a valid index of cardiac vagal tone during exercise?",
    ),
)

DOMAIN = declare(
    name="science-rag-domain-respiratory",
    module=__name__,
    version=__version__,
    core_requires=CORE_REQUIRES,
    job="Research breathing, gas exchange and the chemoreflex by this field's own endpoints.",
    policy=POLICY,
)
