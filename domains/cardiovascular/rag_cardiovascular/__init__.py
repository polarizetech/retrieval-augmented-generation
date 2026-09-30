"""Cardiovascular science: what this field adds on top of the domain-neutral engine.

The engine can already retrieve, quote, verify and grade. What it cannot know is that in this
field a blood-pressure change is a surrogate and not an outcome, that an observational cohort and
a randomised trial are not the same kind of evidence, or that "40" next to mmHg is a different
fact from "40" next to ms. That is what this package declares.

How the field is *searched* (the baroreflex is indexed under a descriptor most people would not
think to type) is the paper library's `cardiovascular` profile, not this package.
"""

from __future__ import annotations

from research_pipeline.domains import (
    CritiqueRule,
    DomainPolicy,
    Measure,
    StudyDesign,
    Taxonomy,
    declare,
)

__version__ = "0.1.0"
CORE_REQUIRES = ">=0.2.0,<0.3"

# Study designs, ranked the way this field ranks them. The ordering is the reason the taxonomy is
# here and not in the engine: a randomised trial outranks a cohort outranks a mechanistic
# recording, and only the field gets to say so.
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
            "Clinical practice guideline",
            "a guideline or scientific statement from a professional body such as the AHA, "
            "ACC or ESC",
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
            "diagnostic_accuracy",
            "Diagnostic or prognostic accuracy study",
            "a test or risk score evaluated against a reference standard",
            primary=True,
            rank=55,
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
            "human_physiology",
            "Human physiology study",
            "new haemodynamic, autonomic or electrophysiological measurement in people, "
            "without a clinical endpoint",
            primary=True,
            rank=45,
            human=True,
        ),
        StudyDesign(
            "primary_animal",
            "Animal study",
            "new data from live animals (in vivo)",
            primary=True,
            rank=35,
        ),
        StudyDesign(
            "in_vitro",
            "In vitro or ex vivo study",
            "isolated hearts, vessels, myocytes or tissue preparations",
            primary=True,
            rank=30,
        ),
        StudyDesign(
            "computational_model",
            "Computational or simulation model",
            "haemodynamic, electrophysiological or epidemiological simulation without new "
            "measurement",
            rank=25,
        ),
        StudyDesign("review", "Review", "narrative summary of other papers", rank=15),
        StudyDesign(
            "methods", "Methods paper", "a device, signal-processing or protocol paper", rank=10
        ),
        StudyDesign("unclear", "Unclear", "the text does not say", rank=0),
    )
)

MEASURES = (
    Measure(
        "beat_interval",
        "Time-domain HRV and intervals",
        ("ms", "msec", "milliseconds"),
        aliases=("SDNN", "RMSSD", "R-R interval", "QT interval"),
        note="SDNN and RMSSD are reported in ms per the joint Task Force standard (Heart "
        "rate variability: standards of measurement, physiological interpretation and "
        "clinical use. Circulation. 1996;93(5):1043-65. PMID 8598068).",
    ),
    Measure(
        "spectral_power",
        "Frequency-domain HRV power",
        ("ms2", "ms²", "n.u.", "nu"),
        aliases=("LF power", "HF power", "LF/HF"),
    ),
    Measure(
        "pressure",
        "Blood and filling pressures",
        ("mmHg", "kPa"),
        aliases=("systolic blood pressure", "diastolic blood pressure", "pulse pressure"),
    ),
    Measure("rate", "Heart rate", ("bpm", "beats/min"), aliases=("heart rate",)),
    Measure(
        "baroreflex_gain",
        "Baroreflex sensitivity",
        ("ms/mmHg",),
        aliases=("BRS", "baroreflex sensitivity"),
    ),
    Measure(
        "fraction",
        "Fractions and proportions",
        ("%", "percent"),
        aliases=("ejection fraction", "stenosis", "relative risk reduction"),
    ),
)

POLICY = DomainPolicy(
    slug="cardiovascular",
    label="Cardiovascular science",
    scope=(
        "Cardiac, vascular and autonomic cardiovascular literature: mechanism, measurement, "
        "diagnosis and clinical outcome, from isolated tissue to randomised trials."
    ),
    taxonomy=TAXONOMY,
    measures=MEASURES,
    reporting_rule=(
        "Keep the population (species, patient group, comorbidity), the intervention and its "
        "dose, the endpoint actually measured, the follow-up time, and any hedging."
    ),
    generalisation_rule=(
        "Do not generalise animal, ex-vivo or simulated results to patients. A surrogate is not "
        "an outcome: a change in blood pressure, ejection fraction or an HRV index is not a "
        "change in mortality and must not be restated as one. An association in a cohort is not "
        "an effect of treatment. A result in a selected trial population is not a result in "
        "unselected practice."
    ),
    gap_guidance=(
        "Look for: a claim resting on one trial or one cohort; a surrogate endpoint standing in "
        "for a clinical one; observational evidence where a randomised trial exists; a trial "
        "population that differs from the one the claim is about; missing null or "
        "failed-replication results; an HRV or baroreflex claim whose source does not state its "
        "recording length, artefact handling or posture."
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
            "observational evidence only; confounding by indication is not excluded.",
        ),
        CritiqueRule(
            "surrogate_only",
            frozenset({"human_physiology", "diagnostic_accuracy"}),
            "measurement studies only; no clinical outcome was retrieved.",
        ),
    ),
    min_primary_for_strong=2,
    examples=(
        "Does higher baroreflex sensitivity predict lower cardiovascular mortality?",
        "Does slow-paced breathing raise RMSSD in adults with hypertension?",
        "Is reduced heart rate variability a cause or a consequence of heart failure?",
    ),
)

DOMAIN = declare(
    name="rag-domain-cardiovascular",
    module=__name__,
    version=__version__,
    core_requires=CORE_REQUIRES,
    job="Research the cardiac, vascular and autonomic literature by its own evidence hierarchy.",
    policy=POLICY,
)
