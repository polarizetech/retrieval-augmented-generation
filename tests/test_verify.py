"""The deterministic half of verification: quotes, numbers, citation markers, settling."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from research_pipeline import verify
from research_pipeline.domains import GENERIC, DomainPolicy, Measure
from research_pipeline.schema import Check, Claim
from tests.conftest import make_evidence

PASSAGE = (
    "Participants were recruited from two clinics.  Resting systolic pressure fell by 5 mmHg\n"
    "after training. The intervention did not change heart rate in 40 adults."
)

WITH_UNITS = DomainPolicy(
    slug="units-test",
    label="Units test",
    scope="A policy that declares units, for testing.",
    taxonomy=GENERIC.taxonomy,
    measures=(
        Measure("pressure", "Blood pressure", ("mmHg",)),
        Measure("duration", "Duration", ("ms", "s")),
        Measure("frequency", "Frequency", ("Hz",)),
    ),
)


class TestQuoteAnchoring:
    def test_exact_quote_survives_whitespace_and_case(self) -> None:
        quote, ratio = verify.anchor_quote(
            "resting systolic pressure fell by 5 mmHg after training.", PASSAGE
        )
        assert ratio == 1.0
        assert quote is not None
        assert quote in PASSAGE  # the source's own text, line break included

    def test_near_paraphrase_is_replaced_by_the_source_sentence(self) -> None:
        quote, ratio = verify.anchor_quote(
            "The intervention did not change the heart rate in 40 adults.", PASSAGE
        )
        assert 0.85 <= ratio < 1.0
        assert quote == "The intervention did not change heart rate in 40 adults."

    def test_a_repair_that_drops_a_negation_is_refused(self) -> None:
        # One word apart and highly similar, but the model's version reverses the finding.
        quote, ratio = verify.anchor_quote(
            "The intervention did change heart rate in 40 adults.", PASSAGE
        )
        assert ratio > 0.85
        assert quote is None

    def test_a_repair_that_changes_a_number_is_refused(self) -> None:
        quote, _ = verify.anchor_quote(
            "The intervention did not change heart rate in 41 adults.", PASSAGE
        )
        assert quote is None

    def test_too_short_a_quote_anchors_nothing(self) -> None:
        assert verify.anchor_quote("heart rate", PASSAGE) == (None, 0.0)

    def test_an_invented_quote_is_rejected(self) -> None:
        quote, _ = verify.anchor_quote(
            "Training abolished all hypertension in every single participant.", PASSAGE
        )
        assert quote is None


class TestNumbers:
    def test_a_number_absent_from_the_source_is_reported(self) -> None:
        assert verify.unsupported_numbers("pressure fell by 7 mmHg in 40 adults", [PASSAGE]) == [
            "7"
        ]

    @pytest.mark.parametrize(
        "claim",
        [
            "Fló et al. 2024 found it in 40 adults",
            "Smith & Jones (2019) found it in 40 adults",
            "as reported (Smith, 2019), 40 adults",
            "Pan 2026: 40 adults",
            "Wells et al. (2016a) studied 40 adults",
        ],
    )
    def test_a_citation_year_is_not_a_claimed_number(self, claim: str) -> None:
        assert verify.unsupported_numbers(claim, [PASSAGE]) == []

    @pytest.mark.parametrize(
        ("claim", "missing"),
        [("In 2019, 40 adults were studied", ["2019"]), ("the 2024 cohort of 40 adults", ["2024"])],
    )
    def test_a_year_that_is_not_a_citation_is_still_checked(
        self, claim: str, missing: list[str]
    ) -> None:
        assert verify.unsupported_numbers(claim, [PASSAGE]) == missing

    def test_identifiers_are_not_numbers(self) -> None:
        assert verify.unsupported_numbers("CA1 and Nav1.7 were studied", ["no numbers here"]) == []

    @pytest.mark.parametrize(
        ("claim", "source"),
        [
            ("p = .05", "p = 0.05"),
            ("0.50 mg", "0.5 mg"),
            ("a 0,5 s delay", "a 0.5 s delay"),
            ("10-Hz tone", "a 10 Hz tone"),
            ("10Hz tone", "a 10 Hz tone"),
        ],
    )
    def test_spellings_of_the_same_value_match(self, claim: str, source: str) -> None:
        assert verify.unsupported_numbers(claim, [source], WITH_UNITS) == []

    def test_a_declared_unit_must_match_its_measure(self) -> None:
        # The digit is present, but as a duration, not a frequency.
        assert verify.unsupported_numbers("a 10 Hz tone", ["latency was 10 ms"], WITH_UNITS) == [
            "10 Hz"
        ]

    def test_the_generic_policy_checks_bare_numbers_only(self) -> None:
        assert verify.unsupported_numbers("a 10 Hz tone", ["latency was 10 ms"]) == []


class TestCitationMarkers:
    @pytest.mark.parametrize(
        "sentence",
        [
            "consistent with earlier responses3-5,10.",
            "as reported previously12.",
            "Smith et al. reported the opposite.",
            "a replicated effect [4, 7].",
            "as shown before (Garcia, 2019).",
        ],
    )
    def test_reported_results_are_second_hand(self, sentence: str) -> None:
        assert verify.cites_other_work(sentence)

    @pytest.mark.parametrize(
        "sentence",
        [
            "Systolic pressure fell by 5 mmHg in 48 adults (p = 0.02).",
            "Knockout of trpv1 and mecp2 abolished the response.",
            "Nav1.7 currents recovered within 100 ms.",
        ],
    )
    def test_first_hand_results_are_not(self, sentence: str) -> None:
        assert not verify.cites_other_work(sentence)


class TestSettling:
    evidence: ClassVar = {"E1": make_evidence("E1", "W1")}
    passages: ClassVar = {"E1": "Training lowered systolic pressure by 5 mmHg in adults."}

    def settle(
        self,
        verdicts: dict[str, str],
        text: str = "Training lowered systolic pressure.",
        evidence_ids: tuple[str, ...] = ("E1",),
    ) -> Claim:
        claim = Claim("C1", "S1", text, list(evidence_ids))
        claim.checks = [Check(model, "E1", verdict, "") for model, verdict in verdicts.items()]
        verify.settle(claim, self.evidence, self.passages, list(verdicts))
        return claim

    def test_unanimous_support(self) -> None:
        claim = self.settle({"a": "SUPPORTED", "b": "SUPPORTED"})
        assert (claim.verdict, claim.supported_by) == ("SUPPORTED", ["E1"])

    def test_split_verifiers_are_disputed_not_averaged(self) -> None:
        assert self.settle({"a": "SUPPORTED", "b": "NOT_SUPPORTED"}).verdict == "DISPUTED"

    def test_the_strongest_objection_is_kept(self) -> None:
        claim = self.settle({"a": "PARTIALLY_SUPPORTED", "b": "CONTRADICTED"})
        assert claim.verdict == "CONTRADICTED"
        assert claim.supported_by == []

    def test_a_number_missing_from_the_source_removes_the_claim(self) -> None:
        # Both verifiers accepted it. The deterministic check still wins.
        claim = self.settle(
            {"a": "SUPPORTED", "b": "SUPPORTED"}, "Training lowered systolic pressure by 9 mmHg."
        )
        assert claim.verdict == verify.VERDICT_NUMBERS
        assert claim.supported_by == []
        assert "numbers_not_in_source:9" in claim.flags

    def test_unknown_evidence_ids_are_flagged_and_ignored(self) -> None:
        claim = self.settle({"a": "SUPPORTED"}, evidence_ids=("E1", "E9"))
        assert claim.verdict == "SUPPORTED"
        assert "unknown_evidence_ids:E9" in claim.flags

    def test_a_claim_with_no_known_evidence_is_not_supported(self) -> None:
        claim = self.settle({"a": "SUPPORTED"}, evidence_ids=("E9",))
        assert claim.verdict == "NOT_SUPPORTED"

    def test_settling_without_verifiers_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="verifier"):
            verify.settle(Claim("C1", "S1", "t", ["E1"]), self.evidence, self.passages, [])


@pytest.mark.parametrize(
    ("doi", "parent"),
    [
        ("10.1371/journal.pcbi.1010490.g004", "10.1371/journal.pcbi.1010490"),
        ("10.1371/journal.pone.0108224.t001", "10.1371/journal.pone.0108224"),
        ("10.7717/peerj.5601/fig-2", "10.7717/peerj.5601"),
        ("10.7554/eLife.09868.003", "10.7554/elife.09868"),
        ("10.1038/nature17427", "10.1038/nature17427"),
        ("10.1371/journal.pone.0108224", "10.1371/journal.pone.0108224"),
    ],
)
def test_component_dois_resolve_to_their_article(doi: str, parent: str) -> None:
    from research_pipeline.schema import parent_doi

    assert parent_doi(doi) == parent


class TestAVerifierCallThatFails:
    """A call that ran out of time or tokens is recorded and the run goes on. (Found by a live
    run: one verify call generated 16,000 tokens and no answer, and ended the whole run.)"""

    def _claims(self) -> tuple[list[Any], dict[str, Any], dict[str, str]]:
        from research_pipeline.schema import Claim

        ev = {"S1-E1": make_evidence("S1-E1", "W1"), "S1-E2": make_evidence("S1-E2", "W2")}
        passages = {"S1-E1": "Training lowered pressure.", "S1-E2": "Training lowered pressure."}
        return [Claim("S1-C1", "S1", "Training lowered pressure.", ["S1-E1"])], ev, passages

    def test_a_claim_no_verifier_answered_is_removed_not_kept(self) -> None:
        from research_pipeline.llm import ModelOutputError

        class Stalls:
            calls: list[Any] = []  # noqa: RUF012

            def chat_json(self, *_: Any, **__: Any) -> dict[str, Any]:
                raise ModelOutputError("verify: no complete JSON answer within 2048 tokens")

        claims, ev, passages = self._claims()
        verify.check_claims(claims, ev, passages, Stalls(), ["m"])  # type: ignore[arg-type]
        assert claims[0].verdict == verify.VERDICT_UNCHECKED
        assert claims[0].supported_by == []
        assert claims[0].checks[0].verdict == verify.CHECK_FAILED

    def test_a_failed_check_never_outranks_a_real_verdict(self) -> None:
        from research_pipeline.schema import Check

        claims, ev, passages = self._claims()
        claims[0].checks = [
            Check("a", "S1-E1", verify.CHECK_FAILED, "x"),
            Check("b", "S1-E1", "NOT_SUPPORTED", "y"),
        ]
        verify.settle(claims[0], ev, passages, ["a", "b"])
        assert claims[0].verdict == "NOT_SUPPORTED"
        claims[0].checks = [
            Check("a", "S1-E1", verify.CHECK_FAILED, "x"),
            Check("b", "S1-E1", "SUPPORTED", "y"),
        ]
        verify.settle(claims[0], ev, passages, ["a", "b"])
        assert claims[0].verdict == "DISPUTED"  # one verifier accepted; the other did not answer

    def test_a_dead_client_or_server_still_ends_the_run(self) -> None:
        from research_pipeline.llm import LLMError

        class Gone:
            calls: list[Any] = []  # noqa: RUF012

            def chat_json(self, *_: Any, **__: Any) -> dict[str, Any]:
                raise LLMError("the client stopped answering this run")

        claims, ev, passages = self._claims()
        with pytest.raises(LLMError, match="stopped answering"):
            verify.check_claims(claims, ev, passages, Gone(), ["m"])  # type: ignore[arg-type]
