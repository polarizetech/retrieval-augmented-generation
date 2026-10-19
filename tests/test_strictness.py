from __future__ import annotations

import pytest

from research_pipeline.strictness import resolve


def test_named_strictness_profiles() -> None:
    assert resolve("").name == "grounded"
    assert resolve("EXPLORE").corpus_ready is False
    publication = resolve("publication")
    assert publication.require_independent_verifier is True
    assert publication.corpus_ready is True


def test_unknown_strictness_is_refused() -> None:
    with pytest.raises(ValueError, match="explore, grounded, publication"):
        resolve("optimistic")
