"""Editorial status from Crossref, with the network replaced by recorded-shape responses."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any
from unittest import mock

import pytest

from research_pipeline import integrity


def respond(message: dict[str, Any]) -> mock.MagicMock:
    response = mock.MagicMock()
    response.__enter__.return_value = io.BytesIO(json.dumps({"message": message}).encode())
    return response


def notices(*kinds: str) -> dict[str, Any]:
    return {
        "updated-by": [{"type": kind, "DOI": f"10.1000/notice-{kind}"} for kind in kinds],
        "container-title": ["Journal of Tests"],
        "type": "journal-article",
    }


@pytest.mark.parametrize(
    ("kinds", "status"),
    [
        ((), "no_notice_found"),
        (("correction",), "corrected"),
        (("expression_of_concern", "correction"), "expression_of_concern"),
        (("partial_retraction",), "partially_retracted"),
        (("correction", "retraction", "expression_of_concern"), "retracted"),
    ],
)
def test_the_most_serious_notice_wins(kinds: tuple[str, ...], status: str) -> None:
    with mock.patch("urllib.request.urlopen", return_value=respond(notices(*kinds))):
        result = integrity.check("10.1000/x")
    assert result["status"] == status
    assert result["venue"] == "Journal of Tests"
    assert len(result["notices"]) == len(kinds)


def test_a_doi_crossref_does_not_know_is_not_called_clean() -> None:
    error = urllib.error.HTTPError("u", 404, "Not Found", {}, None)  # type: ignore[arg-type]
    with mock.patch("urllib.request.urlopen", side_effect=error):
        assert integrity.check("10.5281/zenodo.1")["status"] == "not_in_crossref"


def test_a_failed_lookup_is_unchecked() -> None:
    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
        result = integrity.check("10.1000/x")
    assert result["status"] == "unchecked"
    assert "URLError" in result["error"]


def test_the_contact_address_is_sent_only_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def capture(request: Any, timeout: float) -> mock.MagicMock:
        seen.append(request.full_url)
        return respond(notices())

    monkeypatch.delenv("PIPELINE_CONTACT_EMAIL", raising=False)
    with mock.patch("urllib.request.urlopen", side_effect=capture):
        integrity.check("10.1000/x")
        monkeypatch.setenv("PIPELINE_CONTACT_EMAIL", "ops@example.org")
        integrity.check("10.1000/x")
    assert "mailto" not in seen[0]
    assert seen[1].endswith("?mailto=ops%40example.org")


def test_check_all_skips_blanks_and_duplicates() -> None:
    with mock.patch.object(integrity, "check", side_effect=lambda doi: {"doi": doi}) as check:
        results = integrity.check_all(["10.1/a", "", "10.1/a", "10.1/b"], pause=0)
    assert list(results) == ["10.1/a", "10.1/b"]
    assert check.call_count == 2
