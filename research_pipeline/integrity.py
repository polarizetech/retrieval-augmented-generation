"""Editorial status of cited works, from Crossref (which carries the Retraction Watch database).

The paper library's `is_retracted` comes from OpenAlex, whose flag has a published record of false
positives and negatives. Every DOI that ends up supporting a claim is therefore checked a second
time against Crossref's `updated-by` records at answer time, and the date of the check is logged.
Retraction, expression of concern and correction are reported as different states.

A failed lookup is reported as "unchecked". A successful lookup with no notices is reported as
"no_notice_found": Crossref only knows what publishers and Retraction Watch deposit, so the absence
of a notice is not a clean bill of health.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from . import __version__

USER_AGENT = f"kit-scientific-research-rag/{__version__}"

# Notice types, most severe first. A partial retraction withdraws some of a paper's results, so it
# is reported alongside an expression of concern rather than filed as a routine correction.
SERIOUS = {
    "retraction": "retracted",
    "removal": "retracted",
    "withdrawal": "retracted",
    "partial_retraction": "partially_retracted",
    "partial-retraction": "partially_retracted",
    "expression_of_concern": "expression_of_concern",
    "expression-of-concern": "expression_of_concern",
}
SEVERITY = [
    "retracted",
    "partially_retracted",
    "expression_of_concern",
    "corrected",
    "no_notice_found",
]
MINOR = {"correction", "corrigendum", "erratum", "addendum"}


def check(doi: str, timeout: float = 20.0) -> dict[str, Any]:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/")
    # Crossref's polite pool wants a contact address. Only an address the operator configured for
    # this purpose is ever sent.
    contact = os.environ.get("PIPELINE_CONTACT_EMAIL", "").strip()
    if contact:
        url += "?mailto=" + urllib.parse.quote(contact)
    out: dict[str, Any] = {
        "doi": doi,
        "checked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "crossref",
        "status": "unchecked",
        "notices": [],
    }
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            message = json.load(response)["message"]
    except urllib.error.HTTPError as exc:
        # Not every DOI is Crossref's (arXiv and Zenodo register with DataCite). Absence from
        # Crossref means no editorial record is available here, not that the DOI is fake.
        out["status"] = "not_in_crossref" if exc.code == 404 else "unchecked"
        out["error"] = f"HTTP {exc.code}"
        return out
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"[:160]
        return out
    status = "no_notice_found"
    for notice in message.get("updated-by") or []:
        kind = str(notice.get("type", "")).lower()
        out["notices"].append(
            {
                "type": kind,
                "doi": notice.get("DOI"),
                "source": notice.get("source"),
                "label": notice.get("label"),
            }
        )
        found = SERIOUS.get(kind) or ("corrected" if kind in MINOR else None)
        if found and SEVERITY.index(found) < SEVERITY.index(status):
            status = found
    out["status"] = status
    out["venue"] = (message.get("container-title") or [None])[0]
    out["type"] = message.get("type")
    return out


def check_all(dois: list[str], pause: float = 0.25) -> dict[str, dict[str, Any]]:
    results = {}
    for doi in dict.fromkeys(d for d in dois if d):
        results[doi] = check(doi)
        time.sleep(pause)  # single-record public pool allows 5 requests/s
    return results
