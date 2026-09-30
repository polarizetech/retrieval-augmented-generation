"""math__ lookups: calculator records from the corpus, and BioNumbers entries, verbatim, cached."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_mcp.quantities import QuantityStore, parse_bionumber

# The shape of a BioNumbers entry page (labels, then values), with made-up content.
PAGE = """<html><head><title>Demo quantity - BNID 123456</title><script>var x = 1;</script></head>
<body><a>Home</a><a>Login</a>
<h1>Demo quantity for a test organism</h1>
<div>Value</div><div>4.2</div><div>mm</div>
<div>Range</div><div>3.9 - 4.5 mm</div>
<div>Organism</div><div>Test organism</div>
<div>Reference</div><div>Author A, A demo paper, J Demo 2001 1: 1-2.</div>
<div>PubMed ID</div><div>1234567</div>
<div>Method</div><div>Measured &amp; averaged.</div>
<div>Comments</div><div>Varies with age.</div>
<div>Entered by</div><div>Someone</div>
<div>ID</div><div>123456</div>
<div>Related BioNumbers</div><div>Another quantity</div><div>ID: 999</div>
</body></html>"""

RECORD = """# ear-canal-resonance

**What it computes.** The first resonance of the ear canal from its length.

## Equation

f = c / (4 L), with c the speed of sound (m/s) and L the canal length (m).

## Parameters

| symbol | value | units | source |
|---|---|---|---|
| c | 343 | m/s | [LIT: doi:10.0000/demo] |

## Valid range

Adult canals, 20 to 30 mm.
"""


def test_parse_bionumber_reads_every_field_verbatim() -> None:
    rec = parse_bionumber(PAGE)
    assert rec["title"] == "Demo quantity for a test organism"
    assert (rec["value"], rec["units"]) == ("4.2", "mm")
    assert rec["range"] == "3.9 - 4.5 mm"
    assert rec["pubmed_id"] == "1234567"
    assert rec["method"] == "Measured & averaged."
    assert rec["id"] == "123456"
    assert "Another quantity" not in json.dumps(rec)  # related entries are not this entry


def test_parse_bionumber_refuses_other_pages() -> None:
    with pytest.raises(ValueError, match="not a BioNumbers entry"):
        parse_bionumber("<html><body>Search results</body></html>")


def test_bionumber_is_fetched_once_then_cached(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(url: str) -> str:
        calls.append(url)
        return PAGE

    store = QuantityStore(None, tmp_path, fetch=fetch)
    first = store.bionumber("BNID 123456")
    assert first["cached"] is False
    assert first["url"].endswith("id=123456")
    assert "BNID 123456" in first["cite_as"]
    second = store.bionumber(123456)
    assert second["cached"] is True
    assert second["value"] == "4.2"
    assert len(calls) == 1


def test_bionumber_checks_the_id(tmp_path: Path) -> None:
    store = QuantityStore(None, tmp_path, fetch=lambda _url: PAGE)
    with pytest.raises(ValueError, match="returned entry '123456' for BNID 7"):
        store.bionumber("7")
    with pytest.raises(ValueError, match="a BNID is a number"):
        store.bionumber("abc")


def test_lookup_returns_matching_records_with_their_sections(tmp_path: Path) -> None:
    corpus = tmp_path / "research"
    (corpus / "projects" / "hearing" / "calculators").mkdir(parents=True)
    (corpus / "projects" / "hearing" / "calculators" / "ear-canal-resonance.md").write_text(RECORD)
    other = corpus / "projects" / "hearing" / "calculators" / "other.md"
    other.write_text("# other\n\n## Equation\n\nx = y\n")
    store = QuantityStore(corpus, tmp_path / "cache")
    out = store.lookup("ear canal resonance speed of sound")
    assert out["searched"] == 2
    top = out["records"][0]
    assert top["path"] == "projects/hearing/calculators/ear-canal-resonance.md"
    assert top["project"] == "hearing"
    assert top["sections"]["Equation"].startswith("f = c / (4 L)")
    assert "[LIT: doi:10.0000/demo]" in top["sections"]["Parameters"]
    assert all(r["title"] != "other" for r in out["records"])


def test_lookup_needs_a_corpus_and_words(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no research corpus"):
        QuantityStore(None, tmp_path).lookup("resonance")
    (tmp_path / "projects").mkdir()
    with pytest.raises(ValueError, match="no words"):
        QuantityStore(tmp_path, tmp_path).lookup("??")
