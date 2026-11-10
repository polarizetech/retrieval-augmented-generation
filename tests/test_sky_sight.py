"""sky__measure, sky__locate and the sun-sight step of `ask`: parsing a sight out of plain words,
and arranging polarize-clock's answer. polarize-clock is stubbed; a live check runs with
POLARIZE_CLOCK_LIVE=1."""

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from research_mcp import sky
from tests.test_ask import asker

LOCATE = {
    "input": {"alt_deg": 20.392, "az_deg": 212.0, "jd_ut": 2461349.4166666665, "h_m": 0.0},
    "sun": {"dec_deg": -15.6, "subsolar_lon_deg": -153.2, "ephemeris": "DE440s"},
    "candidates": [
        {
            "lat": 48.29,
            "lon": -123.059,
            "miss_deg": 0.0,
            "converged": True,
            "local_solar_time_h": 14.07,
            "local_mean_time_h": 13.796,
            "eot_min": 16.4,
            "sensitivity": {"km_per_0p1deg_alt": 12.2, "km_per_0p1deg_az": 8.4},
        }
    ],
}
MEASURE = {"alt_deg": 36.87, "az_deg": 180.0, "ray_m": 5.0, "legs": {"sin_el": 0.6, "cos_el": 0.8}}


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def get(path: str, query: dict[str, Any], timeout: float) -> dict[str, Any]:
        calls.append((path, query))
        return {"/api/locate": LOCATE, "/api/measure": MEASURE}[path]

    monkeypatch.setattr(sky, "_get", get)
    return calls


@pytest.mark.parametrize(
    ("question", "want"),
    [
        (
            "Where am I? A 1 m stick casts a 2.69 m shadow pointing 32° at 2026-11-04T22:00Z",
            {"stick": 1.0, "shadow": 2.69, "bearing": 32.0, "t": "2026-11-04T22:00Z"},
        ),
        ("locate me: stick 150 cm, shadow 4 m toward 300", {"stick": 1.5, "shadow": 4.0}),
        ("my stick is 2 metres tall and the shadow is 3.1 m long, bearing 15°", {"stick": 2.0}),
        ("Locate me: the sun is 20.4 degrees up at azimuth 212", {"alt": 20.4, "az": 212.0}),
        ("where am i? altitude 45 deg, azimuth 180", {"alt": 45.0, "az": 180.0}),
    ],
)
def test_parse_sight_reads_the_sight(question: str, want: dict[str, Any]) -> None:
    got = sky.parse_sight(question)
    assert got is not None
    assert got["missing"] == []
    for k, v in want.items():
        assert got[k] == pytest.approx(v) if isinstance(v, float) else got[k] == v


def test_parse_sight_leaves_other_requests_alone() -> None:
    assert sky.parse_sight("what is the sky key at 48.4, -123.4") is None
    assert sky.parse_sight("locate the paper on bee dances") is None


def test_parse_sight_says_what_is_missing() -> None:
    got = sky.parse_sight("where am i, sun 30 degrees high")
    assert got is not None
    assert got["missing"] == ["the Sun's azimuth"]
    got = sky.parse_sight("locate me: a 1 m stick and a 2 m shadow")
    assert got is not None
    assert got["missing"] == ["the shadow's bearing (degrees from true north)"]


def test_locate_and_measure_arrange_polarize_clocks_answer(stub: list) -> None:
    r = sky.locate("2026-11-04T22:00:00Z", stick=1.0, shadow=2.69, bearing=32.0)
    assert stub[0] == (
        "/api/locate",
        {"h": 0.0, "t": "2026-11-04T22:00:00Z", "stick": 1.0, "shadow": 2.69, "bearing": 32.0},
    )
    assert r["candidates"][0]["local_solar_time"] == "14:04:12"
    assert r["time_utc"] == "2026-11-04T22:00:00Z"
    line = sky.sight_summary(r)
    assert "48.290, -123.059" in line
    assert "about 12 km" in line
    m = sky.measure(3.0, 4.0, 0.0)
    assert stub[1] == ("/api/measure", {"stick": 3.0, "shadow": 4.0, "bearing": 0.0})
    assert m["az_deg"] == 180.0
    with pytest.raises(ValueError, match="both alt and az"):
        sky.locate(alt=20.0)
    with pytest.raises(ValueError, match="length and bearing"):
        sky.locate(stick=1.0, shadow=2.0)


def test_ask_runs_a_sun_sight_without_retrieving(tmp_path: Path, stub: list) -> None:
    calls: list[tuple[str, Any]] = []
    q = "Where am I? A 1 m stick casts a 2.69 m shadow pointing 32 degrees at 2026-11-04T22:00Z"
    out = asyncio.run(asker(tmp_path, calls).ask(q))
    assert calls == []  # no passage retrieval for a sun sight
    assert "sky" not in out["results"]  # the sight, not the key path
    assert out["results"]["sight"]["located"]["candidates"][0]["lat"] == 48.29
    assert "places you at 48.290, -123.059" in out["relay"]
    assert "Sun sight" in out["relay_full"]


def test_ask_asks_for_the_missing_bearing(tmp_path: Path, stub: list) -> None:
    out = asyncio.run(asker(tmp_path, []).ask("where am i? a 1 m stick and a 2 m shadow"))
    assert out["results"]["sight"]["missing"]
    assert stub == []
    assert "the shadow's bearing" in out["relay"]


@pytest.mark.skipif(
    os.environ.get("POLARIZE_CLOCK_LIVE") != "1", reason="set POLARIZE_CLOCK_LIVE=1"
)
def test_live_sun_sight() -> None:
    r = sky.locate("2026-11-04T22:00:00Z", stick=1.0, shadow=2.69, bearing=32.0)
    best = r["candidates"][0]
    assert abs(best["lat"] - 48.43) < 0.5
    assert abs(best["lon"] + 123.37) < 0.5
    m = sky.measure(3.0, 4.0, 0.0)
    assert m["ray_m"] == pytest.approx(5.0)
