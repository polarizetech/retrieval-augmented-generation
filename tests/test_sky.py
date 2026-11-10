"""sky__key and the sky step of `ask`: parsing, unit triangles, and arranging polarize-clock's
answer.

polarize-clock itself is stubbed: these tests check this repo's side. A live check against a running
polarize-clock server runs when POLARIZE_CLOCK_LIVE=1.
"""

import asyncio
import math
import os
from pathlib import Path
from typing import Any

import pytest

from research_mcp import sky
from tests.test_ask import asker

KEY = {
    "schema": "key/1",
    "provider": {"tool": "polarize-clock", "version": "0.4.0", "commit": "abc123"},
    "time": {"utc_iso": "2026-11-04T22:00:00", "jd_ut": 2461349.4166666665, "deltaT_s": 69.184},
    "ephemeris": "DE440s",
    "gravity": {"enu": [0.0, 0.0, -9.8093], "g_ms2": 9.8093},
    "sun": {"enu": [-0.49223, -0.79776, 0.34828], "el_deg": 20.382, "az_deg": 211.675},
    "quantisation": {"bin_deg": 0.5},
    "bins": {"lat": 96, "lon": -247, "h": 0, "el": 40, "az": 423, "tilt": None, "jdn": None},
    "hex": "27b01144e2ea8bf82264548343595a5e2292109e79ce14c5b20485babb7096e9",
    "code": {"row": 577, "rule": "prefix32"},
}
CLOCKS = {
    "time": {"ephemeris": "DE440s", "ut1_error_s": 0.9},
    "sun": {"solar_time_h": 14.049217, "lmst_h": 13.775333, "eot_min": 16.433},
    "moon": {
        "lunar_time_h": 17.372428,
        "az": 263.6,
        "el": 6.99,
        "phase_deg": 308.39,
        "illuminated": 0.19,
        "dist_km": 382423.7,
    },
    "events": {
        "sunrise": 2461349.1280978546,
        "sunset": 2461349.534034541,
        "solar_noon": 2461349.3312837584,
        "moonrise": 2461349.9665496834,
        "moonset": 2461349.449763897,
        "moon_transit_next": 2461350.218275414,
        "lunar_day_min": 1484.0,
        "next_quarter": {"quarter": 0, "jd": 2461353.793623159},
    },
    "gravity": {"tide": {"dg": 8.0e-07, "moon": 6.1e-07, "sun": 1.9e-07}},
}


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def get(path: str, query: dict[str, Any], timeout: float) -> dict[str, Any]:
        calls.append((path, query))
        return {"/api/key": KEY, "/api/clocks": CLOCKS}[path]

    monkeypatch.setattr(sky, "_get", get)
    return calls


def test_parse_reads_decimal_and_hemisphere_coordinates_and_time() -> None:
    p = sky.parse("sky key at 48.43, -123.37 on 2026-11-04T22:00Z heading 250")
    assert (p["lat"], p["lon"], p["t"], p["heading"], p["missing"]) == (
        48.43,
        -123.37,
        "2026-11-04T22:00Z",
        250.0,
        [],
    )
    p = sky.parse("solar time at 48.43° N, 123.37° W")
    assert (p["lat"], p["lon"]) == (48.43, -123.37)
    p = sky.parse("lunar time for lat -33.9 lon 151.2 at 2026-11-04 03:00")
    assert (p["lat"], p["lon"], p["t"]) == (-33.9, 151.2, "2026-11-04T03:00")
    assert sky.parse("solar time in Victoria")["missing"] == ["lat", "lon"]


def test_a_date_is_not_read_as_coordinates() -> None:
    assert sky.parse("what was the solar time on 2026-11-04")["missing"] == ["lat", "lon"]


def test_sky_triggers() -> None:
    assert sky.wants_sky("What's the lunar time at 48.4, -123.4?") == "lunar time"
    assert sky.wants_sky("get a key for 48.43, -123.37") == "key"
    assert sky.wants_sky("what is the key finding of this paper?") is None
    assert sky.wants_sky("tell me about the sky at 48.4, -123.4", ("sky",)) == "sky"


def test_a_sky_word_alone_is_a_science_question() -> None:
    """A sky word runs the key only with a place or time; naming the tool always does."""
    q = "What does the Sun's position and gravity have to do with the plasmasphere?"
    assert sky.wants_sky(q) is None
    assert sky.mentions_sky(q) == "sun's position"
    assert sky.wants_sky("why is the sky red at sunset?") is None
    assert sky.wants_sky("sunset on 2026-11-05") == "sunset"
    assert sky.wants_sky("sky key for Victoria") == "sky key"
    assert sky.wants_sky("tell me about the sky", ("sky",)) is None


def test_triangles_close_and_the_dance_cancels_the_sun() -> None:
    tri = sky.triangles(KEY["sun"]["enu"], heading=250.0)
    assert abs(tri["res_elev"]) < 1e-4  # the stub's vector is rounded to 5 places
    assert abs(tri["az"]["res"]) < 1e-12
    assert abs(tri["dance"]["res"]) < 1e-12
    az = math.degrees(math.atan2(tri["az"]["sin"], tri["az"]["cos"])) % 360
    assert az == pytest.approx(211.675, abs=1e-3)
    assert tri["dance"]["deg"] == pytest.approx((250.0 - az) % 360, abs=1e-9)
    assert sky.triangles([0.0, 0.0, 1.0])["az"] is None


def test_reading_arranges_the_key_and_clocks(stub: list) -> None:
    r = sky.reading(48.43, -123.37, "2026-11-04T22:00:00Z", h=10.0, heading=90)
    assert [c[0] for c in stub] == ["/api/key", "/api/clocks"]
    assert stub[0][1]["t"] == "2026-11-04T22:00:00Z"
    assert stub[0][1]["h"] == 10.0
    assert r["key"]["hex"] == KEY["hex"]
    assert r["source"]["commit"] == "abc123"
    assert r["clocks"]["solar_time"] == "14:02:57"
    assert r["clocks"]["lunar_time"] == "17:22:21"
    assert r["events_utc"]["sunrise"].startswith("2026-11-04T15:04")
    assert r["triangles"]["dance"]["deg"] == pytest.approx((90 - 211.675) % 360, abs=1e-3)
    assert "Sky key 27b01144e2ea" in sky.summary(r)
    with pytest.raises(ValueError, match="lat must be"):
        sky.reading(91, 0)


def test_ask_runs_the_sky_key_without_retrieving(tmp_path: Path, stub: list) -> None:
    calls: list[tuple[str, Any]] = []
    out = asyncio.run(
        asker(tmp_path, calls).ask("What is the solar and lunar time at 48.43, -123.37 now?")
    )
    assert calls == []  # no passage retrieval for a sky request
    assert out["results"]["sky"]["reading"]["clocks"]["lunar_time"] == "17:22:21"
    assert "Sky key 27b01144e2ea" in out["relay"]
    assert "Sky key and clock times" in out["relay_full"]


def test_ask_sends_a_sky_word_without_a_place_to_the_papers(tmp_path: Path, stub: list) -> None:
    calls: list[tuple[str, Any]] = []
    q = "What does the Sun's position and gravity have to do with the plasmasphere?"
    out = asyncio.run(asker(tmp_path, calls).ask(q))
    assert calls  # passages were retrieved
    assert stub == []  # polarize-clock was not asked
    assert "sky" not in out["results"]
    assert out["results"]["sky_hint"] == {"matched": "sun's position"}
    assert 'say "sky key"' in out["relay"]
    assert "A sky key was asked for" not in out["relay"]


def test_ask_says_what_is_missing(tmp_path: Path, stub: list) -> None:
    out = asyncio.run(asker(tmp_path, []).ask("What is the sky key in Victoria?"))
    assert out["results"]["sky"]["missing"] == ["lat", "lon"]
    assert stub == []
    assert "decimal latitude and longitude" in out["relay"]


def test_ask_reports_polarize_clock_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def down(path: str, query: dict[str, Any], timeout: float) -> dict[str, Any]:
        raise OSError("polarize-clock is not reachable at http://127.0.0.1:8803: refused")

    monkeypatch.setattr(sky, "_get", down)
    out = asyncio.run(asker(tmp_path, []).ask("sky key at 48.43, -123.37"))
    assert "error" in out["results"]["sky"]
    assert "polarize-clock didn't answer" in out["relay"]


@pytest.mark.skipif(
    os.environ.get("POLARIZE_CLOCK_LIVE") != "1", reason="set POLARIZE_CLOCK_LIVE=1"
)
def test_live_polarize_clock() -> None:
    r = sky.reading(48.43, -123.37, "2026-11-04T22:00:00Z", h=10.0, heading=250)
    assert abs(r["triangles"]["res_elev"]) < 1e-12
    assert r["clocks"]["solar_time"]
    assert r["clocks"]["lunar_time"]
