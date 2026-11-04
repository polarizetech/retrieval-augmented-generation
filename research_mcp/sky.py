"""The sky key and clock times for a place and time, from a running sky-clock data server.

sky-clock (polarizetech/sky-clock) computes them; this module only asks it and arranges the answer.
No model runs here, and nothing is recalled: every number comes back from sky-clock with the version
and commit that produced it. Two routes are read:

  /api/key     the key/1 document: the Sun's direction and gravity at the place, binned, hashed
  /api/clocks  solar time, lunar time, mean solar time and the equation of time, the Sun's and
               Moon's altitude and azimuth, the Moon's phase, rise, set and transit times, and
               tidal gravity

`reading()` also gives the key's angles as unit triangles (the Sun split against gravity into a
vertical leg sin(el) and a horizontal leg cos(el); the azimuth as (cos, sin); with a heading, the
dance angle, heading x conj(azimuth)), each with its Pythagorean residual 1 - (leg^2 + leg^2).

`parse()` reads a place, time and heading out of a request in plain words, for `ask`. It takes
decimal or degree coordinates only: place names are not looked up (no geocoder is wired in).

The location goes only to the sky-clock server named by SKY_CLOCK_URL (default the local one). It is
not logged here.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8803"
USER_AGENT = "retrieval-augmented-generation (sky__key; https://github.com/polarizetech)"
#: Words that make a request a sky-key request (config/routing.default.toml [sky] adds to these).
TRIGGERS = (
    "sky key",
    "sun key",
    "sky clock",
    "sky-clock",
    "solar time",
    "lunar time",
    "moon time",
    "sun time",
    "sundial time",
    "moondial",
    "equation of time",
    "sun position",
    "sun's position",
    "position of the sun",
    "sun and gravity",
    "gravity reference",
    "dance angle",
    "moon phase",
    "sunrise",
    "sunset",
    "moonrise",
    "moonset",
    "solar noon",
)

_NUM = r"[+-]?\d{1,3}(?:\.\d+)?"
_LABELLED = re.compile(
    rf"\b(lat(?:itude)?|lon(?:gitude)?|lng)\s*[:=]?\s*({_NUM})\s*°?\s*([nsew])?\b", re.I
)
_PAIR = re.compile(
    rf"(?<![\w.])({_NUM})\s*°?\s*([ns])?\s*[, ]\s*({_NUM})\s*°?\s*([ew])?(?![\w.])", re.I
)
_TIME = re.compile(
    r"(?<![\w-])"
    r"([+-]?\d{4,6}-\d{1,2}-\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?Z?)"
    r"(?![\w:])"
)
_HEADING = re.compile(r"\b(?:heading|facing|bearing)\s*(?:of|is|=|:)?\s*(\d{1,3}(?:\.\d+)?)", re.I)
_HEIGHT = re.compile(
    r"\b(?:height|altitude|elevation)\s*(?:of|is|=|:)?\s*(-?\d+(?:\.\d+)?)\s*m\b", re.I
)


def base_url() -> str:
    return os.environ.get("SKY_CLOCK_URL", DEFAULT_URL).rstrip("/")


def wants_sky(question: str, extra: tuple[str, ...] = ()) -> str | None:
    """The trigger a request matched, or None. "key" alone is too common to count."""
    low = question.lower()
    for term in (*TRIGGERS, *extra):
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", low):
            return term
    if re.search(r"\bkey\b", low) and (_LABELLED.search(question) or _PAIR.search(question)):
        return "key"
    return None


def _signed(value: str, hemi: str | None) -> float:
    v = float(value)
    return -abs(v) if hemi and hemi.lower() in "sw" else v


def parse(question: str) -> dict[str, Any]:
    """Place, time, height and heading from a request. Missing pieces are listed in `missing`;
    a missing time means now (sky-clock's default)."""
    out: dict[str, Any] = {}
    for m in _LABELLED.finditer(question):
        which = "lat" if m.group(1).lower().startswith("lat") else "lon"
        out.setdefault(which, _signed(m.group(2), m.group(3)))
    if "lat" not in out or "lon" not in out:
        # Drop dates and times first so "2026-11-04" is not read as a coordinate pair.
        bare = _TIME.sub(" ", question)
        m = _PAIR.search(bare)
        if m:
            out.setdefault("lat", _signed(m.group(1), m.group(2)))
            out.setdefault("lon", _signed(m.group(3), m.group(4)))
    t = _TIME.search(question)
    if t:
        out["t"] = t.group(1).replace(" ", "T")
    h = _HEADING.search(question)
    if h:
        out["heading"] = float(h.group(1)) % 360.0
    z = _HEIGHT.search(question)
    if z:
        out["h"] = float(z.group(1))
    missing = [k for k in ("lat", "lon") if k not in out]
    if "lat" in out and not -90 <= out["lat"] <= 90:
        missing.append("lat (out of range)")
    if "lon" in out and not -180 <= out["lon"] <= 360:
        missing.append("lon (out of range)")
    out["missing"] = missing
    return out


def _get(path: str, query: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = f"{base_url()}{path}?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise ValueError(f"sky-clock refused {path}: {exc.code} {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise OSError(f"sky-clock is not reachable at {base_url()}: {exc}") from exc


def jd_iso(jd: float | None) -> str | None:
    """A Julian day as an ISO UTC time (to the second), or None outside years 1-9999."""
    if jd is None or not math.isfinite(jd):
        return None
    try:
        t = dt.datetime(1970, 1, 1, tzinfo=dt.UTC) + dt.timedelta(days=jd - 2440587.5)
    except OverflowError:
        return None
    return t.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def hours_hms(h: float | None) -> str | None:
    """Hours as hh:mm:ss on a 24-hour dial."""
    if h is None or not math.isfinite(h):
        return None
    s = round((h % 24.0) * 3600) % 86400
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def triangles(sun_enu: list[float], heading: float | None = None) -> dict[str, Any]:
    """The Sun against gravity and along the ground as unit triangles, each with its residual."""
    e, n, u = (float(x) for x in sun_enu)
    horiz = math.hypot(e, n)
    out: dict[str, Any] = {
        "sun_v": u,
        "sun_h": horiz,
        "res_elev": 1.0 - (u * u + horiz * horiz),
    }
    if horiz <= 1e-12:
        out["az"] = None
        return out
    ac, as_ = n / horiz, e / horiz
    out["az"] = {"cos": ac, "sin": as_, "res": 1.0 - (ac * ac + as_ * as_)}
    if heading is not None:
        r = math.radians(heading)
        hc, hs = math.cos(r), math.sin(r)
        dc, ds = hc * ac + hs * as_, hs * ac - hc * as_
        out["heading"] = {"deg": heading, "cos": hc, "sin": hs}
        out["dance"] = {
            "deg": math.degrees(math.atan2(ds, dc)) % 360.0,
            "cos": dc,
            "sin": ds,
            "res": 1.0 - (dc * dc + ds * ds),
        }
    return out


def reading(
    lat: float,
    lon: float,
    t: str | None = None,
    *,
    h: float = 0.0,
    heading: float | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """The key and the clock times at one place and time, from sky-clock, arranged for a reply."""
    if not -90 <= lat <= 90:
        raise ValueError("lat must be in -90..90")
    if not -180 <= lon <= 360:
        raise ValueError("lon must be in -180..360")
    q: dict[str, Any] = {"lat": lat, "lon": lon, "h": h}
    if t:
        q["t"] = t
    key = _get("/api/key", {**q, "scan": 0}, timeout)
    clocks = _get("/api/clocks", q, timeout)
    sun, moon, ev = clocks["sun"], clocks["moon"], clocks["events"]
    nq = ev.get("next_quarter") or {}
    tide = clocks["gravity"]["tide"]
    return {
        "source": {
            "tool": key.get("provider", {}).get("tool", "sky-clock"),
            "version": key.get("provider", {}).get("version"),
            "commit": key.get("provider", {}).get("commit"),
            "url": base_url(),
            "ephemeris": clocks["time"].get("ephemeris") or key.get("ephemeris"),
        },
        "place": {"lat": lat, "lon": lon, "h_m": h},
        "time": {
            "utc": key["time"]["utc_iso"],
            "jd_ut": key["time"]["jd_ut"],
            "deltaT_s": key["time"]["deltaT_s"],
            "ut1_error_s": clocks["time"].get("ut1_error_s"),
        },
        "key": {
            "hex": key["hex"],
            "schema": key.get("schema"),
            "bin_deg": key["quantisation"]["bin_deg"],
            "bins": key["bins"],
            "code_row": key["code"]["row"],
            "code_rule": key["code"]["rule"],
        },
        "sun": {
            "az_deg": key["sun"]["az_deg"],
            "el_deg": key["sun"]["el_deg"],
            "enu": key["sun"]["enu"],
        },
        "gravity": {"g_ms2": key["gravity"]["g_ms2"], "enu": key["gravity"]["enu"]},
        "triangles": triangles(key["sun"]["enu"], heading),
        "clocks": {
            "solar_time": hours_hms(sun["solar_time_h"]),
            "mean_solar_time": hours_hms(sun["lmst_h"]),
            "equation_of_time_min": sun["eot_min"],
            "lunar_time": hours_hms(moon["lunar_time_h"]),
            "solar_time_h": sun["solar_time_h"],
            "lunar_time_h": moon["lunar_time_h"],
        },
        "moon": {
            "az_deg": moon["az"],
            "el_deg": moon["el"],
            "phase_deg": moon["phase_deg"],
            "illuminated": moon["illuminated"],
            "dist_km": moon["dist_km"],
        },
        "events_utc": {
            "sunrise": jd_iso(ev.get("sunrise")),
            "solar_noon": jd_iso(ev.get("solar_noon")),
            "sunset": jd_iso(ev.get("sunset")),
            "moonrise": jd_iso(ev.get("moonrise")),
            "moonset": jd_iso(ev.get("moonset")),
            "moon_transit_next": jd_iso(ev.get("moon_transit_next")),
            "next_quarter": {"quarter": nq.get("quarter"), "utc": jd_iso(nq.get("jd"))}
            if nq
            else None,
            "lunar_day_min": ev.get("lunar_day_min"),
        },
        "tide": {"dg_ms2": tide["dg"], "moon_ms2": tide["moon"], "sun_ms2": tide["sun"]},
        "limits": [
            "Computed by sky-clock, an exploratory tool: data, not a finding.",
            "UT1 is taken as UTC from 1972 to 2030 (up to 0.9 s); Delta T is uncertain far from "
            "the present.",
            "Tidal constants and refraction are marked UNVERIFIED by sky-clock.",
        ],
    }


def summary(r: dict[str, Any]) -> str:
    """One line for a short reply."""
    s, c = r["sun"], r["clocks"]
    return (
        f"Sky key {r['key']['hex'][:12]}… at {r['place']['lat']:g}, {r['place']['lon']:g} "
        f"({r['time']['utc']} UTC): Sun {s['el_deg']:.2f}° up at {s['az_deg']:.2f}°; solar time "
        f"{c['solar_time']}, lunar time {c['lunar_time']} (sky-clock {r['source']['version']})."
    )
