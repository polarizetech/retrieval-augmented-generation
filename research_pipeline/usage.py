"""Hub configuration, circuit breakers and provider usage meters (docs/hub-plan.md, phase 1).

Shared by the gateway (health, `hub__usage`, the relay's usage line) and the pipeline (provider
ladders), so both read the same thresholds from config/hub.default.toml plus config/hub.toml.
Key values are read from their files and sent only to their own provider; nothing here prints them.
"""

from __future__ import annotations

import json
import os
import threading
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_pipeline.config import ROOT, Settings

DEFAULT_CONFIG = ROOT / "config" / "hub.default.toml"
LOCAL_CONFIG = ROOT / "config" / "hub.toml"
HTTP_TIMEOUT = 10.0

# A meter reader: (url, headers) -> parsed JSON. Swapped out in tests.
Fetch = Callable[[str, dict[str, str]], Any]


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(default: Path | None = None, local: Path | None = LOCAL_CONFIG) -> dict[str, Any]:
    """The default hub config with the operator's own file merged over it, table by table.

    HUB_CONFIG names a whole config to use instead of both (the tests use it to stay offline)."""
    if default is None:
        override = os.environ.get("HUB_CONFIG", "").strip()
        default = Path(override) if override else DEFAULT_CONFIG
        local = None if override else local
    data = tomllib.loads(default.read_text()) if default.is_file() else {}
    if local is not None and local.is_file():
        data = _merge(data, tomllib.loads(local.read_text()))
    return data


# -- circuit breakers ---------------------------------------------------------------------------
@dataclass
class Breaker:
    """Closed until `failures` in a row; then open (skipped) for `cooldown` s; then one probe."""

    failures: int = 3
    cooldown: float = 120.0
    clock: Callable[[], float] = time.monotonic
    count: int = 0
    opened_at: float | None = None
    last_error: str = ""

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half-open" if self.clock() - self.opened_at >= self.cooldown else "open"

    def allow(self) -> bool:
        """Whether to try the component now: always when closed, once per cool-down when open."""
        return self.state != "open"

    def retry_after(self) -> int:
        if self.opened_at is None:
            return 0
        return max(0, round(self.cooldown - (self.clock() - self.opened_at)))

    def success(self) -> None:
        self.count, self.opened_at, self.last_error = 0, None, ""

    def failure(self, error: str) -> None:
        self.count += 1
        self.last_error = error[:300]
        if self.count >= self.failures:
            # A failed probe restarts the cool-down.
            self.opened_at = self.clock()

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "failures": self.count,
            "last_error": self.last_error,
            "retry_after": self.retry_after(),
        }


class Breakers:
    """One breaker per component name, all with the configured limits."""

    def __init__(self, config: dict[str, Any], clock: Callable[[], float] = time.monotonic):
        cfg = config.get("breaker", {})
        self.failures = int(cfg.get("failures", 3))
        self.cooldown = float(cfg.get("cooldown_seconds", 120))
        self.clock = clock
        self._all: dict[str, Breaker] = {}
        self._lock = threading.Lock()

    def __getitem__(self, name: str) -> Breaker:
        with self._lock:
            if name not in self._all:
                self._all[name] = Breaker(self.failures, self.cooldown, self.clock)
            return self._all[name]

    def items(self) -> list[tuple[str, Breaker]]:
        with self._lock:
            return sorted(self._all.items())


# -- usage meters -------------------------------------------------------------------------------
def _read_key(path: str) -> str:
    p = Path(path).expanduser()
    return p.read_text().strip() if p.is_file() else ""


def _read_env_file(path: str) -> dict[str, str]:
    p = Path(path).expanduser()
    if not p.is_file():
        return {}
    out = {}
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def http_json(url: str, headers: dict[str, str]) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "research-rag-hub", **headers})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as response:
        return json.load(response)


@dataclass
class Reading:
    """One provider's meter at one moment. `state`: ok, low, exhausted, down, unreadable."""

    provider: str
    label: str
    state: str
    readable: bool = True
    balance_usd: float | None = None
    used_fraction: float | None = None
    usage_usd: dict[str, float] = field(default_factory=dict)  # day, week, month
    burn_per_day: float | None = None
    days_left: float | None = None
    detail: str = ""
    loaded: list[str] = field(default_factory=list)
    read_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [], {}, "")}

    def short(self) -> str:
        """A few words for the relay's usage line."""
        name = self.provider_name()
        if not self.readable:
            return f"{name}: {self.detail or 'not readable'}"
        if self.state == "down":
            return f"{name}: down"
        if self.balance_usd is None:
            parts = [f"{name}: {self.state}"]
        else:
            parts = [f"{name}: ${self.balance_usd:.2f} left"]
        if self.burn_per_day:
            parts.append(f"${self.burn_per_day:.2f}/day")
        if self.days_left is not None:
            parts.append(f"~{self.days_left:.0f} days")
        if self.state in ("low", "exhausted") and self.balance_usd is not None:
            parts.append(self.state.upper())
        if self.loaded:
            parts.append(f"loaded: {', '.join(self.loaded)}")
        return ", ".join(parts)

    def provider_name(self) -> str:
        return {"openrouter": "OpenRouter", "nearai": "NEAR AI", "local": "Ollama"}.get(
            self.provider, self.provider
        )


def _low_state(balance: float | None, used: float | None, spec: dict[str, Any]) -> str:
    if balance is not None and balance <= 0:
        return "exhausted"
    if used is not None and used >= 1:
        return "exhausted"
    if balance is not None and balance <= float(spec.get("min_usd", 1.0)):
        return "low"
    if used is not None and used >= float(spec.get("max_used_fraction", 0.95)):
        return "low"
    return "ok"


def read_openrouter(name: str, spec: dict[str, Any], fetch: Fetch) -> Reading:
    """`GET /key` (usage today, this week, this month; key limit) and `GET /credits` (balance)."""
    label = str(spec.get("label", name))
    key = _read_key(str(spec.get("key_file", "")))
    if not key:
        return Reading(name, label, "unreadable", readable=False, detail="no key file")
    base = str(spec.get("url", "https://openrouter.ai/api/v1")).rstrip("/")
    headers = {"Authorization": f"Bearer {key}"}
    try:
        info = (fetch(f"{base}/key", headers) or {}).get("data", {})
        credits = (fetch(f"{base}/credits", headers) or {}).get("data", {})
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return Reading(name, label, "down", detail=f"meter unreachable: {type(exc).__name__}")
    total, spent = credits.get("total_credits"), credits.get("total_usage")
    balance = float(total) - float(spent) if total is not None and spent is not None else None
    limit, remaining = info.get("limit"), info.get("limit_remaining")
    used = None
    if limit:
        used = 1 - float(remaining or 0) / float(limit)
        if balance is None or float(remaining or 0) < balance:
            balance = float(remaining or 0)
    fields = {"day": "usage_daily", "week": "usage_weekly", "month": "usage_monthly"}
    usage = {k: round(float(info[f]), 4) for k, f in fields.items() if info.get(f) is not None}
    burn = round(usage["week"] / 7, 4) if "week" in usage else None
    days = round(balance / burn, 1) if balance is not None and burn else None
    return Reading(
        name,
        label,
        _low_state(balance, used, spec),
        balance_usd=None if balance is None else round(balance, 4),
        used_fraction=None if used is None else round(used, 4),
        usage_usd=usage,
        burn_per_day=burn,
        days_left=days,
    )


def read_nearai(name: str, spec: dict[str, Any], fetch: Fetch) -> Reading:
    """The organisation balance, which needs a reporting token; the API key is refused (401)."""
    label = str(spec.get("label", name))
    env = _read_env_file(str(spec.get("reporting_file", "")))
    org, token = env.get("NEARAI_ORG_ID", ""), env.get("NEARAI_REPORTING_TOKEN", "")
    if not (org and token):
        return Reading(name, label, "unreadable", readable=False, detail="needs a reporting token")
    base = str(spec.get("url", "https://cloud-api.near.ai/v1")).rstrip("/")
    try:
        body = fetch(
            f"{base}/organizations/{org}/usage/balance", {"Authorization": f"Bearer {token}"}
        )
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return Reading(name, label, "down", detail=f"meter unreachable: {type(exc).__name__}")
    data = body.get("data", body) if isinstance(body, dict) else {}
    balance = next(
        (float(data[k]) for k in ("balance_usd", "balance", "remaining", "amount") if k in data),
        None,
    )
    if balance is None:
        return Reading(name, label, "unreadable", readable=False, detail="balance not understood")
    return Reading(name, label, _low_state(balance, None, spec), balance_usd=round(balance, 4))


def read_ollama(name: str, spec: dict[str, Any], fetch: Fetch, url: str) -> Reading:
    """Free; whether Ollama answers, whether the model is installed, and what is loaded."""
    label = str(spec.get("label", name))
    try:
        tags = fetch(f"{url}/api/tags", {})
        loaded = fetch(f"{url}/api/ps", {})
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return Reading(name, label, "down", detail=f"Ollama unreachable: {type(exc).__name__}")
    names = {str(m.get("name", "")) for m in tags.get("models", [])}
    model = str(spec.get("model", ""))
    if model and model not in names and f"{model}:latest" not in names:
        return Reading(name, label, "down", detail=f"{model} is not installed")
    running = [str(m.get("name", "")) for m in loaded.get("models", [])]
    return Reading(name, label, "ok", detail="free", loaded=running)


class Meters:
    """Readings per provider, cached for `ttl_seconds`; plain notes for the plans it cannot read."""

    PLANS = (
        "Claude plan: only a Claude session can read it (the app's usage reading)",
        "ChatGPT plan: no usage API for a consumer plan; not readable",
    )

    def __init__(
        self,
        config: dict[str, Any],
        settings: Settings | None = None,
        fetch: Fetch = http_json,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.providers: dict[str, dict[str, Any]] = config.get("providers", {})
        self.ttl = float(config.get("usage", {}).get("ttl_seconds", 300))
        self.ollama_url = (settings or Settings()).ollama_url
        self.fetch = fetch
        self.clock = clock
        self._cache: dict[str, tuple[float, Reading]] = {}
        self._lock = threading.Lock()

    def read(self, name: str, *, fresh: bool = False) -> Reading:
        with self._lock:
            hit = self._cache.get(name)
        if hit and not fresh and self.clock() - hit[0] < self.ttl:
            return hit[1]
        spec = self.providers.get(name, {})
        if spec.get("kind") == "ollama":
            reading = read_ollama(name, spec, self.fetch, self.ollama_url)
        elif "openrouter.ai" in str(spec.get("url", "")):
            reading = read_openrouter(name, spec, self.fetch)
        elif "near.ai" in str(spec.get("url", "")):
            reading = read_nearai(name, spec, self.fetch)
        else:
            reading = Reading(name, str(spec.get("label", name)), "unreadable", readable=False)
        with self._lock:
            self._cache[name] = (self.clock(), reading)
        return reading

    def cached(self, name: str) -> Reading | None:
        with self._lock:
            hit = self._cache.get(name)
        return hit[1] if hit else None

    def all(self, *, fresh: bool = False) -> list[Reading]:
        return [self.read(name, fresh=fresh) for name in self.providers]

    def usable(self, name: str) -> tuple[bool, str]:
        """Whether a rung may be used: not low, exhausted or down. Unreadable meters pass."""
        reading = self.read(name)
        if reading.state in ("low", "exhausted", "down"):
            why = reading.detail or (
                f"${reading.balance_usd:.2f} left" if reading.balance_usd is not None else ""
            )
            return False, f"{reading.provider_name()} is {reading.state}" + (
                f" ({why})" if why else ""
            )
        return True, ""

    def line(self) -> str:
        """One line for the relay, from cached readings only (never waits on a provider)."""
        parts = [r.short() for name in self.providers if (r := self.cached(name)) is not None]
        return "Usage: " + (" · ".join(parts) if parts else "not read yet (see hub__usage)")
