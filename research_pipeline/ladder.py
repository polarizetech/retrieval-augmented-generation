"""Per-step provider ladders: each model step tries its providers in order (docs/hub-plan.md, 6).

A rung is skipped while its circuit breaker is open or its meter reads low, exhausted or down,
and a failed call moves the step to the next rung. The last rung is the free local model, which
is tried whatever its meter says. Every move down is written to the run's notes and ladder log,
so the run log and the relay name the provider each step used. Checking is not laddered: claim
checks call the local verifier by name, which goes to Ollama as before.
A request may pin a step, or the claim checks, to one provider (`providers`).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, TypeVar

from research_pipeline.client_llm import ClientLLM
from research_pipeline.config import Settings
from research_pipeline.llm import Endpoint, LLMError, ModelOutputError, Ollama
from research_pipeline.usage import Breakers, Meters, _read_key, load_config

T = TypeVar("T")
# Tries per call on a rung that has another below it: fall down rather than wait minutes.
RUNG_RETRIES = 3

# Model calls at once per provider, shared by every run in this process ([limits.providers]).
_SLOTS: dict[str, threading.BoundedSemaphore] = {}
_SLOTS_LOCK = threading.Lock()

# The model steps a request may give its own provider (research_start's `providers`); the first
# three are ladders, "verify" is the claim checks.
STEPS = ("planning", "extraction", "writing", "verify")
LADDER_STEPS = ("planning", "extraction", "writing")
# Two more steps choose where an ask's passage retrieval runs (not a research run), each with its
# own values, not [providers.*] names. modal and mac name the models router's upstream; droplet
# runs paper-fetch's in-process ONNX cross-encoder (rerank only: the library has no in-process
# embedding model); off skips reranking; llm reranks with the text model (research_mcp/
# llm_rerank.py).
RETRIEVAL_STEPS: dict[str, tuple[str, ...]] = {
    "embed": ("modal", "mac"),
    "rerank": ("modal", "mac", "droplet", "off", "llm"),
}
NEEDS_LADDER = (
    "providers: per-step providers need the hub's ladder: set PIPELINE_ENDPOINT_URL and "
    "PIPELINE_ENDPOINT_MODEL to one of config/hub.toml's [providers.*] (and leave "
    "PIPELINE_LADDER on), or use 'client'"
)


def _retrieval_choice(step: str, raw: Any) -> str:
    """A retrieval step's value, checked against RETRIEVAL_STEPS. Raises ValueError."""
    value = str(raw).strip().lower()
    if step == "embed" and value == "droplet":
        raise ValueError(
            "providers: embed = droplet is not available: the paper library has no in-process "
            "embedding model; use modal or mac"
        )
    allowed = RETRIEVAL_STEPS[step]
    if value not in allowed:
        raise ValueError(f"providers: {step} must be one of {', '.join(allowed)}, not {value!r}")
    return value


def check_providers(
    providers: Mapping[str, Any] | None, config: Mapping[str, Any]
) -> dict[str, str]:
    """A request's per-step providers, checked against the hub config: {step: provider}.

    A provider is a [providers.*] name or "client" (the calling assistant). "client" for
    planning, extraction or writing hands the whole run to the assistant, so those steps are
    all "client" or none is. The retrieval steps (RETRIEVAL_STEPS) take their own values.
    Raises ValueError with the reason."""
    if not providers:
        return {}
    if not isinstance(providers, Mapping):
        raise ValueError("providers must be an object: {step: provider}")
    known = sorted(str(n) for n in config.get("providers", {}))
    out: dict[str, str] = {}
    for step, raw in providers.items():
        if step in RETRIEVAL_STEPS:
            out[str(step)] = _retrieval_choice(str(step), raw)
            continue
        if step not in STEPS:
            every = ", ".join((*STEPS, *RETRIEVAL_STEPS))
            raise ValueError(f"providers: unknown step {step!r}; the steps are {every}")
        name = str(raw).strip()
        if name != "client" and name not in known:
            raise ValueError(
                f"providers: unknown provider {name!r} for {step}; use 'client' or one of "
                f"{', '.join(known) or '(none configured)'}"
            )
        out[str(step)] = name
    client = [s for s in LADDER_STEPS if out.get(s) == "client"]
    pinned = [s for s in LADDER_STEPS if s in out and out[s] != "client"]
    if client and pinned:
        raise ValueError(
            "providers: 'client' for planning, extraction or writing hands the whole run to the "
            f"calling assistant, so {', '.join(pinned)} cannot use another provider"
        )
    return out


def model_steps(chosen: Mapping[str, str]) -> dict[str, str]:
    """`chosen` without the retrieval steps: what a research run and its ladder take."""
    return {k: v for k, v in chosen.items() if k not in RETRIEVAL_STEPS}


def provider_slot(name: str, limit: int) -> threading.BoundedSemaphore:
    with _SLOTS_LOCK:
        if name not in _SLOTS:
            _SLOTS[name] = threading.BoundedSemaphore(max(1, limit))
        return _SLOTS[name]


def configured_provider(settings: Settings, config: dict[str, Any]) -> str | None:
    """The hub provider that matches the configured endpoint (same URL and model), if any."""
    for name, spec in config.get("providers", {}).items():
        url = str(spec.get("url", "")).rstrip("/")
        if url and url == settings.endpoint_url and spec.get("model") == settings.endpoint_model:
            return name
    return None


class Ladder(Endpoint):
    """An Endpoint whose calls climb down each step's ladder of providers."""

    def __init__(
        self,
        settings: Settings,
        config: dict[str, Any],
        *,
        meters: Meters | None = None,
        breakers: Breakers | None = None,
        rungs: dict[str, Endpoint | Ollama] | None = None,
        overrides: Mapping[str, str] | None = None,
        client_verifier: ClientLLM | None = None,
    ):
        super().__init__(settings)
        self.config = config
        self.providers: dict[str, dict[str, Any]] = config.get("providers", {})
        self.ladders: dict[str, list[str]] = {
            k: [str(n) for n in v] for k, v in config.get("ladders", {}).items()
        }
        self.steps: dict[str, str] = {str(k): str(v) for k, v in config.get("steps", {}).items()}
        self.top = configured_provider(settings, config) or next(iter(self.providers), "")
        self.meters = meters or Meters(config, settings)
        self.breakers = breakers or Breakers(config)
        self._rungs: dict[str, Endpoint | Ollama] = dict(rungs or {})
        for rung in self._rungs.values():
            rung.calls = self.calls
        # The models any rung writes with; anything else named in a call is the local verifier.
        self.text_models = {self.name} | {
            str(spec.get("model")) for spec in self.providers.values() if spec.get("model")
        }
        self.ladder_log: list[dict[str, Any]] = []
        self.limits: dict[str, int] = {
            str(k): int(v) for k, v in config.get("limits", {}).get("providers", {}).items()
        }
        # A request's own providers (check_providers): a pinned step uses that provider alone,
        # with no fallback. "verify" pins the claim checks; "client" there is client_verifier.
        self.overrides: dict[str, str] = {
            str(k): str(v) for k, v in (overrides or {}).items() if str(v) != "client"
        }
        self.client_verifier = client_verifier
        if client_verifier is not None:
            client_verifier.calls = self.calls  # one call log per run
        pinned = [f"{s} to {self.label(n)}" for s, n in self.overrides.items()]
        if client_verifier is not None:
            pinned.append("verify to the calling assistant")
        if pinned:
            self.notes.append(
                "This request pinned " + ", ".join(pinned) + "; a pinned step does not fall "
                "back to another provider."
            )

    # -- rungs ---------------------------------------------------------------------------------
    def rung(self, name: str) -> Endpoint | Ollama:
        if name not in self._rungs:
            spec = self.providers.get(name)
            if spec is None:
                raise LLMError(f"the ladder names an unknown provider {name!r}")
            if spec.get("kind") == "ollama":
                rung: Endpoint | Ollama = Ollama(replace(self.s, text_model=str(spec["model"])))
            else:
                key = _read_key(str(spec.get("key_file", ""))) or (
                    self.s.endpoint_key if name == self.top else ""
                )
                rung = Endpoint(
                    replace(
                        self.s,
                        endpoint_url=str(spec["url"]).rstrip("/"),
                        endpoint_model=str(spec["model"]),
                        endpoint_key=key,
                        endpoint_retries=min(self.s.endpoint_retries, RUNG_RETRIES),
                    )
                )
            rung.calls = self.calls
            self._rungs[name] = rung
        return self._rungs[name]

    def ladder_for(self, task: str) -> tuple[str, list[str]]:
        step = self.steps.get(task, task)
        if step in self.overrides:
            return step, [self.overrides[step]]
        return step, self.ladders.get(step) or [self.top]

    def label(self, name: str) -> str:
        return str(self.providers.get(name, {}).get("label", name))

    def resolve(self, model: str) -> tuple[str, str | None]:
        """The request's own verifier resolves where it runs; anything else as on Endpoint."""
        if self.client_verifier is not None and model == self.client_verifier.name:
            return self.client_verifier.resolve(model)
        pinned = self.overrides.get("verify", "")
        spec = self.providers.get(pinned, {})
        if pinned and model == spec.get("model"):
            if spec.get("kind") == "ollama":
                return self.rung(pinned).resolve(model)
            return model, None
        return super().resolve(model)

    def climb(self, task: str, call: Callable[[Endpoint | Ollama], T]) -> T:
        step, names = self.ladder_for(task)
        passed: list[tuple[str, str]] = []
        for i, name in enumerate(names):
            last = i == len(names) - 1
            breaker = self.breakers[f"provider:{name}"]
            if not breaker.allow():
                passed.append(
                    (
                        name,
                        f"skipped for {breaker.retry_after()} s more after "
                        f"{breaker.count} failures ({breaker.last_error})",
                    )
                )
                continue
            if not last:
                ok, why = self.meters.usable(name)
                if not ok:
                    passed.append((name, why))
                    continue
            try:
                if name in self.limits:
                    with provider_slot(name, self.limits[name]):
                        out = call(self.rung(name))
                else:
                    out = call(self.rung(name))
            except ModelOutputError:
                raise  # the model's answer was wrong, not the provider: the pipeline handles it
            except LLMError as exc:
                breaker.failure(str(exc))
                passed.append((name, f"failed: {str(exc)[:200]}"))
                continue
            breaker.success()
            if passed:
                self._record(step, task, name, passed)
            return out
        reasons = "; ".join(f"{self.label(n)}: {why}" for n, why in passed)
        raise LLMError(f"{step}: every provider on the ladder failed ({reasons})")

    def _record(self, step: str, task: str, used: str, passed: list[tuple[str, str]]) -> None:
        entry = {"step": step, "task": task, "used": used, "passed": dict(passed)}
        self.ladder_log.append(entry)
        note = (
            f"{step} ran on {self.label(used)} instead of "
            + "; ".join(f"{self.label(n)} ({why})" for n, why in passed)
            + (
                ". Its output is labelled: a weaker local model."
                if used == "local" or self.providers.get(used, {}).get("kind") == "ollama"
                else "."
            )
        )
        if note not in self.notes:
            self.notes.append(note)

    # -- the ChatModel interface ---------------------------------------------------------------
    def chat_json(
        self,
        task: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        if (
            task == "verify"
            and self.client_verifier is not None
            and model == self.client_verifier.name
        ):
            return self.client_verifier.chat_json(task, system, user, schema)
        if task == "verify" and "verify" in self.overrides:
            return self.climb(
                task,
                lambda rung: rung.chat_json(
                    task, system, user, schema, temperature=temperature, timeout=timeout
                ),
            )
        if model and model not in self.text_models:
            return self.local.chat_json(task, system, user, schema, model=model)
        return self.climb(
            task,
            lambda rung: rung.chat_json(
                task, system, user, schema, temperature=temperature, timeout=timeout
            ),
        )

    def chat_text(
        self, task: str, model: str, user: str, *, num_predict: int = 8, timeout: float = 300.0
    ) -> str:
        def call(rung: Endpoint | Ollama) -> str:
            name = rung.name if isinstance(rung, Endpoint) else rung.s.text_model
            return rung.chat_text(task, name, user, num_predict=num_predict, timeout=timeout)

        if task == "verify" and "verify" in self.overrides:
            return self.climb(task, call)
        if model not in self.text_models:
            return self.local.chat_text(task, model, user, num_predict=num_predict, timeout=timeout)
        return self.climb(task, call)


def make_ladder(
    settings: Settings,
    overrides: Mapping[str, str] | None = None,
    *,
    client_verifier: ClientLLM | None = None,
) -> Ladder | None:
    """A ladder when the configured endpoint is one of the hub's providers and it has ladders.
    `overrides` and `client_verifier` pin a request's own providers (check_providers)."""
    if not (settings.endpoint_url and settings.endpoint_model) or not settings.ladder:
        return None
    config = load_config()
    if not config.get("ladders") or configured_provider(settings, config) is None:
        return None
    return Ladder(settings, config, overrides=overrides, client_verifier=client_verifier)
