"""Per-step provider ladders: each model step tries its providers in order (docs/hub-plan.md, 6).

A rung is skipped while its circuit breaker is open or its meter reads low, exhausted or down,
and a failed call moves the step to the next rung. The last rung is the free local model, which
is tried whatever its meter says. Every move down is written to the run's notes and ladder log,
so the run log and the relay name the provider each step used. Checking is not laddered: claim
checks call the local verifier by name, which goes to Ollama as before.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, TypeVar

from research_pipeline.config import Settings
from research_pipeline.llm import Endpoint, LLMError, ModelOutputError, Ollama
from research_pipeline.usage import Breakers, Meters, _read_key, load_config

T = TypeVar("T")
# Tries per call on a rung that has another below it: fall down rather than wait minutes.
RUNG_RETRIES = 3


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
        return step, self.ladders.get(step) or [self.top]

    def label(self, name: str) -> str:
        return str(self.providers.get(name, {}).get("label", name))

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
        if model not in self.text_models:
            return self.local.chat_text(task, model, user, num_predict=num_predict, timeout=timeout)

        def call(rung: Endpoint | Ollama) -> str:
            name = rung.name if isinstance(rung, Endpoint) else rung.s.text_model
            return rung.chat_text(task, name, user, num_predict=num_predict, timeout=timeout)

        return self.climb(task, call)


def make_ladder(settings: Settings) -> Ladder | None:
    """A ladder when the configured endpoint is one of the hub's providers and it has ladders."""
    if not (settings.endpoint_url and settings.endpoint_model) or not settings.ladder:
        return None
    config = load_config()
    if not config.get("ladders") or configured_provider(settings, config) is None:
        return None
    return Ladder(settings, config)
