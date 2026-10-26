"""The pipeline's model backends: schema-constrained JSON chat.

Ollama (local) uses the native API rather than the OpenAI-compatible surface because `format` (JSON
schema constrained decoding) and `options.num_ctx` are only reliable there. A small model that is
merely *asked* for JSON drifts; one that is *constrained* to a schema cannot.

Endpoint is any OpenAI-compatible chat API (OpenRouter, Fireworks, vLLM...): a hosted model that is
large enough to follow a schema it is asked for, checked against that schema on every answer. Only
the text model moves there; a configured local verifier (MiniCheck) still runs on Ollama.
"""

from __future__ import annotations

import http.client
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Protocol
from urllib.parse import urlparse

import jsonschema

from .config import Settings


class LLMError(RuntimeError):
    pass


class ModelOutputError(LLMError):
    """One call produced no usable answer (ran out of time or tokens, or broke its schema).

    Unlike an unreachable server or a client that stopped answering, this is about one call, so a
    stage may record it against that call and go on.
    """


class ChatModel(Protocol):
    """What the stages need from a model: Ollama, or the MCP client (client_llm.ClientLLM)."""

    calls: list[dict[str, Any]]

    def chat_json(
        self, task: str, system: str, user: str, schema: dict[str, Any], *, model: str | None = None
    ) -> dict[str, Any]: ...

    def chat_text(self, task: str, model: str, user: str) -> str: ...


class Ollama:
    backend = "ollama"

    def __init__(self, settings: Settings):
        self.s = settings
        self.calls: list[dict[str, Any]] = []  # per-run accounting for the reproducibility log

    def _post(self, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        req = urllib.request.Request(
            self.s.ollama_url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise LLMError(f"ollama {path} -> HTTP {exc.code}: {exc.read()[:300]!r}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"ollama unreachable at {self.s.ollama_url}: {exc}") from exc

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
        """One constrained call. `task` names the stage for the run log."""
        model = model or self.s.text_model
        body = {
            "model": model,
            "stream": True,
            "keep_alive": self.s.ollama_keep_alive,
            "format": schema,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_ctx": self.s.num_ctx,
                "num_predict": self.s.max_output_tokens,
                "seed": 7,
            },
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        started = time.time()
        try:
            text, value = self._stream_json(body, timeout)
        except LLMError as exc:
            # Models without a thinking mode reject the `think` flag; retry once without it.
            if "think" not in str(exc):
                raise
            body.pop("think")
            text, value = self._stream_json(body, timeout)
        self.calls.append(
            {
                "task": task,
                "model": model,
                "prompt_chars": len(system) + len(user),
                "output_chars": len(text),
                "seconds": round(time.time() - started, 2),
            }
        )
        if value is None:
            raise ModelOutputError(
                f"{task}: no complete JSON answer (non-JSON despite the schema, or cut off at the "
                f"{self.s.max_output_tokens}-token cap; {len(text)} chars): {text[:200]!r}"
            )
        return value

    def _stream_json(
        self, body: dict[str, Any], timeout: float
    ) -> tuple[str, dict[str, Any] | None]:
        """Read the stream and hang up as soon as the JSON value is complete.

        Under grammar-constrained decoding a small model can finish the object and keep emitting
        legal-but-pointless whitespace (observed: 227 tokens generated for a 10-token reply).
        Closing the connection stops generation. Whitespace that runs away *inside* the value
        cannot be cut short, so that case is abandoned and retried once with another seed.
        """
        text, value, runaway = self._stream_once(body, timeout)
        if runaway:
            retry = {**body, "options": {**body["options"], "seed": body["options"]["seed"] + 1}}
            text, value, _ = self._stream_once(retry, timeout)
        return text, value

    def _stream_once(
        self, body: dict[str, Any], timeout: float
    ) -> tuple[str, dict[str, Any] | None, bool]:
        """`timeout` on `urlopen`/the per-line read only bounds a single socket operation: a
        response that keeps streaming *some* bytes (even slow, even garbage) never trips it, so
        without a separate wall-clock deadline a runaway generation can run far past the caller's
        stated timeout (observed: 600s timeout, no return after 24+ minutes). `deadline` below is
        that missing wall-clock bound, checked every chunk regardless of read-level activity.
        """
        req = urllib.request.Request(
            self.s.ollama_url + "/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        text = ""
        deadline = time.monotonic() + timeout
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                for line in response:
                    if time.monotonic() > deadline:
                        raise ModelOutputError(
                            f"ollama /api/chat: exceeded wall-clock timeout of {timeout}s "
                            f"({len(text)} chars generated so far)"
                        )
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise LLMError(f"ollama: {chunk['error']}")
                    piece = chunk.get("message", {}).get("content", "")
                    text += piece
                    if len(text) - len(text.rstrip()) > 200:
                        return text, None, True
                    if "}" in piece or chunk.get("done"):
                        try:
                            return text, json.loads(text), False
                        except json.JSONDecodeError:
                            pass
                    if chunk.get("done"):
                        break
        except urllib.error.HTTPError as exc:
            raise LLMError(f"ollama /api/chat -> HTTP {exc.code}: {exc.read()[:300]!r}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"ollama unreachable at {self.s.ollama_url}: {exc}") from exc
        return text, None, False

    def chat_text(
        self, task: str, model: str, user: str, *, num_predict: int = 8, timeout: float = 300.0
    ) -> str:
        """Unconstrained short reply, for classifier models that answer in a fixed word."""
        started = time.time()
        out = self._post(
            "/api/chat",
            {
                "model": model,
                "stream": False,
                "keep_alive": self.s.ollama_keep_alive,
                "options": {
                    "temperature": 0.0,
                    "num_ctx": self.s.num_ctx,
                    "num_predict": num_predict,
                },
                "messages": [{"role": "user", "content": user}],
            },
            timeout,
        )
        text = out.get("message", {}).get("content", "").strip()
        self.calls.append(
            {
                "task": task,
                "model": model,
                "prompt_chars": len(user),
                "output_chars": len(text),
                "seconds": round(time.time() - started, 2),
            }
        )
        return text

    def resolve(self, model: str) -> tuple[str, str | None]:
        """Map a configured model name onto an installed tag; return (tag, digest).

        The digest identifies the exact weights, so a run can be reconstructed later. A configured
        alias such as `qwen3:4b-instruct-2507` is matched to its one installed quantisation
        (`...-q4_K_M`); an ambiguous or absent name is an error, not a silent substitution.
        """
        rows = {r["name"]: r.get("digest") for r in self._post_get("/api/tags").get("models", [])}
        for name in (model, f"{model}:latest"):
            if name in rows:
                return name, rows[name]
        close = [n for n in rows if n.startswith(model + "-")]
        if len(close) == 1:
            return close[0], rows[close[0]]
        hint = f"; candidates: {', '.join(close)}" if close else ""
        raise LLMError(f"model {model!r} is not installed in Ollama{hint}")

    def _post_get(self, path: str) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(self.s.ollama_url + path, timeout=10) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"ollama unreachable at {self.s.ollama_url}: {exc}") from exc


class EndpointHTTPError(LLMError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"endpoint -> HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


# Statuses worth another try: rate limits and an overloaded or restarting upstream.
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
SCHEMA_NOTE = (
    "\n\nAnswer with one JSON object and nothing else: no prose, no code fence. It must match "
    "this JSON schema:\n"
)


class Endpoint:
    """The text model on any OpenAI-compatible chat API ("your own endpoint").

    Calls naming another model (a local verifier such as MiniCheck) go to Ollama, so claim checking
    can stay on this machine while drafting runs on hosted open weights.
    """

    backend = "endpoint"

    def __init__(self, settings: Settings, local: Ollama | None = None):
        if not settings.endpoint_url or not settings.endpoint_model:
            raise LLMError(
                "the endpoint backend needs PIPELINE_ENDPOINT_URL and PIPELINE_ENDPOINT_MODEL "
                "(and usually PIPELINE_ENDPOINT_KEY or PIPELINE_ENDPOINT_KEY_FILE)"
            )
        self.s = settings
        self.name = settings.endpoint_model
        self.url = settings.endpoint_url
        self.concurrency = max(1, settings.endpoint_concurrency)
        host = urlparse(self.url).hostname or ""
        self.openrouter = host.endswith("openrouter.ai")
        self.nearai = host.endswith("near.ai")
        # Hosts other than OpenRouter route to one provider and do not name it in the response.
        self.host_label = "Near AI" if self.nearai else host
        self._prices: tuple[float, float, float] | None = None  # USD per token: in, out, cached
        self.calls: list[dict[str, Any]] = []
        self._local = local
        self._lock = threading.Lock()
        # What this endpoint turned out to accept; each is dropped once on a 400 naming it.
        self._json_mode = "json_schema"
        self._send_reasoning_off = settings.endpoint_reasoning == "off"
        self.notes: list[str] = []

    @property
    def local(self) -> Ollama:
        if self._local is None:
            self._local = Ollama(self.s)
            # One call log per run: local verifier calls belong in it too.
            self._local.calls = self.calls
        return self._local

    # -- the ChatModel interface -----------------------------------------------------------
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
        if model and model != self.name:
            return self.local.chat_json(task, system, user, schema, model=model)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system + SCHEMA_NOTE + json.dumps(schema)},
            {"role": "user", "content": user},
        ]
        problem = ""
        for _ in range(2):  # one corrected retry, as an MCP client gets
            text, finish = self._complete(task, messages, temperature, timeout, schema=schema)
            if finish == "length":
                raise ModelOutputError(
                    f"{task}: cut off at the {self.s.max_output_tokens}-token cap "
                    f"({len(text)} chars): {text[:200]!r}"
                )
            value, problem = parse_json(text, schema)
            if value is not None:
                return value
            messages += [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": f"That answer was rejected: {problem}. Reply again with only the "
                    "corrected JSON object.",
                },
            ]
        raise ModelOutputError(f"{task}: no valid JSON answer after a retry: {problem}")

    def chat_text(
        self, task: str, model: str, user: str, *, num_predict: int = 8, timeout: float = 300.0
    ) -> str:
        if model != self.name:
            return self.local.chat_text(task, model, user, num_predict=num_predict, timeout=timeout)
        text, _ = self._complete(
            task, [{"role": "user", "content": user}], 0.0, timeout, max_tokens=num_predict
        )
        return text.strip()

    def resolve(self, model: str) -> tuple[str, str | None]:
        """A hosted model has no digest to pin; each call logs the version the host served."""
        if model == self.name:
            return model, None
        return self.local.resolve(model)

    # -- HTTP --------------------------------------------------------------------------------
    def _complete(
        self,
        task: str,
        messages: list[dict[str, str]],
        temperature: float,
        timeout: float,
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, str]:
        started = time.time()
        while True:
            body = self._body(task, messages, temperature, schema, max_tokens)
            try:
                out = self._post(body, timeout)
                break
            except EndpointHTTPError as exc:
                if not self._relax(exc, body):
                    raise
        choice = (out.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = out.get("usage") or {}
        self.calls.append(
            {
                "task": task,
                "model": self.name,
                "served_model": out.get("model"),
                "provider": out.get("provider") or self.host_label,
                "prompt_chars": sum(len(m["content"]) for m in messages),
                "output_chars": len(text),
                "input_tokens": usage.get("prompt_tokens"),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                ),
                "cost": usage.get("cost") if "cost" in usage else self._cost(usage),
                "seconds": round(time.time() - started, 2),
            }
        )
        return text, choice.get("finish_reason") or ""

    def _body(
        self,
        task: str,
        messages: list[dict[str, str]],
        temperature: float,
        schema: dict[str, Any] | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.s.max_output_tokens,
            "seed": 7,
        }
        if schema is not None and self._json_mode == "json_schema":
            name = re.sub(r"[^A-Za-z0-9_-]", "_", task)[:64] or "answer"
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": False},
            }
        elif schema is not None and self._json_mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        if self._send_reasoning_off:
            # OpenRouter's field; also passed through by vLLM-style servers that ignore it.
            body["reasoning"] = {"effort": "none"}
        if self.openrouter:
            body["usage"] = {"include": True}
            provider: dict[str, Any] = {}
            if self.s.endpoint_zdr:
                provider["zdr"] = True
            if self.s.endpoint_sort:
                provider["sort"] = self.s.endpoint_sort
            if provider:
                body["provider"] = provider
        return body

    def _relax(self, exc: EndpointHTTPError, body: dict[str, Any]) -> bool:
        """Drop one optional field the endpoint rejected. False when there is nothing to drop."""
        if exc.status not in (400, 404, 422):
            return False
        detail = exc.detail.lower()
        with self._lock:
            if "reasoning" in body and ("reasoning" in detail or "effort" in detail):
                self._send_reasoning_off = False
                self.notes.append(
                    f"{self.name} rejected reasoning off ({exc.status}); it ran with its default "
                    "reasoning, which costs extra output tokens."
                )
                return True
            format_words = ("response_format", "json", "schema", "structured")
            if not any(w in detail for w in format_words):
                return False
            if "response_format" in body and self._json_mode == "json_schema":
                self._json_mode = "json_object"
                self.notes.append(
                    f"{self.name} rejected json_schema output ({exc.status}); answers were asked "
                    "for as JSON objects and checked against the schema afterwards."
                )
                return True
            if "response_format" in body and self._json_mode == "json_object":
                self._json_mode = "none"
                return True
        return False

    def _post(self, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.s.endpoint_key:
            headers["Authorization"] = f"Bearer {self.s.endpoint_key}"
        if self.openrouter:
            headers["X-Title"] = "research-pipeline"
        data = json.dumps(body).encode()
        attempts = max(1, self.s.endpoint_retries)
        for attempt in range(attempts):
            last = attempt == attempts - 1
            req = urllib.request.Request(self.url + "/chat/completions", data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    out = json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:500].decode(errors="replace")
                if exc.code in RETRY_STATUS and not last:
                    _backoff(attempt)
                    continue
                if exc.code == 402 or any(
                    w in detail.lower() for w in ("insufficient", "credit", "balance", "quota")
                ):
                    detail = f"out of credits at {self.host_label}? {detail}"
                raise EndpointHTTPError(exc.code, detail) from exc
            except (OSError, http.client.HTTPException) as exc:
                # A dropped connection or TLS handshake (URLError, SSLError, reset, cut-off body):
                # observed from Near AI on 2026-10-26 for 20-65% of requests, so retry patiently.
                if not last:
                    _backoff(attempt)
                    continue
                raise LLMError(
                    f"endpoint unreachable at {self.url} after {attempts} tries: {exc}"
                ) from exc
            # OpenRouter reports some upstream failures inside a 200 response.
            error = out.get("error")
            if error:
                raw_code = str(error.get("code", ""))
                code = int(raw_code) if raw_code.isdigit() else 500
                if code in RETRY_STATUS and not last:
                    _backoff(attempt)
                    continue
                raise EndpointHTTPError(code, str(error.get("message", error))[:500])
            return out
        raise LLMError("endpoint: retries exhausted")  # pragma: no cover

    # -- cost, for hosts that do not report it -------------------------------------------------
    def _cost(self, usage: dict[str, Any]) -> float | None:
        prices = self._price_table()
        if prices is None:
            return None
        per_in, per_out, per_cached = prices
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        fresh = (usage.get("prompt_tokens") or 0) - cached
        output = usage.get("completion_tokens") or 0
        return round(fresh * per_in + cached * per_cached + output * per_out, 9)

    def _price_table(self) -> tuple[float, float, float] | None:
        """USD per token from PIPELINE_ENDPOINT_PRICE, or Near AI's public model list."""
        if self._prices is not None:
            return self._prices
        configured = self.s.endpoint_price.strip()
        if configured:
            per_in, per_out = (float(x) / 1e6 for x in configured.split(","))
            self._prices = (per_in, per_out, per_in)
        elif self.nearai:
            self._prices = nearai_prices(self.url, self.name)
        return self._prices


def _backoff(attempt: int) -> None:
    """Wait 1, 2, 4, 8, 16, 30, 30... s, jittered so concurrent calls do not retry in step."""
    time.sleep(min(30.0, 2.0**attempt) * random.uniform(0.5, 1.5))  # noqa: S311


def nearai_prices(url: str, model: str) -> tuple[float, float, float] | None:
    """A model's per-token prices from Near AI's public catalogue; None if it cannot be read."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/model/list", timeout=10) as response:
            rows = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    for row in rows if isinstance(rows, list) else rows.get("models", []):
        if row.get("modelId") == model:
            per_in = _per_token(row.get("inputCostPerToken"))
            cached = _per_token(row.get("cacheReadCostPerToken")) or per_in
            return per_in, _per_token(row.get("outputCostPerToken")), cached
    return None


def _per_token(cost: dict[str, Any] | None) -> float:
    """Near AI writes prices as {"amount": 40, "scale": 9}: 40e-9 USD per token."""
    cost = cost or {}
    return float(cost.get("amount", 0)) / 10 ** int(cost.get("scale", 0))


def parse_json(text: str, schema: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """The JSON object in a reply, checked against `schema`; or (None, what was wrong)."""
    raw = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        try:
            value = json.loads(raw[start : end + 1]) if 0 <= start < end else None
        except json.JSONDecodeError:
            value = None
    if not isinstance(value, dict):
        return None, f"not a JSON object: {text[:120]!r}"
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError as exc:
        return None, f"does not match the schema: {exc.message}"
    return value, ""


BACKENDS = ("auto", "client", "endpoint", "ollama")


def resolve_backend(settings: Settings, requested: str, *, otherwise: str) -> str:
    """Who does a run's model steps. "auto" (or empty) is your configured endpoint when one is
    set, else `otherwise`: the calling model behind MCP, local Ollama on the command line."""
    choice = (requested or "auto").strip().lower()
    if choice not in BACKENDS:
        raise LLMError(f"unknown model backend {requested!r}: use one of {', '.join(BACKENDS)}")
    if choice == "auto":
        configured = bool(settings.endpoint_url and settings.endpoint_model)
        return "endpoint" if configured else otherwise
    return choice


def make_llm(settings: Settings, backend: str | None = None) -> Ollama | Endpoint:
    """The configured model backend: PIPELINE_LLM (CLI, benchmarks) or the one asked for."""
    backend = resolve_backend(settings, backend or settings.llm_backend, otherwise="ollama")
    if backend == "ollama":
        return Ollama(settings)
    if backend == "endpoint":
        return Endpoint(settings)
    raise LLMError(f"the {backend!r} backend needs an MCP client; use ollama or endpoint here")
