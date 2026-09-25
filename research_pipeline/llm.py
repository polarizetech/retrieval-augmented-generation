"""Minimal Ollama client: schema-constrained JSON chat and embeddings.

Uses the native API rather than the OpenAI-compatible surface because `format` (JSON schema
constrained decoding) and `options.num_ctx` are only reliable there. A small model that is merely
*asked* for JSON drifts; one that is *constrained* to a schema cannot.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .config import Settings


class LLMError(RuntimeError):
    pass


class Ollama:
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
            "format": schema,
            "think": False,
            "options": {"temperature": temperature, "num_ctx": self.s.num_ctx, "seed": 7},
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
            raise LLMError(f"{task}: model returned non-JSON despite schema: {text[:200]!r}")
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
                        raise LLMError(
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

    def embed(self, texts: list[str], *, timeout: float = 300.0) -> list[list[float]]:
        if not texts:
            return []
        out = self._post("/api/embed", {"model": self.s.embedding_model, "input": texts}, timeout)
        vectors = out.get("embeddings") or []
        if len(vectors) != len(texts):
            raise LLMError(f"embed: asked for {len(texts)} vectors, got {len(vectors)}")
        return vectors

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
