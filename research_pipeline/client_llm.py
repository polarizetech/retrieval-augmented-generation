"""The MCP client's own model as the pipeline's model.

When the pipeline runs behind an MCP server, the model that is calling the server (Claude, ChatGPT,
a local chat model) can fill in the pipeline's forms instead of a local Ollama model. MCP clients
cannot be called back in the middle of a tool call in practice, so the exchange is turn-based:

    pipeline thread                          MCP tool handler (client's turn)
    ---------------                          --------------------------------
    chat_json(...) -> queue request, wait    take()   -> hand queued requests to the client
                                             answer() -> validate against the schema, release

Code still decides every step, and every answer is checked against its JSON schema before the
pipeline sees it. Only the model changes. Stages issue independent calls concurrently, so one client
turn answers a whole batch (every passage of an extraction pass, every claim of a verification).

Embeddings and retrieval belong to the paper library (paper-fetch), not to either model.
"""

from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from .llm import LLMError, Ollama

CLIENT_PREFIX = "mcp-client"
# Invalid answers allowed per task. A client that cannot produce a valid answer ends the run with
# an error rather than looping, and the pipeline never sees an answer that broke its schema.
MAX_ATTEMPTS = 3


@dataclass
class Request:
    id: str
    task: str
    system: str
    user: str
    schema: dict[str, Any]
    future: Future[dict[str, Any]] = field(default_factory=Future)
    issued: float = field(default_factory=time.time)
    handed_out: bool = False
    attempts: int = 0


class ClientLLM:
    """Model calls answered by the MCP client. Calls naming another model go to Ollama.

    `name` identifies the client in the run log. Nothing here can see which model the client runs,
    so the log records the client's self-reported name, not a model digest.
    """

    def __init__(self, ollama: Ollama, client_name: str = "", timeout: float = 1800.0):
        self.ollama = ollama
        self.name = f"{CLIENT_PREFIX}:{client_name or 'unknown'}"
        self.timeout = timeout
        self.calls: list[dict[str, Any]] = []
        self._pending: dict[str, Request] = {}
        self._ids = itertools.count(1)
        self._changed = threading.Condition()
        self.closed = False

    # -- the pipeline's side ---------------------------------------------------------------
    def chat_json(
        self,
        task: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        model: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if model and not model.startswith(CLIENT_PREFIX):
            return self.ollama.chat_json(task, system, user, schema, model=model)
        with self._changed:
            if self.closed:
                raise LLMError("the client stopped answering this run")
            request = Request(f"t{next(self._ids)}", task, system, user, schema)
            self._pending[request.id] = request
            self._changed.notify_all()
        try:
            value = request.future.result(timeout=self.timeout)
        except FutureTimeout as exc:
            raise LLMError(f"{task}: no answer from the MCP client in {self.timeout:.0f}s") from exc
        self.calls.append(
            {
                "task": task,
                "model": self.name,
                "prompt_chars": len(system) + len(user),
                "output_chars": len(str(value)),
                "seconds": round(time.time() - request.issued, 2),
            }
        )
        return value

    def chat_text(self, task: str, model: str, user: str, **kwargs: Any) -> str:
        """Classifier verifiers (e.g. MiniCheck) are local models by nature."""
        return self.ollama.chat_text(task, model, user, **kwargs)

    def resolve(self, model: str) -> tuple[str, str | None]:
        if model.startswith(CLIENT_PREFIX):
            return self.name, None
        return self.ollama.resolve(model)

    @property
    def s(self) -> Any:  # rerankers read the configured text model from here
        return self.ollama.s

    # -- the MCP handler's side ------------------------------------------------------------
    def take(self, wait: float, settle: float = 0.5, limit: int = 12) -> list[Request]:
        """Requests not yet handed to the client, waiting up to `wait` seconds for some.

        After the first request arrives, keep collecting until none has arrived for `settle`
        seconds, so a stage's concurrent calls are handed out as one batch. At most `limit` are
        handed out per turn, so a chat model is never asked for dozens of answers at once.
        """
        deadline = time.time() + wait
        with self._changed:
            while not self._fresh() and not self.closed:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return []
                self._changed.wait(remaining)
            while not self.closed:
                count = len(self._fresh())
                self._changed.wait(settle)
                if len(self._fresh()) == count or time.time() >= deadline:
                    break
            batch = self._fresh()[:limit]
            for request in batch:
                request.handed_out = True
            return batch

    def outstanding(self) -> list[Request]:
        """Requests handed out and not yet answered, e.g. to re-send after a lost reply."""
        with self._changed:
            return [r for r in self._pending.values() if r.handed_out]

    def answer(self, request_id: str, result: Any) -> str | None:
        """Deliver one answer. Returns an error to show the client, or None if it was accepted."""
        with self._changed:
            request = self._pending.get(request_id)
            if request is None:
                return f"unknown or already answered task_id {request_id!r}"
            try:
                jsonschema.validate(result, request.schema)
            except jsonschema.ValidationError as exc:
                request.attempts += 1
                problem = f"{request_id}: does not match output_schema: {exc.message}"
                if request.attempts < MAX_ATTEMPTS:
                    return problem
                del self._pending[request_id]
                self._changed.notify_all()
                request.future.set_exception(
                    LLMError(
                        f"{request.task}: {MAX_ATTEMPTS} answers broke the schema; last: "
                        f"{exc.message}"
                    )
                )
                return problem + f" ({MAX_ATTEMPTS} attempts; the run cannot continue)"
            del self._pending[request_id]
            self._changed.notify_all()
        request.future.set_result(result)
        return None

    def close(self, reason: str = "run ended") -> None:
        """Fail every waiting call, so a run the client abandoned does not hang forever."""
        with self._changed:
            self.closed = True
            pending, self._pending = list(self._pending.values()), {}
            self._changed.notify_all()
        for request in pending:
            if not request.future.done():
                request.future.set_exception(LLMError(reason))

    def _fresh(self) -> list[Request]:
        return [r for r in self._pending.values() if not r.handed_out]
