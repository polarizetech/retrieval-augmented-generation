"""Offline tests for the Ollama client's streaming and timeout logic. No Ollama, no network."""

from __future__ import annotations

import json
import time
import unittest
from unittest import mock

from research_pipeline.config import Settings
from research_pipeline.llm import LLMError, Ollama


class FakeStream:
    """A fake `urlopen` response: an iterable of already-encoded NDJSON lines.

    `delay` simulates a server that keeps sending *some* bytes without ever finishing quickly
    enough — the case a per-read `timeout` cannot catch, since each individual read still
    succeeds within its own window.
    """

    def __init__(self, lines: list[dict], delay: float = 0.0):
        self.lines = lines
        self.delay = delay

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for line in self.lines:
            if self.delay:
                time.sleep(self.delay)
            yield json.dumps(line).encode()


class StreamDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.ollama = Ollama(Settings(ollama_url="http://fake", text_model="t", num_ctx=8192))

    def test_runaway_stream_hits_wall_clock_deadline(self):
        """A response that never closes the JSON object, but keeps streaming small chunks
        forever, must be aborted once the wall-clock timeout elapses -- not left to run until
        the per-read socket timeout (which never fires, since bytes keep arriving)."""

        # An endless generator of harmless filler chunks that never contains a closing "}".
        def never_ending():
            i = 0
            while True:
                yield {"message": {"content": "filler "}, "done": False}
                i += 1
                if i > 100000:  # safety valve so a test failure doesn't hang the suite forever
                    return

        with mock.patch(
            "urllib.request.urlopen", return_value=FakeStream(list(never_ending()), delay=0.01)
        ):
            started = time.monotonic()
            with self.assertRaises(LLMError) as ctx:
                self.ollama._stream_once({"model": "t", "options": {}}, timeout=0.3)
            elapsed = time.monotonic() - started
        self.assertIn("wall-clock", str(ctx.exception))
        # Bounded well under what an unpatched per-read timeout would allow (it would never fire).
        self.assertLess(elapsed, 5.0)

    def test_prompt_response_completes_normally(self):
        lines = [{"message": {"content": '{"a": 1}'}, "done": True}]
        with mock.patch("urllib.request.urlopen", return_value=FakeStream(lines)):
            _, value, runaway = self.ollama._stream_once({"model": "t", "options": {}}, timeout=5.0)
        self.assertFalse(runaway)
        self.assertEqual(value, {"a": 1})


if __name__ == "__main__":
    unittest.main()
