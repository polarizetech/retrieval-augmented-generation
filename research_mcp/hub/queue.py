"""One limit and queue per scarce resource, and memory pressure (docs/hub-plan.md, section 2).

A lane lets `limit` requests in at once and queues the rest in arrival order, so a queued request
can say its position and an estimate. A heavy lane also waits while the machine is short of
memory. Nothing here makes a caller hang: `ask` answers at its deadline whatever the lane says.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

MEMORY_POLL = 2.0


def free_memory_percent() -> int | None:
    """Roughly how much memory is free, 0-100: macOS's memorystatus level, or Linux's
    MemAvailable. None when it cannot be read (then memory never holds a request)."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_level"],  # noqa: S607 - fixed argv, no shell
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
        info = dict(line.split(":", 1) for line in lines if ":" in line)
        total = int(info["MemTotal"].split()[0])
        return round(100 * int(info["MemAvailable"].split()[0]) / total)
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return None


class Lane:
    """`limit` at once, the rest waiting in arrival order."""

    def __init__(
        self,
        name: str,
        limit: int,
        *,
        heavy: bool = False,
        typical_seconds: float = 10.0,
        memory: Callable[[], int | None] = free_memory_percent,
        min_free: int = 10,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.name = name
        self.limit = max(1, limit)
        self.heavy = heavy
        self.memory = memory
        self.min_free = min_free
        self.clock = clock
        self.running: set[str] = set()
        self.waiting: deque[str] = deque()
        self._event: tuple[asyncio.AbstractEventLoop, asyncio.Event] | None = None
        # A running average of how long one holder keeps its slot, for estimates.
        self.typical = typical_seconds
        self.held_by_memory = False

    @property
    def _changed(self) -> asyncio.Event:
        """One event per running loop (tests run several loops over one gateway)."""
        loop = asyncio.get_running_loop()
        if self._event is None or self._event[0] is not loop:
            self._event = (loop, asyncio.Event())
        return self._event[1]

    def position(self, rid: str) -> int | None:
        """1 for the next to go in; None when `rid` is not waiting."""
        try:
            return self.waiting.index(rid) + 1
        except ValueError:
            return None

    def estimate(self, rid: str) -> float | None:
        pos = self.position(rid)
        if pos is None:
            return None
        return round(self.typical * pos / self.limit, 1)

    def _short_of_memory(self) -> bool:
        if not self.heavy:
            return False
        free = self.memory()
        return free is not None and free < self.min_free

    def _may_enter(self, rid: str) -> bool:
        return (
            len(self.running) < self.limit
            and bool(self.waiting)
            and self.waiting[0] == rid
            and not self._short_of_memory()
        )

    @asynccontextmanager
    async def slot(self, rid: str) -> AsyncIterator[None]:
        self.waiting.append(rid)
        try:
            while not self._may_enter(rid):
                self.held_by_memory = self._short_of_memory()
                self._changed.clear()
                # Memory frees without anyone releasing a slot, so a held lane looks again.
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._changed.wait(), MEMORY_POLL)
        except BaseException:
            self.waiting.remove(rid)
            self._changed.set()
            raise
        self.waiting.popleft()
        self.running.add(rid)
        self.held_by_memory = False
        started = self.clock()
        try:
            yield
        finally:
            self.running.discard(rid)
            self.typical = round(0.8 * self.typical + 0.2 * (self.clock() - started), 2)
            self._changed.set()

    def as_dict(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "running": len(self.running),
            "waiting": len(self.waiting),
            "typical_seconds": self.typical,
            "held_by_memory": self.held_by_memory,
        }


class Lanes:
    """The gateway's lanes, from the hub config's [limits] and [memory] tables."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        memory: Callable[[], int | None] = free_memory_percent,
        clock: Callable[[], float] = time.monotonic,
    ):
        limits = config.get("limits", {})
        min_free = int(config.get("memory", {}).get("min_free_percent", 10))
        self.retrieval = Lane(
            "retrieval",
            int(limits.get("retrieval", 2)),
            heavy=True,
            typical_seconds=10.0,
            memory=memory,
            min_free=min_free,
            clock=clock,
        )
        # Research runs are limited by the pipeline (PIPELINE_MAX_RUNS); requests it turns away
        # wait here, in order, and are started as runs finish.
        self.research: deque[str] = deque()
        self.memory = memory
        self.min_free = min_free

    def short_of_memory(self) -> bool:
        free = self.memory()
        return free is not None and free < self.min_free

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieval": self.retrieval.as_dict(),
            "research_waiting": len(self.research),
            "free_memory_percent": self.memory(),
        }
