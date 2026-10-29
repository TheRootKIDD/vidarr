"""A hand-rolled discrete-event loop (CLAUDE.md: SimPy or hand-rolled; no new dependency).

Events are (time, sequence number, callback). The sequence number breaks ties in scheduling
order, so a run is a deterministic function of its inputs and seed.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from itertools import count


class Engine:
    def __init__(self) -> None:
        self.now = 0.0
        self._queue: list[tuple[float, int, Callable[[], None]]] = []
        self._seq = count()
        self.events = 0

    def at(self, t: float, fn: Callable[[], None]) -> None:
        """Run `fn` at absolute time `t` (never in the past)."""
        if t < self.now:
            raise ValueError(f"event scheduled in the past: {t} < {self.now}")
        heapq.heappush(self._queue, (t, next(self._seq), fn))

    def after(self, dt: float, fn: Callable[[], None]) -> None:
        self.at(self.now + dt, fn)

    def run(self, done: Callable[[], bool]) -> None:
        """Process events until `done()`; an empty queue before that is a deadlock (02 §5.5)."""
        while not done():
            if not self._queue:
                raise RuntimeError(f"no events left at t = {self.now:.6g} s: the system deadlocked")
            self.now, _, fn = heapq.heappop(self._queue)
            self.events += 1
            fn()
