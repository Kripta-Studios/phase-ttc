"""Stage clock and in-memory event slicing used by the timed pipelines (pure Python/numpy)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


class StageClock:
    """Monotonic stage spans; an optional ``sync`` (e.g. CUDA synchronize) runs before each stamp.

    ``mark(name)`` closes the span opened by the previous stamp and adds it to ``name`` (spans of
    a repeated name accumulate).  ``finish()`` returns milliseconds per stage plus ``total``,
    the single span from ``start()`` to the last stamp.
    """

    def __init__(
        self,
        sync: Callable[[], None] | None = None,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._sync = sync
        self._clock = clock
        self._start: int | None = None
        self._last = 0
        self._spans: dict[str, int] = {}

    def _stamp(self) -> int:
        if self._sync is not None:
            self._sync()
        return self._clock()

    def start(self) -> None:
        """Open the end-to-end span."""
        self._spans = {}
        self._start = self._last = self._stamp()

    def mark(self, name: str) -> None:
        """Close the current stage span under ``name``."""
        if self._start is None:
            raise RuntimeError("StageClock.mark before start")
        now = self._stamp()
        self._spans[name] = self._spans.get(name, 0) + (now - self._last)
        self._last = now

    def finish(self) -> dict[str, float]:
        """Milliseconds per stage and the end-to-end ``total`` (last stamp minus start)."""
        if self._start is None:
            raise RuntimeError("StageClock.finish before start")
        out = {name: value / 1e6 for name, value in self._spans.items()}
        out["total"] = (self._last - self._start) / 1e6
        self._start = None
        return out


@dataclass
class EventBuffer:
    """Time-sorted events held in memory; ``window`` slices ``[start, end)`` by binary search."""

    t: NDArray[np.int64]
    x: NDArray[Any]
    y: NDArray[Any]
    nbytes: int = field(init=False)

    def __post_init__(self) -> None:
        if not (len(self.t) == len(self.x) == len(self.y)):
            raise ValueError("event arrays are misaligned")
        if len(self.t) > 1 and bool(np.any(np.diff(self.t) < 0)):
            raise ValueError("event timestamps must be non-decreasing")
        self.nbytes = int(self.t.nbytes + self.x.nbytes + self.y.nbytes)

    @classmethod
    def concatenate(cls, chunks: list[dict[str, NDArray[Any]]]) -> EventBuffer:
        """Join chunks given in time order (keys ``t``, ``x``, ``y``)."""
        if not chunks:
            empty = np.zeros(0, dtype=np.int64)
            return cls(empty, empty.copy(), empty.copy())
        return cls(
            np.concatenate([np.asarray(c["t"], dtype=np.int64) for c in chunks]),
            np.concatenate([np.asarray(c["x"]) for c in chunks]),
            np.concatenate([np.asarray(c["y"]) for c in chunks]),
        )

    def window(self, start_us: int, end_us: int) -> tuple[NDArray[Any], NDArray[Any]]:
        """Coordinates of the events with ``start_us <= t < end_us``."""
        lo = int(np.searchsorted(self.t, start_us, side="left"))
        hi = int(np.searchsorted(self.t, end_us, side="left"))
        return self.x[lo:hi], self.y[lo:hi]


def causal_window(end_us: int, tau_us: int, stream_start_us: int) -> tuple[int, int] | None:
    """``[key*1000 - tau, key*1000)`` with ``key = floor(end/1000)`` (``WindowCache`` semantics),
    or ``None`` when the window would start before the stream."""
    end = (int(end_us) // 1000) * 1000
    start = end - int(tau_us)
    return None if start < int(stream_start_us) else (start, end)


__all__ = ["EventBuffer", "StageClock", "causal_window"]
