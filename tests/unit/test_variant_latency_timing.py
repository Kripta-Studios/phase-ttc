"""Stage clock arithmetic, synchronization hook and causal in-memory event slicing."""

from __future__ import annotations

import numpy as np
import pytest

from operational.variant_latency.timing import EventBuffer, StageClock, causal_window


def _fake_clock(values: list[int]):
    iterator = iter(values)
    return lambda: next(iterator)


def test_stage_clock_spans_accumulate_and_total_is_one_span() -> None:
    calls: list[str] = []
    clock = StageClock(
        sync=lambda: calls.append("sync"),
        clock=_fake_clock([0, 2_000_000, 5_000_000, 6_000_000, 10_000_000]),
    )
    clock.start()
    clock.mark("crop")
    clock.mark("model")
    clock.mark("crop")
    clock.mark("head")
    ms = clock.finish()
    assert ms == {"crop": 3.0, "model": 3.0, "head": 4.0, "total": 10.0}
    assert calls == ["sync"] * 5  # one synchronization before every stamp
    with pytest.raises(RuntimeError):
        clock.mark("late")


def test_stage_clock_requires_start() -> None:
    clock = StageClock()
    with pytest.raises(RuntimeError):
        clock.finish()


def test_event_buffer_slices_half_open_windows() -> None:
    t = np.asarray([10, 20, 20, 30, 40, 50], dtype=np.int64)
    buffer = EventBuffer(t, np.arange(6), np.arange(6) * 10)
    x, y = buffer.window(20, 40)
    assert x.tolist() == [1, 2, 3]
    assert y.tolist() == [10, 20, 30]
    assert buffer.window(41, 50)[0].size == 0
    assert buffer.nbytes == t.nbytes + 2 * np.arange(6).nbytes
    joined = EventBuffer.concatenate(
        [
            {"t": t[:3], "x": np.arange(3), "y": np.arange(3)},
            {"t": t[3:], "x": np.arange(3), "y": np.arange(3)},
        ]
    )
    assert joined.t.tolist() == t.tolist()
    assert EventBuffer.concatenate([]).t.size == 0
    with pytest.raises(ValueError, match="non-decreasing"):
        EventBuffer(t[::-1].copy(), np.arange(6), np.arange(6))
    with pytest.raises(ValueError, match="misaligned"):
        EventBuffer(t, np.arange(5), np.arange(6))


def test_causal_window_matches_window_cache_semantics() -> None:
    assert causal_window(1_234_567, 30_000, 0) == (1_204_000, 1_234_000)
    assert causal_window(1_234_000, 30_000, 0) == (1_204_000, 1_234_000)
    assert causal_window(40_999, 30_000, 20_000) is None
    assert causal_window(50_999, 30_000, 20_000) == (20_000, 50_000)
