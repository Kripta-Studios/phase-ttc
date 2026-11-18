"""Synthetic encode tests: one crop for all slots, no event at/after the query is read."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from operational.event_history_match import engine  # noqa: E402
from operational.event_history_match.history import training_slot_plan  # noqa: E402

ANCHOR = 2_000_000
WINDOWS = [[ANCHOR - 300_000, ANCHOR - 200_000], [ANCHOR - 200_000, ANCHOR - 100_000],
           [ANCHOR - 100_000, ANCHOR]]  # fmt: skip
SQUARE = (100.0, 100.0, 228.0, 228.0)


class RecordingReader:
    """In-memory event stream that records every requested half-open interval."""

    def __init__(self, seed: int = 3) -> None:
        rng = np.random.default_rng(seed)
        count = 40_000
        self.t = np.sort(rng.integers(0, ANCHOR + 500_000, size=count)).astype(np.int64)
        self.x = rng.integers(0, 400, size=count).astype(np.int32)
        self.y = rng.integers(0, 400, size=count).astype(np.int32)
        self.p = rng.choice(np.array([-1, 1], dtype=np.int8), size=count)
        self.requests: list[tuple[int, int]] = []

    def iter_window_chunks(self, start_us: int, end_us: int, *, chunk_events: int = 250_000):
        self.requests.append((int(start_us), int(end_us)))
        lo, hi = np.searchsorted(self.t, [start_us, end_us], side="left")
        for cursor in range(lo, hi, chunk_events):
            stop = min(hi, cursor + chunk_events)
            yield {
                "x": self.x[cursor:stop],
                "y": self.y[cursor:stop],
                "t": self.t[cursor:stop],
                "p": self.p[cursor:stop],
            }


def _plan() -> Any:
    return training_slot_plan(WINDOWS, first_event_us=0, available_us=ANCHOR, max_lag_us=300_000)


def test_reader_never_asked_for_events_after_query() -> None:
    reader = RecordingReader()
    encoded = engine.encode_slots(
        reader, _plan(), WINDOWS, SQUARE, sequence_id="synthetic", event_pixel_diff=0.0
    )
    assert encoded.shape == (8, 3, 12, 128, 128)
    assert reader.requests, "encode must read the raw stream"
    assert max(end for _, end in reader.requests) <= ANCHOR
    assert min(start for start, _ in reader.requests) >= ANCHOR - 600_000
    assert torch.count_nonzero(encoded[0]) == 0  # padded 350 ms slot never encoded
    assert all(torch.count_nonzero(encoded[i]) > 0 for i in range(1, 8))


def test_future_events_do_not_change_the_encoding() -> None:
    reader = RecordingReader()
    baseline = engine.encode_slots(
        reader, _plan(), WINDOWS, SQUARE, sequence_id="synthetic", event_pixel_diff=0.0
    )
    future = reader.t >= ANCHOR
    reader.x[future] = 150
    reader.y[future] = 150
    altered = engine.encode_slots(
        reader, _plan(), WINDOWS, SQUARE, sequence_id="synthetic", event_pixel_diff=0.0
    )
    assert torch.equal(baseline, altered)


def test_every_slot_uses_the_current_query_square(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def spy(reader, windows, lags, valid, roi, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"windows": windows, "lags": lags, "valid": valid, "roi": roi, **kwargs})
        return torch.zeros(16, 3, 12, 128, 128)

    import e_jepa_ttc.efficient_context.mapped_union as mapped_union

    monkeypatch.setattr(mapped_union, "encode_union", spy)
    engine.encode_slots(
        RecordingReader(), _plan(), WINDOWS, SQUARE, sequence_id="s", event_pixel_diff=5
    )
    assert len(calls) == 1  # a single crop for the whole history
    call = calls[0]
    assert call["roi"] == SQUARE
    assert call["roi_size"] == 128 and call["event_pixel_diff"] == 5
    assert call["lags"].tolist() == list(range(750_000, -1, -50_000))
    assert call["valid"].tolist() == [False] * 9 + [True] * 7
    np.testing.assert_array_equal(call["windows"], np.asarray(WINDOWS))


def test_slot_encoding_equals_shifted_current_window_encoding() -> None:
    """A lagged slot equals the current-slot encoding of windows shifted by that lag."""
    reader = RecordingReader()
    encoded = engine.encode_slots(
        reader, _plan(), WINDOWS, SQUARE, sequence_id="synthetic", event_pixel_diff=0.0
    )
    shifted = (np.asarray(WINDOWS) - 150_000).tolist()
    plan = training_slot_plan(shifted, first_event_us=0, available_us=ANCHOR - 150_000)
    direct = engine.encode_slots(
        reader, plan, shifted, SQUARE, sequence_id="synthetic", event_pixel_diff=0.0
    )
    assert torch.equal(encoded[4], direct[-1])  # slot 4 = lag 150 ms
