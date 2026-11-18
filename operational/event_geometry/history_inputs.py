"""One event-history constructor for eAP TRAIN40 rows and FCWD queries (pure, no models).

Both datasets go through :func:`event_query`: three contiguous ~100 ms context windows ending
at the event anchor, the TRAIN40 slot rule (``history.training_slot_plan``: 50 ms lags, raw
support, 650 ms budget) restricted to lags <= 300 ms so that exactly the seven TRAIN40 slots
can be valid, and one crop square shared by every slot (current query's square).  The only
dataset-specific inputs are the windows' source, the two boxes of the square, the event pixel
offset and the availability time (PROTOCOL section 4.4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from operational.event_history_match.history import (
    BUDGET_US,
    SlotPlan,
    earliest_event_time_read,
    latest_event_time_read,
    training_slot_plan,
)
from operational.second_benchmark import fcwd

WINDOW_US = 100_000
MAX_LAG_US = 300_000
TRAINING_VALID_SLOTS = 7
FIRST_DELTA_US = 100_000
FIRST_DELTA_TOLERANCE_US = 25_000
SQUARE_MARGIN = 0.25
FCWD_EVENT_PIXEL_DIFF = 0.0
EAP_EVENT_PIXEL_DIFF = 5.0
DELTA_S = 0.1
EVENT_BLOCK_FIELDS = slice(160, 170)


@dataclass(frozen=True)
class EventQuery:
    """Everything the sealed event chain needs for one query."""

    windows_us: NDArray[np.int64]
    square_xyxy: tuple[float, float, float, float]
    plan: SlotPlan
    delta_s: float
    event_pixel_diff: float


def context_windows(anchor_us: int) -> NDArray[np.int64]:
    """``[a-300, a-200), [a-200, a-100), [a-100, a)`` ms as a ``[3, 2]`` array."""
    a = int(anchor_us)
    return np.asarray(
        [
            [a - 3 * WINDOW_US, a - 2 * WINDOW_US],
            [a - 2 * WINDOW_US, a - WINDOW_US],
            [a - WINDOW_US, a],
        ],
        dtype=np.int64,
    )


def crop_square(
    previous_box: Sequence[float], current_box: Sequence[float]
) -> tuple[float, float, float, float]:
    """TRAIN40 crop: common square of the -100 ms and current boxes, margin 0.25."""
    return common_square_from_boxes(
        [tuple(map(float, previous_box)), tuple(map(float, current_box))],
        (0, 1),
        margin_fraction=SQUARE_MARGIN,
    )


def event_query(
    windows_us: object,
    previous_box: Sequence[float],
    current_box: Sequence[float],
    *,
    first_event_us: int,
    available_us: int,
    event_pixel_diff: float,
    delta_s: float = DELTA_S,
) -> EventQuery:
    """Plan, square and causality checks shared by eAP and FCWD."""
    windows = np.asarray(windows_us, dtype=np.int64)
    plan = training_slot_plan(
        windows,
        first_event_us=int(first_event_us),
        available_us=int(available_us),
        max_lag_us=MAX_LAG_US,
    )
    query = plan.query_us
    if latest_event_time_read(plan, windows) > int(windows[-1, 1]):
        raise AssertionError("event history would read past the event anchor")
    if query - earliest_event_time_read(plan, windows) > BUDGET_US:
        raise AssertionError("event history exceeds the 650 ms causal budget")
    return EventQuery(
        windows, crop_square(previous_box, current_box), plan, float(delta_s), event_pixel_diff
    )


def clip_fcwd_box(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Published event-space box clipped to the 1280x720 sensor (``sota_eval.fcwd_inputs``)."""
    box = np.asarray(values, dtype=np.float64).reshape(4)
    if not np.isfinite(box).all():
        raise ValueError("FCWD box must be finite")
    x0, y0, x1, y1 = (
        float(np.clip(box[0], 0, fcwd.SENSOR_WIDTH)),
        float(np.clip(box[1], 0, fcwd.SENSOR_HEIGHT)),
        float(np.clip(box[2], 0, fcwd.SENSOR_WIDTH)),
        float(np.clip(box[3], 0, fcwd.SENSOR_HEIGHT)),
    )
    if x1 <= x0 or y1 <= y0:
        raise ValueError("clipped FCWD box is degenerate")
    return x0, y0, x1, y1


def fcwd_crop_rows(table: fcwd.BoxTable, anchor_us: int) -> tuple[int, int]:
    """Rows of the current and the ~100 ms earlier box (10 Hz parity, availability P)."""
    rows = fcwd.select_frames(table, int(anchor_us), 0, 10)
    if len(rows) < 2:
        raise ValueError("FCWD query lacks two causal boxes at 10 Hz")
    previous, current = int(rows[-2]), int(rows[-1])
    if int(table.time_us[current]) > int(anchor_us):
        raise AssertionError("current box is later than the anchor")
    gap = int(table.time_us[current] - table.time_us[previous])
    if abs(gap - FIRST_DELTA_US) > FIRST_DELTA_TOLERANCE_US:
        raise ValueError(f"FCWD -100 ms box is {gap} us before the current box")
    return previous, current


def fcwd_event_query(
    table: fcwd.BoxTable,
    anchor_us: int,
    *,
    first_event_us: int,
    arrival_offset_us: int = 0,
) -> EventQuery:
    """FCWD query: windows end at the anchor, boxes P, ``available = anchor + offset``."""
    previous, current = fcwd_crop_rows(table, anchor_us)
    return event_query(
        context_windows(anchor_us),
        clip_fcwd_box(table.boxes[previous]),
        clip_fcwd_box(table.boxes[current]),
        first_event_us=first_event_us,
        available_us=int(anchor_us) + int(arrival_offset_us),
        event_pixel_diff=FCWD_EVENT_PIXEL_DIFF,
    )


def event10_from_block(block: NDArray[Any]) -> NDArray[np.float32]:
    """PHASE20 event fields 10-19 = frozen E_CTX state coordinates 160:170."""
    values = np.asarray(block, dtype=np.float32)
    if values.shape[-1] != 170:
        raise ValueError("E_CTX frozen state must have 170 coordinates")
    return np.ascontiguousarray(values[..., EVENT_BLOCK_FIELDS])


def crop_event_stats(
    events: dict[str, NDArray[Any]],
    square: Sequence[float],
    duration_us: int,
    event_pixel_diff: float,
    width: int = fcwd.SENSOR_WIDTH,
    height: int = fcwd.SENSOR_HEIGHT,
) -> dict[str, float]:
    """Encoder-input sanity statistics of one raw window inside the crop square."""
    x = np.asarray(events["x"], dtype=np.float64) + float(event_pixel_diff)
    y = np.asarray(events["y"], dtype=np.float64)
    p = np.asarray(events["p"])
    x0, y0, x1, y1 = (float(v) for v in square)
    keep = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    keep &= (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    inside_w = max(0.0, min(x1, width) - max(x0, 0.0))
    inside_h = max(0.0, min(y1, height) - max(y0, 0.0))
    area = (x1 - x0) * (y1 - y0)
    n = int(keep.sum())
    seconds = duration_us * 1e-6
    return {
        "events_in_crop": float(n),
        "crop_rate_mev_s": n / seconds / 1e6,
        "sensor_rate_mev_s": len(x) / seconds / 1e6,
        "positive_fraction": float((p[keep] > 0).mean()) if n else float("nan"),
        "square_edge_px": x1 - x0,
        "square_inside_fraction": inside_w * inside_h / area if area > 0 else float("nan"),
        "events_per_crop_pixel": n / max(inside_w * inside_h, 1.0),
    }


__all__ = [
    "DELTA_S",
    "EAP_EVENT_PIXEL_DIFF",
    "FCWD_EVENT_PIXEL_DIFF",
    "MAX_LAG_US",
    "TRAINING_VALID_SLOTS",
    "EventQuery",
    "clip_fcwd_box",
    "context_windows",
    "crop_event_stats",
    "crop_square",
    "event10_from_block",
    "event_query",
    "fcwd_crop_rows",
    "fcwd_event_query",
]
