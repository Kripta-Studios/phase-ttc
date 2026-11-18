"""Pure slot-plan construction for event histories (no I/O, no models).

The training rule is a literal re-statement of
``operational/final_refit/event_history.py::_query_plan`` and ``causal_history_mask``:
eight candidate slots at lags ``350, 300, ..., 0`` ms, all encoded in the current query's
crop, valid iff the shifted windows have raw support and the complete sensor input of the
slot fits the 650 ms causal budget measured from the availability time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SLOTS = 8
STORAGE_SLOTS = 16
TRAINING_LAG_STEP_US = 50_000
DELIVERED_EVTTC_LAGS_US = (300_000, 200_000, 100_000, 0)
BUDGET_US = 650_000


@dataclass(frozen=True)
class SlotPlan:
    """Eight-slot history plan; padded slots have ``valid=False`` and are never encoded."""

    lags_us: np.ndarray
    valid: np.ndarray
    anchors_us: np.ndarray
    arrivals_us: np.ndarray
    query_us: int
    span_us: int

    @property
    def count(self) -> int:
        return int(self.valid.sum())

    def storage_lags(self) -> np.ndarray:
        """H16 storage lags for ``encode_union``; the last eight slots carry this plan."""
        step = int(self.lags_us[-2] - self.lags_us[-1]) if len(self.lags_us) > 1 else 0
        if step <= 0 or np.any(np.diff(self.lags_us) != -step):
            raise ValueError("storage layout requires a uniform lag grid")
        return np.arange(STORAGE_SLOTS - 1, -1, -1, dtype=np.int64) * step

    def storage_valid(self) -> np.ndarray:
        return np.concatenate((np.zeros(STORAGE_SLOTS - SLOTS, dtype=np.bool_), self.valid))


def clock_features(anchors: Sequence[int], available: Sequence[int], query_us: int) -> np.ndarray:
    """Same four fields as ``operational.rgb_port.infer_experts.clock_features``."""
    values = np.zeros((len(anchors), 4), np.float32)
    for index, (anchor, arrival) in enumerate(zip(anchors, available, strict=True)):
        if anchor > query_us or arrival > query_us or arrival < anchor:
            raise ValueError("Expert observation is unavailable at the query")
        values[index] = [
            (query_us - anchor) / 1e6,
            (query_us - arrival) / 1e6,
            0.0 if index == 0 else (anchor - anchors[index - 1]) / 1e6,
            (arrival - anchor) / 1e6,
        ]
    return values


def _windows(windows_us: object) -> np.ndarray:
    windows = np.asarray(windows_us, dtype=np.int64)
    if (
        windows.shape != (3, 2)
        or np.any(windows[:, 1] <= windows[:, 0])
        or np.any(np.diff(windows[:, 1]) <= 0)
    ):
        raise ValueError("three increasing half-open context windows are required")
    return windows


def training_slot_plan(
    windows_us: object,
    *,
    first_event_us: int,
    available_us: int,
    query_us: int | None = None,
    max_lag_us: int | None = None,
    lag_step_us: int = TRAINING_LAG_STEP_US,
    budget_us: int = BUDGET_US,
    support_mask: np.ndarray | None = None,
) -> SlotPlan:
    """Training-rule plan: 50 ms lags, raw support, 650 ms budget, suffix-causal.

    ``max_lag_us`` additionally pads slots with a larger lag (EvTTC arm B pads the 350 ms
    slot because that slot never occurs in TRAIN40, see PROTOCOL section 2).
    ``support_mask`` ([8] bool) intersects a precomputed raw-support mask, e.g. the
    ``valid`` array stored with frozen Test12 H8 tensors.
    """
    windows = _windows(windows_us)
    anchor = int(windows[-1, 1])
    query = int(available_us if query_us is None else query_us)
    if available_us < anchor or query < available_us:
        raise ValueError("availability must follow the event anchor and precede the query")
    span = int(windows[-1, 1] - windows[0, 0])
    lags = np.arange(SLOTS - 1, -1, -1, dtype=np.int64) * int(lag_step_us)
    support = int(windows.min()) - lags >= int(first_event_us)
    if support_mask is not None:
        mask = np.asarray(support_mask, dtype=np.bool_)
        if mask.shape != (SLOTS,):
            raise ValueError("support mask must be [8] bool")
        support &= mask
    causal = int(available_us) - (anchor - lags) + span <= int(budget_us)
    valid = support & causal
    if max_lag_us is not None:
        valid &= lags <= int(max_lag_us)
    if not bool(valid[-1]):
        raise ValueError("current event observation lacks causal raw support")
    if np.any(valid[:-1] & ~valid[1:]):
        raise ValueError("event history selection is not suffix-causal")
    anchors = anchor - lags
    arrivals = np.full(SLOTS, int(available_us), dtype=np.int64)
    return SlotPlan(lags, valid, anchors, arrivals, query, span)


def delivered_evttc_plan(windows_us: object, *, query_us: int) -> SlotPlan:
    """The delivered EvTTC history: four slots at 100 ms cadence, arrivals at the query."""
    windows = _windows(windows_us)
    anchor = int(windows[-1, 1])
    if anchor != int(query_us):
        raise ValueError("delivered EvTTC queries are anchored at the last event endpoint")
    span = int(windows[-1, 1] - windows[0, 0])
    lags = np.arange(SLOTS - 1, -1, -1, dtype=np.int64) * 100_000
    valid = np.isin(lags, DELIVERED_EVTTC_LAGS_US)
    if anchor - (int(lags[valid].max()) + span) < anchor - BUDGET_US:
        raise ValueError("delivered history exceeds its 650 ms span")
    anchors = anchor - lags
    arrivals = np.full(SLOTS, int(query_us), dtype=np.int64)
    return SlotPlan(lags, valid, anchors, arrivals, int(query_us), span)


def plan_timing(plan: SlotPlan) -> np.ndarray:
    """[8,4] timing with zero padding, computed only over valid chronological slots."""
    timing = np.zeros((SLOTS, 4), dtype=np.float32)
    columns = np.flatnonzero(plan.valid)
    timing[columns] = clock_features(
        plan.anchors_us[columns].tolist(), plan.arrivals_us[columns].tolist(), plan.query_us
    )
    return timing


def plan_intervals(plan: SlotPlan, windows_us: object) -> np.ndarray:
    """[8,3,2] half-open event intervals actually read for each slot (padded slots zero)."""
    windows = _windows(windows_us)
    intervals = windows[None] - plan.lags_us[:, None, None]
    intervals[~plan.valid] = 0
    return intervals


def latest_event_time_read(plan: SlotPlan, windows_us: object) -> int:
    """Exclusive upper bound of every event timestamp the plan can read."""
    intervals = plan_intervals(plan, windows_us)
    return int(intervals[plan.valid][:, :, 1].max())


def earliest_event_time_read(plan: SlotPlan, windows_us: object) -> int:
    intervals = plan_intervals(plan, windows_us)
    return int(intervals[plan.valid][:, :, 0].min())


def fusion_event_fields(plan: SlotPlan) -> dict[str, float]:
    """Timing/valid fields as they appear at the fusion input (event fields 15..19)."""
    last = plan_timing(plan)[-1]
    return {
        "query_minus_anchor_s": float(last[0]),
        "query_minus_arrival_s": float(last[1]),
        "gap_s": float(last[2]),
        "arrival_minus_anchor_s": float(last[3]),
        "valid_fraction": plan.count / SLOTS,
    }
