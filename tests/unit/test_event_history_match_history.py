"""Synthetic tests for the training-matched event-history slot plans."""

from __future__ import annotations

import numpy as np
import pytest

from operational.event_history_match.history import (
    delivered_evttc_plan,
    earliest_event_time_read,
    fusion_event_fields,
    latest_event_time_read,
    plan_intervals,
    plan_timing,
    training_slot_plan,
)

ANCHOR = 10_000_000


def _windows(anchor: int = ANCHOR, step: int = 100_000) -> list[list[int]]:
    return [[anchor + offset - step, anchor + offset] for offset in (-2 * step, -step, 0)]


def test_training_rule_with_positive_offset_has_seven_50ms_slots() -> None:
    plan = training_slot_plan(_windows(), first_event_us=0, available_us=ANCHOR + 5_000)
    assert plan.lags_us.tolist() == list(range(350_000, -1, -50_000))
    assert plan.valid.tolist() == [False] + [True] * 7
    fields = fusion_event_fields(plan)
    assert fields["valid_fraction"] == pytest.approx(7 / 8)
    assert fields["gap_s"] == pytest.approx(0.05)
    assert fields["query_minus_anchor_s"] == pytest.approx(0.005)
    assert fields["arrival_minus_anchor_s"] == pytest.approx(0.005)
    assert fields["query_minus_arrival_s"] == 0.0


def test_zero_offset_literal_rule_admits_eight_and_arm_b_pads_lag_350() -> None:
    literal = training_slot_plan(_windows(), first_event_us=0, available_us=ANCHOR)
    assert literal.count == 8
    matched = training_slot_plan(
        _windows(), first_event_us=0, available_us=ANCHOR, max_lag_us=300_000
    )
    assert matched.valid.tolist() == [False] + [True] * 7
    fields = fusion_event_fields(matched)
    assert fields["valid_fraction"] == pytest.approx(0.875)
    assert fields["gap_s"] == pytest.approx(0.05)


def test_delivered_plan_reproduces_off_support_fields() -> None:
    plan = delivered_evttc_plan(_windows(), query_us=ANCHOR)
    assert plan.lags_us[plan.valid].tolist() == [300_000, 200_000, 100_000, 0]
    assert plan.storage_valid().tolist() == [False] * 12 + [True] * 4
    fields = fusion_event_fields(plan)
    assert fields["gap_s"] == pytest.approx(0.1)
    assert fields["valid_fraction"] == pytest.approx(0.5)


def test_raw_support_masks_leading_slots_suffix_causally() -> None:
    windows = _windows()
    first_event = int(np.min(windows)) - 120_000  # only lags <= 100 ms have raw support
    plan = training_slot_plan(windows, first_event_us=first_event, available_us=ANCHOR + 1_000)
    assert plan.lags_us[plan.valid].tolist() == [100_000, 50_000, 0]
    timing = plan_timing(plan)
    assert np.all(timing[~plan.valid] == 0)
    first_valid = int(np.flatnonzero(plan.valid)[0])
    assert timing[first_valid, 2] == 0.0  # no gap before the first valid slot


def test_no_slot_reads_events_after_the_query_and_budget_holds() -> None:
    for available in (ANCHOR, ANCHOR + 1, ANCHOR + 19_992):
        for max_lag in (None, 300_000):
            plan = training_slot_plan(
                _windows(), first_event_us=0, available_us=available, max_lag_us=max_lag
            )
            assert latest_event_time_read(plan, _windows()) <= ANCHOR <= plan.query_us
            assert plan.query_us - earliest_event_time_read(plan, _windows()) <= 650_000
            intervals = plan_intervals(plan, _windows())[plan.valid]
            assert np.all(intervals[:, :, 1] > intervals[:, :, 0])


def test_longer_span_reduces_valid_count_like_training_cold_rows() -> None:
    windows = _windows(step=150_000)  # 450 ms span
    plan = training_slot_plan(windows, first_event_us=0, available_us=ANCHOR + 2_000)
    assert plan.lags_us[plan.valid].tolist() == [150_000, 100_000, 50_000, 0]


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        training_slot_plan(_windows(), first_event_us=0, available_us=ANCHOR - 1)
    with pytest.raises(ValueError):
        training_slot_plan(_windows(), first_event_us=ANCHOR, available_us=ANCHOR)
    with pytest.raises(ValueError):
        delivered_evttc_plan(_windows(), query_us=ANCHOR + 1)
