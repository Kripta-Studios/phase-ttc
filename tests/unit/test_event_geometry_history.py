"""Event-history constructor: TRAIN40 parity and causality on synthetic FCWD box tables."""

from __future__ import annotations

import numpy as np
import pytest

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from operational.event_geometry import history_inputs as hi
from operational.event_geometry.variants import VARIANTS, design
from operational.event_history_match.history import plan_timing, training_slot_plan
from operational.second_benchmark import fcwd
from operational.sota_eval.fcwd_inputs import _clip_published_box

STEP_US = 50_050


def _table(n: int = 40, *, saturate_from: int | None = None) -> fcwd.BoxTable:
    rows = []
    for i in range(n):
        s = 1.0 + 0.02 * i
        box = [640 - 60 * s, 360 - 40 * s, 640 + 60 * s, 360 + 40 * s]
        if saturate_from is not None and i >= saturate_from:
            box = [1, 1, 1281, 721]
        rows.append([100 + i, 1.0 + i * STEP_US / 1e6, *box])
    return fcwd.parse_boxes(np.asarray(rows), "FCWD_SYN")


def test_windows_are_three_contiguous_100ms_windows_ending_at_anchor() -> None:
    w = hi.context_windows(2_000_000)
    assert w.tolist() == [[1_700_000, 1_800_000], [1_800_000, 1_900_000], [1_900_000, 2_000_000]]


def test_fcwd_plan_has_the_seven_training_slots() -> None:
    table = _table()
    anchor = int(table.time_us[30])
    query = hi.fcwd_event_query(table, anchor, first_event_us=0)
    assert query.plan.count == hi.TRAINING_VALID_SLOTS
    assert query.plan.lags_us[query.plan.valid].tolist() == list(range(300_000, -1, -50_000))
    assert query.delta_s == pytest.approx(0.1)
    assert query.event_pixel_diff == 0.0


def test_fcwd_plan_equals_training_rule_with_exposure_offset() -> None:
    """With a TRAIN40-like 1.7 ms exposure offset the plain training rule gives the same slots."""
    table = _table()
    anchor = int(table.time_us[30])
    windows = hi.context_windows(anchor)
    training = training_slot_plan(windows, first_event_us=0, available_us=anchor + 1_700)
    shared = hi.fcwd_event_query(table, anchor, first_event_us=0, arrival_offset_us=1_700).plan
    primary = hi.fcwd_event_query(table, anchor, first_event_us=0).plan
    np.testing.assert_array_equal(training.valid, shared.valid)
    np.testing.assert_array_equal(training.valid, primary.valid)
    np.testing.assert_array_equal(plan_timing(training), plan_timing(shared))
    last = plan_timing(primary)[-1]
    np.testing.assert_allclose(last, [0.0, 0.0, 0.05, 0.0])
    np.testing.assert_allclose(plan_timing(shared)[-1], [0.0017, 0.0, 0.05, 0.0017], atol=1e-7)


def test_eap_and_fcwd_go_through_the_same_constructor() -> None:
    table = _table()
    anchor = int(table.time_us[30])
    previous, current = hi.fcwd_crop_rows(table, anchor)
    via_fcwd = hi.fcwd_event_query(table, anchor, first_event_us=0)
    via_shared = hi.event_query(
        hi.context_windows(anchor),
        hi.clip_fcwd_box(table.boxes[previous]),
        hi.clip_fcwd_box(table.boxes[current]),
        first_event_us=0,
        available_us=anchor,
        event_pixel_diff=0.0,
    )
    assert via_fcwd.square_xyxy == via_shared.square_xyxy
    np.testing.assert_array_equal(via_fcwd.plan.valid, via_shared.plan.valid)
    np.testing.assert_array_equal(plan_timing(via_fcwd.plan), plan_timing(via_shared.plan))


def test_crop_uses_current_and_100ms_earlier_box() -> None:
    table = _table()
    anchor = int(table.time_us[30])
    previous, current = hi.fcwd_crop_rows(table, anchor)
    assert (previous, current) == (28, 30)
    expected = common_square_from_boxes(
        [tuple(table.boxes[28]), tuple(table.boxes[30])], (0, 1), margin_fraction=0.25
    )
    assert hi.fcwd_event_query(table, anchor, first_event_us=0).square_xyxy == pytest.approx(
        expected
    )


def test_clip_matches_previous_fcwd_adapter() -> None:
    for box in ([1, 1, 1281, 721], [-5, 3, 400, 900], [10.5, 20.25, 30.0, 40.0]):
        assert list(hi.clip_fcwd_box(box)) == _clip_published_box(np.asarray(box, float))


def test_history_never_reads_after_anchor_or_before_budget() -> None:
    table = _table()
    anchor = int(table.time_us[30])
    query = hi.fcwd_event_query(table, anchor, first_event_us=0)
    intervals = query.windows_us[None] - query.plan.lags_us[:, None, None]
    used = intervals[query.plan.valid]
    assert used.max() <= anchor
    assert anchor - used.min() <= 650_000


def test_future_boxes_cannot_change_crop_or_geometry() -> None:
    table = _table()
    anchor = int(table.time_us[30])
    mutated = fcwd.BoxTable(table.sequence_id, table.frame_index, table.time_us, table.boxes.copy())
    mutated.boxes[31:] = [1, 1, 1281, 721]
    a = hi.fcwd_event_query(table, anchor, first_event_us=0)
    b = hi.fcwd_event_query(mutated, anchor, first_event_us=0)
    assert a.square_xyxy == b.square_xyxy
    ga = fcwd.query_geometry(table, np.asarray([anchor]), 0, 10).raw
    gb = fcwd.query_geometry(mutated, np.asarray([anchor]), 0, 10).raw
    np.testing.assert_array_equal(ga, gb)
    event10 = np.zeros((1, 10), np.float32)
    for name in ("EVT_GEO_BAND", "EVT_GEO_BAND_GA"):
        for left, right in zip(
            design(VARIANTS[name], event10, ga), design(VARIANTS[name], event10, gb), strict=True
        ):
            np.testing.assert_array_equal(left, right)


def test_saturated_box_gives_full_sensor_crop() -> None:
    table = _table(saturate_from=25)
    query = hi.fcwd_event_query(table, int(table.time_us[30]), first_event_us=0)
    x0, y0, x1, y1 = query.square_xyxy
    assert x0 < 0 and y0 < 0 and x1 > fcwd.SENSOR_WIDTH and y1 > fcwd.SENSOR_HEIGHT


def test_missing_raw_support_is_an_error() -> None:
    table = _table()
    anchor = int(table.time_us[30])
    with pytest.raises(ValueError):
        hi.fcwd_event_query(table, anchor, first_event_us=anchor - 100_000)


def test_crop_stats_count_only_events_inside_square_and_sensor() -> None:
    events = {
        "x": np.asarray([10, 50, 60, 1279, 2000]),
        "y": np.asarray([10, 50, 60, 700, 10]),
        "p": np.asarray([1, 0, 1, 1, 1]),
    }
    stats = hi.crop_event_stats(events, (40.0, 40.0, 80.0, 80.0), 100_000, 0.0)
    assert stats["events_in_crop"] == 2.0
    assert stats["positive_fraction"] == 0.5
    assert stats["square_inside_fraction"] == 1.0
