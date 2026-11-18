"""Synthetic checks of the causal box-expansion geometry (numpy only)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from operational.phase_geometry import geometry
from operational.phase_geometry.metrics import ttc_to_phase

STEP_US = 100_000


def _expanding_track(ttc_s: float, frames: int = 9, query_index: int = 6):
    """Boxes whose area-root, height and width grow exactly as exp(t / ttc)."""
    times = np.arange(frames, dtype=np.int64) * STEP_US + 1_000_000_000
    query_us = int(times[query_index])
    t = (times - query_us) * 1e-6
    w0, h0 = 120.0, 80.0
    w = w0 * np.exp(t / ttc_s)
    h = h0 * np.exp(t / ttc_s)
    x1, y1 = 500.0 - w / 2, 300.0 - h / 2
    boxes = np.stack((x1, y1, x1 + w - 1.0, y1 + h - 1.0), axis=1)
    return times, boxes, query_us


@pytest.mark.parametrize("ttc_s", [0.8, 2.5, 7.0, -4.0])
def test_slope_recovers_known_ttc(ttc_s):
    times, boxes, query_us = _expanding_track(ttc_s)
    out = geometry.window_features(times, boxes, query_us)
    assert out["n_frames"] == 7
    for name in ("area", "height", "width", "area_last2"):
        assert out[f"k_{name}"] == pytest.approx(1.0 / ttc_s, rel=1e-9)
        assert out[f"phi_{name}"] == pytest.approx(
            float(ttc_to_phase(np.array([ttc_s]))[0]), rel=1e-9
        )
    assert out["rms_residual"] == pytest.approx(0.0, abs=1e-12)
    assert out["span_s"] == pytest.approx(0.6)


def test_constant_velocity_approach_is_close_to_true_ttc():
    """Pinhole scaling s ~ 1/Z under constant closing speed: OLS slope ~ 1/TTC at mid-window."""
    times = np.arange(7, dtype=np.int64) * STEP_US
    query_us = int(times[-1])
    ttc_at_query = 3.0
    remaining = ttc_at_query - (times - query_us) * 1e-6
    size = 100.0 / remaining
    boxes = np.stack(
        (600 - size / 2, 300 - size / 2, 600 + size / 2 - 1, 300 + size / 2 - 1), axis=1
    )
    out = geometry.window_features(times, boxes, query_us)
    assert 1.0 / out["k_area"] == pytest.approx(ttc_at_query + 0.3, rel=0.02)
    assert 1.0 / out["k_area_last2"] == pytest.approx(ttc_at_query + 0.05, rel=0.02)


def test_future_and_stale_frames_are_ignored():
    times, boxes, query_us = _expanding_track(2.0, frames=12, query_index=8)
    base = geometry.window_features(times[:9], boxes[:9], query_us)
    corrupted = boxes.copy()
    corrupted[9:] *= 5.0
    corrupted[0] *= 0.1  # 800 ms before the query: outside the 600 ms window
    assert geometry.window_features(times, corrupted, query_us) == base


def test_track_pool_uses_only_rows_supplied_by_query_time():
    times, boxes, _ = _expanding_track(2.0, frames=10, query_index=0)
    row_us = times[1:]
    frame_us = [times[i - 1 : i + 1] for i in range(1, len(times))]
    row_boxes = [boxes[i - 1 : i + 1] for i in range(1, len(times))]
    pool = geometry.build_track_pool(row_us, frame_us, row_boxes)
    assert pool.disagreements == 0
    query_us = int(times[5])
    reference = geometry.query_features(pool, query_us)
    row_boxes_future = [b.copy() for b in row_boxes]
    for b in row_boxes_future[5:]:
        b[-1] *= 3.0  # later rows introduce later frames only
    altered = geometry.build_track_pool(row_us, frame_us, row_boxes_future)
    assert geometry.query_features(altered, query_us) == reference


def test_compute_table_separates_tracks_and_rows():
    times, boxes, _ = _expanding_track(1.5, frames=4, query_index=0)
    rows = 3
    frame_us = [times[i : i + 2] for i in range(rows)]
    grow = [boxes[i : i + 2] for i in range(rows)]
    shrink = [b[::-1].copy() for b in grow]
    table, disagreements = geometry.compute_table(
        ["S"] * (2 * rows),
        ["A"] * rows + ["B"] * rows,
        [int(t[-1]) for t in frame_us] * 2,
        frame_us + frame_us,
        grow + shrink,
    )
    k = table[:, geometry.RAW_FIELDS.index("k_area")]
    assert disagreements > 0  # track B reuses frame times with different boxes
    assert np.all(k[:rows] > 0)
    assert np.isclose(k[0], 1.0 / 1.5)


def test_fewer_than_two_frames_gives_infinite_ttc():
    out = geometry.window_features([5_000_000], [[10, 10, 50, 60]], 5_000_000)
    assert out["n_frames"] == 1
    assert out["k_area"] == 0.0
    assert out["phi_area"] == 0.0
    empty = geometry.window_features([], np.zeros((0, 4)), 5_000_000)
    assert empty["n_frames"] == 0


def test_truncation_flags_and_trunc_aware_choice():
    tx, ty = geometry.truncation_flags(
        np.array([[0, 10, 50, 60], [10, 10, 1279, 60], [10, 0, 50, 719], [5, 5, 9, 9]])
    )
    assert tx.tolist() == [True, True, False, False]
    assert ty.tolist() == [False, False, True, False]
    times = np.array([0, STEP_US, 2 * STEP_US])
    boxes = np.array([[0, 100, 100, 200], [0, 98, 120, 202], [0, 96, 140, 204]], dtype=float)
    out = geometry.window_features(times, boxes, 2 * STEP_US)
    assert out["trunc_x_current"] == 1.0 and out["trunc_y_current"] == 0.0
    assert out["phi_trunc_aware"] == out["phi_height"]


def test_slope_to_phase_respects_output_support():
    assert geometry.slope_to_phase(100.0) == pytest.approx(8.0)
    assert geometry.slope_to_phase(-100.0) == pytest.approx(-math.log(2.0) + 1e-6)
    assert geometry.slope_to_phase(0.0) == 0.0


def test_learned_inputs_shape_and_clip():
    raw = np.zeros((2, len(geometry.RAW_FIELDS)))
    raw[0, geometry.RAW_FIELDS.index("phi_area")] = 3.0
    g = geometry.learned_inputs(raw)
    assert g.shape == (2, len(geometry.G_FIELDS))
    assert g[0, 0] == pytest.approx(0.5 / 0.03)
