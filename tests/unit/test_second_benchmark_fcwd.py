"""FCWD geometry arms: causality, frame selection, estimator exactness and scoring helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from operational.phase_geometry import geometry
from operational.second_benchmark import fcwd, run

STEP_US = 50_050


def _approach_table(n: int = 60, z0: float = 40.0, v: float = 8.0) -> fcwd.BoxTable:
    """Constant-velocity frontal approach: box side proportional to 1/Z, 20 Hz."""
    times = 5_000_000 + STEP_US * np.arange(n)
    z = z0 - v * (times - times[0]) * 1e-6
    side = 4000.0 / z
    cx, cy = 640.0, 360.0
    # geometry.py measures w = x2 - x1 + 1, so x2 = x1 + side - 1 makes the measured side exact.
    x1, y1 = cx - side / 2, cy - side / 2
    boxes = np.stack([x1, y1, x1 + side - 1.0, y1 + side - 1.0], axis=1)
    values = np.column_stack([100 + np.arange(n), times / 1e6, boxes])
    return fcwd.parse_boxes(values, "SYN")


def _depth(table: fcwd.BoxTable, t_us: int, z0: float = 40.0, v: float = 8.0) -> float:
    return z0 - v * (t_us - int(table.time_us[0])) * 1e-6


@pytest.mark.parametrize("latency", [0, 50_000])
@pytest.mark.parametrize("rate", [20, 10])
def test_selected_frames_are_causal_and_future_boxes_do_not_matter(latency: int, rate: int) -> None:
    table = _approach_table()
    anchor = int(table.time_us[30])
    rows = fcwd.select_frames(table, anchor, latency, rate)
    assert (table.time_us[rows] <= anchor - latency).all()
    assert (table.time_us[rows] >= anchor - fcwd.WINDOW_US).all()
    before = fcwd.raw_features(table, rows, anchor)
    ratio_before = fcwd.size_ratio_last2(table, rows)
    perturbed = table.boxes.copy()
    perturbed[31:] = perturbed[31:] * 3.0 + 17.0
    changed = fcwd.BoxTable(table.sequence_id, table.frame_index, table.time_us, perturbed)
    rows2 = fcwd.select_frames(changed, anchor, latency, rate)
    assert np.array_equal(rows, rows2)
    assert fcwd.raw_features(changed, rows2, anchor) == before
    assert fcwd.size_ratio_last2(changed, rows2) == ratio_before


def test_window_gives_ead_like_frame_counts() -> None:
    table = _approach_table()
    anchor = int(table.time_us[40])
    rows20 = fcwd.select_frames(table, anchor, 0, 20)
    rows10 = fcwd.select_frames(table, anchor, 0, 10)
    assert len(rows20) == 13 and len(rows10) == 7
    assert np.all(np.diff(table.frame_index[rows10]) == 2)
    raw = fcwd.raw_features(table, rows10, anchor)
    assert raw["span_s"] == pytest.approx(0.6006)
    g = geometry.learned_inputs(np.array([[raw[name] for name in geometry.RAW_FIELDS]]))
    assert g[0, geometry.G_FIELDS.index("n_frames")] == pytest.approx(1.0)


def test_size_ratio_last2_is_exact_for_constant_velocity() -> None:
    table = _approach_table()
    anchor = int(table.time_us[20])
    rows = fcwd.select_frames(table, anchor, 0, 20)
    assert fcwd.size_ratio_last2(table, rows) == pytest.approx(
        _depth(table, anchor) / 8.0, rel=1e-6
    )


def test_geo_area_is_close_to_true_ttc_and_capped_when_static() -> None:
    table = _approach_table()
    anchor = int(table.time_us[40])
    q = fcwd.query_geometry(table, np.array([anchor]), 0, 20)
    arms = fcwd.fixed_arms(q)
    truth = _depth(table, anchor) / 8.0
    # OLS slope of log-size estimates 1/T at the window centre: T_hat ~ T_now + ~0.3 s.
    assert truth < arms["GEO_AREA"][0] < truth + 0.4
    static = fcwd.parse_boxes(
        np.column_stack(
            [np.arange(20), 1 + 0.05 * np.arange(20), np.tile([1, 1, 1281, 721], (20, 1))]
        ),
        "STATIC",
    )
    qs = fcwd.query_geometry(static, static.time_us[-1:], 0, 20)
    assert fcwd.fixed_arms(qs)["GEO_AREA"][0] == pytest.approx(fcwd.TTC_CAP_S)
    assert fcwd.fixed_arms(qs)["SIZE_RATIO_LAST2"][0] == fcwd.TTC_CAP_S
    assert bool(qs.both_truncated_current[0])


def test_fcwd_truncation_uses_published_border_convention() -> None:
    tx, ty = fcwd.truncation_fcwd(
        np.array([[1, 50, 100, 200], [10, 1, 1281, 300], [5, 5, 900, 700]])
    )
    assert tx.tolist() == [True, True, False]
    assert ty.tolist() == [False, True, False]


def test_cohort_filter_requires_history_and_causal_boxes() -> None:
    table = _approach_table(n=40)
    start = int(table.time_us[0]) - 100_000
    anchors = fcwd.cohort_anchors(table, start, int(table.time_us[-1]) + 1)
    assert anchors.min() - fcwd.HISTORY_US >= start
    assert anchors.max() <= int(table.time_us[-1])
    assert np.isin(anchors, table.time_us).all()


def test_learned_design_matches_variant_design() -> None:
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "as_tensor"):
        pytest.skip("torch namespace stub without the runtime")
    from operational.phase_geometry.variants import VARIANTS, design

    table = _approach_table()
    q = fcwd.query_geometry(table, table.time_us[15:25], 0, 10)
    x, anchor = fcwd.learned_design(q.raw)
    arrays = {
        "features": np.zeros((len(x), 20), np.float32),
        "geometry_g": geometry.learned_inputs(q.raw),
        "geometry_raw": q.raw,
        "rgb_width_logits": np.zeros((len(x), 2), np.float32),
    }
    xd, ad, wd = design(VARIANTS["GEO_LEARNED"], arrays, slice(None))
    np.testing.assert_array_equal(x, xd)
    np.testing.assert_array_equal(anchor, ad)
    assert not wd.any()


def test_rte_extrapolation_and_sequence_bootstrap() -> None:
    assert run.rte(np.array([3.0, -1.0]), np.array([2.0, -2.0])).tolist() == [50.0, 50.0]
    times, targets = np.array([1.0, 2.0, 3.0]), np.array([10.0, 8.0, 7.0])
    assert run.extrapolate_truth(times, targets, 2.5) == pytest.approx(7.5)
    assert run.extrapolate_truth(times, targets, 4.0) == pytest.approx(6.0)
    assert run.extrapolate_truth(times, targets, 0.0) == pytest.approx(12.0)
    draws = np.random.default_rng(0).integers(0, 3, size=(2000, 3))
    lo, hi = run.bootstrap_macro(np.array([1.0, 2.0, 3.0]), draws)
    assert 1.0 <= lo < 2.0 < hi <= 3.0
    seq = np.array(["a", "a", "b"])
    assert run.sequence_means(np.array([1.0, 3.0, 5.0]), seq, ["a", "b"]).tolist() == [2.0, 5.0]
    assert math.isclose(float(run.rte(np.array([60.0]), np.array([30.0]))[0]), 100.0)


def test_root_requires_environment_variable() -> None:
    with pytest.raises(RuntimeError, match="PHASE_TTC_FCWD_ROOT"):
        fcwd.fcwd_root({})
