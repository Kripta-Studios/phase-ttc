"""Robust geometry on synthetic looming boxes that saturate the frame: recovery and causality."""

from __future__ import annotations

import numpy as np
import pytest

from operational.geometry_robust import robust
from operational.phase_geometry import geometry
from operational.phase_geometry.metrics import phase_to_ttc

STEP_US = 100_000
F_PX = 1000.0


def _looming(
    n: int,
    z0: float,
    v: float,
    width_m: float = 2.0,
    height_m: float = 1.5,
    centre: tuple[float, float] = (640.0, 380.0),
    start_us: int = 10_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constant-speed approach; boxes clipped to the 1280x720 frame like supplied boxes."""
    times = start_us + STEP_US * np.arange(n, dtype=np.int64)
    z = z0 - v * (times - times[0]) * 1e-6
    w, h = F_PX * width_m / z, F_PX * height_m / z
    x1, y1 = centre[0] - w / 2, centre[1] - h / 2
    boxes = np.stack([x1, y1, x1 + w - 1.0, y1 + h - 1.0], axis=1)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, 1279)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, 719)
    return times, boxes, z


def _ttc_from_phase(phi: float) -> float:
    return float(phase_to_ttc(np.asarray([phi]))[0])


def _features(
    times: np.ndarray, boxes: np.ndarray, query: int, memory_us: int = 600_000
) -> tuple[dict[str, float], dict[str, float]]:
    g = geometry.window_features(times, boxes, query)
    return g, robust.robust_features(times, boxes, query, g, memory_us=memory_us)


def _first_saturated(boxes: np.ndarray) -> int:
    flags = robust.edge_flags(boxes)
    return int(np.flatnonzero(flags.all(axis=1))[0])


def test_robust_recovers_ttc_after_saturation_where_plain_slope_fails() -> None:
    times, boxes, z = _looming(40, z0=6.0, v=1.5)
    first = _first_saturated(boxes)
    query_index = first + 3  # window: three or four free frames, then a saturated tail
    query = int(times[query_index])
    g, r = _features(times[: query_index + 1], boxes[: query_index + 1], query)
    truth = z[query_index] / 1.5
    assert r["sat"] == 1.0 and r["n_clipped"] == 4.0
    assert r["src_free_axis"] == 1.0 and r["age_free_s"] > 0.0
    robust_ttc = 1.0 / r["k_robust"]
    assert robust_ttc == pytest.approx(truth, rel=1e-6)
    assert _ttc_from_phase(r["phi_robust"]) == pytest.approx(truth, rel=1e-3)
    plain_ttc = _ttc_from_phase(g["phi_area"])
    assert abs(plain_ttc - truth) / truth > 0.5


def test_memory_reaches_beyond_a_fully_saturated_window() -> None:
    times, boxes, z = _looming(80, z0=10.0, v=1.0, width_m=4.0, height_m=3.0)
    first = _first_saturated(boxes)
    query_index = first + 9  # the whole 600 ms window is saturated
    query = int(times[query_index])
    assert z[query_index] > 1.0
    _, short = _features(times[: query_index + 1], boxes[: query_index + 1], query)
    _, long = _features(
        times[: query_index + 1], boxes[: query_index + 1], query, memory_us=2_000_000
    )
    assert short["src_missing"] == 1.0 and short["sat"] == 1.0
    assert short["clip_run_age_s"] == pytest.approx(0.6)
    assert long["src_free_axis"] == 1.0 and long["age_free_s"] > 0.6
    assert 1.0 / long["k_robust"] == pytest.approx(z[query_index] / 1.0, rel=1e-6)


def test_single_clipped_axis_uses_the_free_axis_at_the_query() -> None:
    times, boxes, z = _looming(30, z0=4.0, v=0.5, width_m=4.5, height_m=0.6)
    flags = robust.edge_flags(boxes)
    index = int(np.flatnonzero(flags[:, 0] & flags[:, 1] & ~flags[:, 2] & ~flags[:, 3])[3])
    query = int(times[index])
    _, r = _features(times[: index + 1], boxes[: index + 1], query)
    assert r["src_free_axis"] == 1.0 and r["age_free_s"] == 0.0
    assert 1.0 / r["k_robust"] == pytest.approx(z[index] / 0.5, rel=1e-6)


def test_clean_window_is_identical_to_plain_geometry() -> None:
    times, boxes, _ = _looming(7, z0=30.0, v=3.0)
    g, r = _features(times, boxes, int(times[-1]))
    assert r["src_clean"] == 1.0 and r["phi_robust"] == g["phi_area"]
    assert r["sat"] == 0.0 and r["n_clipped"] == 0.0 and r["edge_valid"] == 0.0


def test_future_and_out_of_memory_frames_never_change_features() -> None:
    times, boxes, _ = _looming(40, z0=6.0, v=1.5)
    index = _first_saturated(boxes) + 2
    query = int(times[index])
    g = geometry.window_features(times[: index + 1], boxes[: index + 1], query)
    base = robust.robust_features(times[: index + 1], boxes[: index + 1], query, g)
    noisy = boxes.copy()
    noisy[index + 1 :] = [100.0, 100.0, 120.0, 130.0]  # arbitrary future boxes
    noisy[: index - 7] = [5.0, 5.0, 9.0, 9.0]  # frames older than the 600 ms memory
    after = robust.robust_features(times, noisy, query, g)
    assert after == base


def test_saturation_rule_and_fcwd_full_frame_box() -> None:
    full = np.asarray([[1.0, 1.0, 1281.0, 721.0]])
    flags = robust.edge_flags(full, robust.FCWD)
    cover = robust.coverage(full[0], robust.FCWD)
    assert flags.all() and cover > 0.99 and robust.is_saturated(flags[0], cover)
    small = np.asarray([0.0, 300.0, 200.0, 400.0])
    assert not robust.is_saturated(robust.edge_flags(small)[0], robust.coverage(small))
    big = np.asarray([100.0, 0.0, 1100.0, 719.0])
    assert robust.is_saturated(robust.edge_flags(big)[0], robust.coverage(big))


def test_inverse_size_rate_is_exact_under_constant_velocity() -> None:
    t = np.asarray([-0.5, -0.4, -0.3])
    z = 7.0 - 2.0 * t  # depth at query time 7 m, closing speed 2 m/s
    assert robust.inverse_size_rate(t, 100.0 / z) == pytest.approx(2.0 / 7.0, rel=1e-9)
    receding = 7.0 + 2.0 * t
    assert robust.inverse_size_rate(t, 100.0 / receding) == pytest.approx(-2.0 / 7.0, rel=1e-9)


def test_crops_are_deterministic_inside_the_image_and_preserve_scale_ratios() -> None:
    current = np.asarray([[100.0, 100.0, 300.0, 250.0], [900.0, 400.0, 1279.0, 719.0]])
    a, b = robust.draw_crops(current), robust.draw_crops(current)
    assert np.array_equal(a, b)
    zoom = a[:, 2]
    assert ((zoom >= 1.0) & (zoom <= robust.AUG_MAX_ZOOM)).all()
    assert (a[:, 0] >= 0).all() and (a[:, 0] + 1280 / zoom <= 1280 + 1e-9).all()
    identity = robust.crop_boxes(current, np.asarray([0.0, 0.0, 1.0]))
    assert np.array_equal(identity, current)
    inside = np.asarray([[500.0, 300.0, 540.0, 330.0], [495.0, 296.0, 545.0, 334.0]])
    zoomed = robust.crop_boxes(inside, np.asarray([400.0, 250.0, 2.0]))
    ratio = (zoomed[1, 2] - zoomed[1, 0]) / (zoomed[0, 2] - zoomed[0, 0])
    assert ratio == pytest.approx((inside[1, 2] - inside[1, 0]) / (inside[0, 2] - inside[0, 0]))
    gone = robust.crop_boxes(np.asarray([[0.0, 0.0, 10.0, 10.0]]), np.asarray([600.0, 300.0, 3.0]))
    assert gone[0, 2] - gone[0, 0] + 1.0 <= 0.0


def test_compute_tables_matches_plain_geometry_and_is_causal() -> None:
    times, boxes, _ = _looming(12, z0=20.0, v=4.0)
    rows = list(range(3, 12))
    frame_lists = [times[max(0, i - 6) : i + 1] for i in rows]
    box_lists = [boxes[max(0, i - 6) : i + 1] for i in rows]
    seq, track = ["S"] * len(rows), ["T"] * len(rows)
    stamps = times[rows]
    g, r = robust.compute_tables(seq, track, stamps, frame_lists, box_lists, memories_us=[600_000])
    reference, _ = geometry.compute_table(seq, track, stamps, frame_lists, box_lists)
    assert np.array_equal(g, reference)
    g_short, r_short = robust.compute_tables(
        seq[:5], track[:5], stamps[:5], frame_lists[:5], box_lists[:5], memories_us=[600_000]
    )
    assert np.array_equal(g_short, g[:5]) and np.array_equal(r_short[600_000], r[600_000][:5])
    assert robust.learned_r(r[600_000]).shape == (len(rows), len(robust.R_FIELDS))
