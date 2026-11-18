"""Garl feature vector R and arm designs of the matched-comparator study (numpy only)."""

from __future__ import annotations

import numpy as np
import pytest

from operational.matched_comparator.features import (
    design,
    garl_anchor,
    garl_inputs,
    garl_phase,
    release_ttc,
)


def test_garl_phase_equals_benchmark_phase_of_release_ttc() -> None:
    heights = np.asarray([[90.0, 92.0], [100.0, 99.0], [50.0, 50.5]])
    phase, valid = garl_phase(heights)
    ttc = release_ttc(heights)
    assert valid.all()
    np.testing.assert_allclose(phase, -np.log1p(-0.1 / ttc), rtol=1e-12)


def test_invalid_heights_are_flagged_and_zeroed() -> None:
    heights = np.asarray([[np.nan, 80.0], [-1.0, 80.0], [80.0, 0.0], [80.0, 81.0]])
    phase, valid = garl_phase(heights)
    assert valid.tolist() == [False, False, False, True]
    r = garl_inputs(heights)
    assert r.dtype == np.float32
    np.testing.assert_array_equal(r[:3, :2], 0.0)
    np.testing.assert_array_equal(r[:, 2], [0.0, 0.0, 0.0, 1.0])
    assert np.isfinite(r).all()
    assert phase[0] == 0.0


def test_r_vector_clips_phase_and_log_height() -> None:
    heights = np.asarray([[10.0, 100.0], [128.0, 128.0], [1e-9, 1e-9 * 1.01]])
    r = garl_inputs(heights)
    assert r[0, 0] == pytest.approx(0.5 / 0.03)
    assert r[1, 0] == pytest.approx(0.0)
    assert r[1, 1] == pytest.approx(0.0)
    assert r[2, 1] == pytest.approx(-3.0)


def test_anchor_falls_back_to_clipped_geometry() -> None:
    heights = np.asarray([[np.nan, 1.0], [99.0, 100.0]])
    anchor = garl_anchor(heights, np.asarray([0.9, 0.0]))
    assert anchor[0] == pytest.approx(0.5)
    assert anchor[1] == pytest.approx(-np.log(0.99), rel=1e-6)


def test_designs_have_pre_registered_shapes() -> None:
    rng = np.random.default_rng(0)
    g = rng.normal(size=(5, 10)).astype(np.float32)
    heights = rng.uniform(80, 100, size=(5, 2))
    phi = rng.normal(scale=0.01, size=5)
    x_r, anchor, width = design("R", g, heights, phi)
    x_gr, anchor2, _ = design("GEO_R", g, heights, phi)
    assert x_r.shape == (5, 3) and x_gr.shape == (5, 13)
    np.testing.assert_array_equal(x_gr[:, :10], g)
    np.testing.assert_array_equal(x_gr[:, 10:], x_r)
    np.testing.assert_array_equal(anchor, anchor2)
    np.testing.assert_array_equal(width, 0.0)
    with pytest.raises(ValueError):
        design("EVENTS", g, heights, phi)


def test_s8_rows_keeps_every_eighth_query_of_each_track() -> None:
    from operational.matched_comparator.features import s8_rows

    sequence = np.asarray(["a"] * 10 + ["b"] * 3)
    track = np.asarray(["1"] * 10 + ["1"] * 3)
    times = np.asarray([9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 5, 6, 7])
    tokens = np.asarray([f"t{i}" for i in range(13)])
    rows = s8_rows(sequence, track, times, tokens)
    assert rows.tolist() == [1, 9, 10]
