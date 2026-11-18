"""Official-formula MiD, band weights, band-weighted loss and bootstrap determinism."""

from __future__ import annotations

import numpy as np
import pytest

from operational.phase_geometry import metrics
from operational.phase_geometry.weights import band_mass, group_mass, weighted_l1


def test_band_edges_follow_official_scorer():
    t = np.array([-10.0, -9.99, 0.0, 0.2, 3.0, 3.01, 6.0, 6.5, 10.0, 10.5])
    assert metrics.band_index(t).tolist() == [-1, 3, 3, 0, 0, 1, 1, 2, 2, -1]


def test_row_mid_matches_formula_and_invalid_cases():
    gt = np.array([2.0, -3.0, 5.0, 4.0])
    pred = np.array([2.5, -2.0, 0.05, 0.0])
    mid = metrics.row_mid(gt, pred)
    expected0 = abs(np.log(1 - 0.1 / 2.0) - np.log(1 - 0.1 / 2.5)) * 1e4
    assert mid[0] == pytest.approx(expected0)
    assert np.isfinite(mid[1])
    assert np.isnan(mid[2]) and np.isnan(mid[3])
    assert metrics.failed(pred).tolist() == [False, False, True, True]


def test_overall_weights_and_phase_round_trip():
    gt = np.array([1.0, 2.0, 4.0, 8.0, -5.0])
    pred = np.array([1.1, 2.0, 4.4, 8.0, -4.0])
    score = metrics.summarize(gt, pred)
    mid = metrics.row_mid(gt, pred)
    expected = 0.5 * mid[:2].mean() + 0.3 * mid[2] + 0.1 * mid[3] + 0.1 * mid[4]
    assert score["overall_MiD"] == pytest.approx(expected)
    phase = metrics.ttc_to_phase(gt)
    assert metrics.phase_to_ttc(phase) == pytest.approx(gt)


def test_band_mass_sums_to_band_weights_and_excludes_outside_rows():
    rng = np.random.default_rng(3)
    truth = np.concatenate(
        (
            rng.uniform(0.2, 3, 7),
            rng.uniform(3.1, 6, 3),
            rng.uniform(6.1, 10, 5),
            rng.uniform(-9, -0.1, 11),
            [12.0],
        )
    )
    mass = band_mass(truth)
    bands = metrics.band_index(truth)
    assert mass.sum() == pytest.approx(1.0, rel=1e-6)
    assert mass[bands == -1].tolist() == [0.0]
    for index, name in enumerate(metrics.BANDS):
        assert mass[bands == index].sum() == pytest.approx(metrics.BAND_WEIGHTS[name], rel=1e-6)


def test_band_weighted_l1_equals_overall_mid_in_phase_units():
    rng = np.random.default_rng(4)
    truth = np.concatenate(
        (
            rng.uniform(0.3, 3, 20),
            rng.uniform(3.1, 6, 10),
            rng.uniform(6.1, 10, 5),
            rng.uniform(-9, -0.5, 30),
        )
    )
    target_phase = metrics.ttc_to_phase(truth)
    prediction_phase = target_phase + rng.normal(0, 0.004, len(truth))
    loss = weighted_l1(prediction_phase, target_phase, band_mass(truth))
    mid = 1e4 * np.abs(prediction_phase - target_phase)
    assert loss * 0.03 * 1e4 == pytest.approx(
        metrics.overall_from_mid(mid, metrics.band_index(truth)), rel=1e-5
    )


def test_group_mass_balances_sequences():
    groups = np.array(["a"] * 3 + ["b"] * 7)
    mass = group_mass(groups)
    assert mass[:3].sum() == pytest.approx(0.5) and mass[3:].sum() == pytest.approx(0.5)


def test_bootstrap_is_deterministic_and_point_is_overall():
    rng = np.random.default_rng(5)
    n = 400
    truth = rng.choice([1.0, 4.0, 8.0, -3.0], n)
    clusters = np.repeat(np.arange(20), n // 20).astype(str)
    values = rng.gamma(2.0, 40.0, n)
    bands = metrics.band_index(truth)
    first = metrics.bootstrap_overall(values, bands, clusters, replicates=2000, seed=11)
    second = metrics.bootstrap_overall(values, bands, clusters, replicates=2000, seed=11)
    other = metrics.bootstrap_overall(values, bands, clusters, replicates=2000, seed=12)
    assert first == second
    assert first["ci95_low"] != other["ci95_low"]
    assert first["point"] == pytest.approx(metrics.overall_from_mid(values, bands))
    assert first["ci95_low"] < first["point"] < first["ci95_high"]


def test_weighted_median_minimizes_weighted_l1():
    values = np.array([0.0, 1.0, 2.0, 10.0])
    weights = np.array([0.1, 0.1, 0.1, 0.7])
    assert metrics.weighted_median(values, weights) == 10.0
