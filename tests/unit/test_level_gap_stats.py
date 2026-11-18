"""Unit tests for the label-free level-gap helpers."""

from __future__ import annotations

import numpy as np
import pytest

from operational.level_gap import stats
from operational.level_gap.addenda import tilt_weights


def test_predicted_band_strata() -> None:
    ttc = np.array([0.5, 3.0, 3.1, 6.0, 7.0, 10.5, -0.5, -10.0, -12.0])
    assert stats.predicted_band(ttc).tolist() == [0, 0, 1, 1, 2, 3, 4, 5, 5]


def test_weighted_overall_matches_unweighted_and_ignores_zero_weights() -> None:
    mid = np.array([10.0, 20.0, 30.0, 40.0, 1000.0])
    bands = np.array([0, 1, 2, 3, 0])
    plain = stats.weighted_overall(mid[:4], bands[:4])
    assert plain["overall"] == pytest.approx(0.5 * 10 + 0.3 * 20 + 0.1 * 30 + 0.1 * 40)
    masked = stats.weighted_overall(mid, bands, np.array([1, 1, 1, 1, 0.0]))
    assert masked["overall"] == pytest.approx(plain["overall"])


def test_cell_keys_use_fixed_radices() -> None:
    a = stats.cell_keys([np.array([0, 1]), np.array([1, 0])], [2, 3])
    b = stats.cell_keys([np.array([1]), np.array([0])], [2, 3])
    assert a.tolist() == [1, 3] and b.tolist() == [3]
    with pytest.raises(ValueError):
        stats.cell_keys([np.array([3])], [3])


def test_stratified_weights_reproduce_test_cell_shares() -> None:
    train = np.array([0, 0, 0, 1])
    test = np.array([0, 1, 1, 1])
    weights, info = stats.stratified_weights(train, test)
    share_1 = weights[train == 1].sum() / weights.sum()
    assert share_1 == pytest.approx(0.75)
    assert info["test_mass_unrepresented"] == 0.0
    _, info_missing = stats.stratified_weights(train, np.array([0, 2]))
    assert info_missing["test_mass_unrepresented"] == pytest.approx(0.5)


def test_bootstrap_band_means_point_and_shape() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    bands = np.array([0, 0, 1, 1])
    clusters = np.array(["a", "b", "a", "b"])
    reps = stats.bootstrap_band_means({"x": values}, np.ones(4), bands, clusters, replicates=50)
    assert reps["x"].shape == (50, 4)
    finite = reps["x"][:, 0]
    assert np.all((finite >= 1.0) & (finite <= 2.0))


def test_three_cornered_hat_recovers_independent_noise() -> None:
    rng = np.random.default_rng(0)
    truth = rng.normal(size=200_000)
    a = truth + rng.normal(scale=0.1, size=truth.size)
    b = truth + rng.normal(scale=0.2, size=truth.size)
    c = truth + rng.normal(scale=0.3, size=truth.size)
    est = stats.three_cornered_hat(a, b, c, clip=10.0)
    assert est["sd_a"] == pytest.approx(0.1, rel=0.05)
    assert est["sd_b"] == pytest.approx(0.2, rel=0.03)
    assert est["sd_c"] == pytest.approx(0.3, rel=0.03)


def test_conditional_wins_bins() -> None:
    rows = stats.conditional_wins(
        np.array([1.0, 5.0, 0.0]), np.array([2.0, 1.0, 3.0]), np.array([5.0, 15.0, 15.0]), [0, 10]
    )
    assert rows[0]["win_rate"] == 1.0 and rows[0]["rows"] == 1
    assert rows[1]["win_rate"] == 0.5 and rows[1]["mean_change"] == pytest.approx(0.5)


def test_concentration_and_blend() -> None:
    assert stats.concentration_share(np.array([10.0, 0.0, 0.0, 0.0, 0.0]), 0.1) == 1.0
    assert stats.blend_phase(np.array([1.0]), np.array([0.0]), 0.25)[0] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        stats.blend_phase(np.array([1.0]), np.array([0.0]), 1.5)


def test_multiplicative_projection_identity() -> None:
    reference = {"c": 40.0, "s": 30.0, "l": 20.0, "n": 10.0}
    value = stats.project_multiplicative(reference, np.ones((1, 4)))[0]
    assert value == pytest.approx(0.5 * 40 + 0.3 * 30 + 0.1 * 20 + 0.1 * 10)


def test_tilt_weights_match_band_targets() -> None:
    rng = np.random.default_rng(1)
    error = np.abs(rng.normal(scale=100.0, size=4000))
    bands = np.repeat(np.arange(4), 1000)
    targets = {"c": 30.0, "s": 40.0, "l": 50.0, "n": 1e6}
    weights, lambdas = tilt_weights(error, bands, targets)
    for index, band in enumerate(("c", "s", "l")):
        keep = bands == index
        assert np.average(error[keep], weights=weights[keep]) == pytest.approx(
            targets[band], rel=1e-6
        )
    assert lambdas["n"] == 0.0
