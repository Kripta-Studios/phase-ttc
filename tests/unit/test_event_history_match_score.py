"""Synthetic tests for the pre-declared scoring, bootstrap and decision rule."""

from __future__ import annotations

import numpy as np
import pytest

from operational.event_history_match.score import (
    bootstrap_draws,
    decide,
    metrics,
    paired_contrast,
    phase,
)


def test_metrics_known_values() -> None:
    target = np.array([1.0, 2.0, 4.0])
    prediction = np.array([1.5, 2.0, 3.0])
    value = metrics(prediction, target)
    assert value["MAE_s"] == pytest.approx(0.5)
    assert value["median_AE_s"] == pytest.approx(0.5)
    assert value["RTE_pct"] == pytest.approx((50 + 0 + 25) / 3)
    expected_mid = 100 * np.mean(np.abs(phase(prediction) - phase(target)))
    assert value["MiD"] == pytest.approx(expected_mid)
    assert phase(np.array([0.2]))[0] == pytest.approx(np.log(2.0))


def test_bootstrap_resamples_whole_recordings_deterministically() -> None:
    groups = np.array(["a"] * 3 + ["b"] * 2 + ["c"])
    first = bootstrap_draws(groups, 50, 7)
    second = bootstrap_draws(groups, 50, 7)
    assert all(np.array_equal(x, y) for x, y in zip(first, second, strict=True))
    for rows in first:
        for name in ("a", "b", "c"):
            members = np.flatnonzero(groups == name)
            count = np.isin(rows, members).sum()
            assert count % len(members) == 0  # recordings enter whole


def test_decision_rule() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20).astype(str), 10)
    target = rng.uniform(1, 8, size=len(groups))
    better = target * np.exp(rng.normal(0, 0.02, size=len(groups)))
    worse = target * np.exp(rng.normal(0, 0.3, size=len(groups)))
    draws = bootstrap_draws(groups, 500, 1)
    assert decide(paired_contrast(better, worse, target, draws)) == "HELPS"
    assert decide(paired_contrast(worse, better, target, draws)) == "HURTS"
    assert decide(paired_contrast(better, better, target, draws)) == "NO_MEASURABLE_EFFECT"
