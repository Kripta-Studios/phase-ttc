"""Synthetic tests: official-scorer equivalence and bootstrap determinism."""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest

from operational.history_ablation.metrics import (
    BANDS,
    bootstrap_weights,
    interval,
    point_metrics,
    resampled_metrics,
    sequence_sums,
    summarize,
)

OFFICIAL = Path(
    os.environ.get(
        "GARL_TTC_OFFICIAL_SCORER",
        r"E:\Garl-TTC\garl_ttc_benchmark\codabench\scoring_program\score.py",
    )
)
# A target of exactly 0 s crashes the official scorer (1 - dT/0); real targets never are.
TOY_TARGET = np.array(
    [0.5, 2.0, 2.9, 3.0, 4.5, 5.9, 6.5, 9.9, 10.0, -0.5, -9.0, -0.05, -12.0, 15.0]
)
TOY_PRED = np.array([0.6, 1.5, 3.4, 0.05, 4.0, 7.0, 8.0, 9.0, -11.0, -0.4, -60.0, 60.0, 3.0, -2.0])


def _official():
    if not OFFICIAL.is_file():
        pytest.skip(f"official scorer not present at {OFFICIAL}")
    spec = importlib.util.spec_from_file_location("garl_official_score", OFFICIAL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metric_equals_official_scorer_on_toy_case():
    official = _official()
    reference = [{"sample_token": f"t{i}", "ttc": float(t)} for i, t in enumerate(TOY_TARGET)]
    prediction = {f"t{i}": float(p) for i, p in enumerate(TOY_PRED)}
    expected = official.evaluate_rows(reference, prediction, dT=0.1)
    ours = summarize(TOY_TARGET, TOY_PRED)
    for short, name in zip("csln", BANDS, strict=True):
        assert ours[f"MiD_{name}"] == pytest.approx(expected[f"MiD{short}"], rel=1e-12)
    assert ours["Overall_MiD"] == pytest.approx(expected["overall_MiD"], rel=1e-12)
    assert ours["mean_MiD"] == pytest.approx(expected["mean_MiD"], rel=1e-12)
    assert ours["n_nan_MiD"] == 1  # the 0.05 s prediction has no defined MiD


def test_sequence_sums_reproduce_pooled_metrics():
    rng = np.random.default_rng(0)
    target = rng.uniform(-9.9, 9.9, size=500)
    target[np.abs(target) < 0.2] = 1.0
    pred = target * rng.uniform(0.7, 1.3, size=500)
    groups = np.array([f"s{i % 7}" for i in range(500)])
    sums = sequence_sums(target, pred, groups, sorted(set(groups.tolist())))
    pooled = summarize(target, pred)
    from_sums = point_metrics(sums)
    for key in ("Overall_MiD", "MAE_s", *(f"MiD_{b}" for b in BANDS)):
        assert from_sums[key] == pytest.approx(pooled[key], rel=1e-10)


def test_bootstrap_is_deterministic_and_resamples_whole_sequences():
    first = bootstrap_weights(40, 2_000, seed=20261118)
    second = bootstrap_weights(40, 2_000, seed=20261118)
    other = bootstrap_weights(40, 2_000, seed=1)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.all(first.sum(1) == 40)
    assert np.all(first == np.round(first)) and first.min() >= 0


def test_paired_bootstrap_difference_is_zero_for_identical_arms_and_deterministic():
    rng = np.random.default_rng(1)
    target = rng.uniform(0.3, 9.5, size=400) * rng.choice([-1.0, 1.0], size=400)
    groups = np.array([f"s{i % 10}" for i in range(400)])
    names = sorted(set(groups.tolist()))
    a = sequence_sums(target, target * 1.1, groups, names)
    b = sequence_sums(target, target * 1.2, groups, names)
    weights = bootstrap_weights(len(names), 500, seed=3)
    delta = (
        resampled_metrics(b, weights)["Overall_MiD"] - resampled_metrics(a, weights)["Overall_MiD"]
    )
    again = (
        resampled_metrics(b, weights)["Overall_MiD"] - resampled_metrics(a, weights)["Overall_MiD"]
    )
    assert np.array_equal(delta, again)
    same = (
        resampled_metrics(a, weights)["Overall_MiD"] - resampled_metrics(a, weights)["Overall_MiD"]
    )
    assert np.all(np.nan_to_num(same) == 0)
    ci = interval(delta)
    assert ci["lo"] <= ci["hi"] and ci["replicates"] + ci["dropped_nonfinite"] == 500
    assert not math.isnan(ci["lo"])
