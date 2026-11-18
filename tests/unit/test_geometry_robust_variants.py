"""Robust-geometry variant designs, augmented training mass and stratum statistics."""

from __future__ import annotations

import numpy as np
import pytest

from operational.geometry_robust import evaluate, variants


def test_variant_columns_respect_their_declared_information() -> None:
    for name, v in variants.VARIANTS.items():
        assert max(v.columns) < 56 and len(set(v.columns)) == len(v.columns)
        if name.startswith("GEO_"):
            assert min(v.columns) >= 20 and not v.use_width  # no RGB/event fields
        if name in variants.TEST12_ELIGIBLE:
            assert not set(v.columns) & set(variants.R_LONG_COLS)  # 650 ms history only
        assert not set(v.columns) & set(range(10, 20))  # no event fields anywhere
    assert set(variants.FCWD_VARIANTS) <= {n for n in variants.VARIANTS if n.startswith("GEO_")}


def test_augmented_mass_keeps_band_weights_and_copy_fraction() -> None:
    truth = np.asarray([1.0, 2.0, 4.0, 7.0, -3.0, -1.0, 5.0, 2.5])
    plain = variants.training_mass(truth, augmented=False)
    aug = variants.training_mass(truth, augmented=True)
    assert plain.sum() == pytest.approx(1.0) and aug.sum() == pytest.approx(1.0)
    assert aug[len(truth) :].sum() == pytest.approx(variants.AUG_MASS_FRACTION)
    assert np.allclose(aug[: len(truth)], 0.75 * plain)


def test_stratum_overall_renormalizes_missing_bands() -> None:
    mid = np.asarray([10.0, 30.0, 20.0])
    bands = np.asarray([0, 0, 1])  # crucial and small only
    out = evaluate.stratum_overall(mid, bands)
    assert out["overall"] == pytest.approx((0.5 * 20.0 + 0.3 * 20.0) / 0.8)
    assert np.isnan(out["MiDl"]) and out["rows"] == 3


def test_bootstrap_point_matches_and_interval_brackets_it() -> None:
    rng = np.random.default_rng(0)
    n = 400
    clusters = rng.integers(0, 10, n)
    bands = rng.integers(0, 4, n)
    values = rng.normal(-1.0, 1.0, n)
    draws = np.random.default_rng(1).integers(0, 10, (500, 10))
    result = evaluate.bootstrap(values, bands, clusters, draws)
    assert result["point"] == pytest.approx(evaluate.stratum_overall(values, bands)["overall"])
    assert result["ci95"][0] <= result["point"] <= result["ci95"][1]
    masked = evaluate.bootstrap(np.where(bands == 0, values, np.nan), bands, clusters, draws)
    assert masked["point"] == pytest.approx(values[bands == 0].mean())
