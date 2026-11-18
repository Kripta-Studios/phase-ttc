"""Vectorized official GarlTTC MiD, band-weighted Overall and sequence bootstrap (numpy).

The row formula mirrors ``score.py::evaluate_rows`` (dT = 0.1); every Overall reported by
this study is additionally cross-checked against the pinned official scorer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

DT = 0.1
BANDS = ("c", "s", "l", "n")
BAND_WEIGHTS = {"c": 0.5, "s": 0.3, "l": 0.1, "n": 0.1}
WEIGHT_VECTOR = np.asarray([BAND_WEIGHTS[b] for b in BANDS], dtype=np.float64)
MIN_PHASE = -math.log(2.0) + 1e-6
MAX_PHASE = 8.0
BOOTSTRAP_SEED = 20261118
BOOTSTRAP_REPLICATES = 10_000

FloatArray = NDArray[np.float64]


def ttc_to_phase(ttc: NDArray[Any]) -> FloatArray:
    """Benchmark phase ``-log(1 - 0.1/T)`` (requires T < 0 or T > 0.1)."""
    t = np.asarray(ttc, dtype=np.float64)
    if not np.isfinite(t).all() or not ((t < 0) | (t > DT)).all():
        raise ValueError("TTC must be finite with T<0 or T>0.1 s")
    return -np.log1p(-DT / t)


def phase_to_ttc(phase: NDArray[Any]) -> FloatArray:
    """Numpy mirror of ``e_jepa_ttc.simplex_t.phase.phase_to_ttc`` (|T| <= 60 s)."""
    phi = np.clip(np.asarray(phase, dtype=np.float64), MIN_PHASE, MAX_PHASE)
    q = -np.expm1(-phi) / DT
    sign = np.where(q < 0, -1.0, 1.0)
    return sign / np.maximum(np.abs(q), 1.0 / 60.0)


def band_index(truth_ttc: NDArray[Any]) -> NDArray[np.int64]:
    """Official band of each target TTC: 0=c (0,3], 1=s (3,6], 2=l (6,10], 3=n (-10,0], -1 none."""
    t = np.asarray(truth_ttc, dtype=np.float64)
    out = np.full(t.shape, -1, dtype=np.int64)
    out[(t > -10) & (t <= 0)] = 3
    out[(t > 0) & (t <= 3)] = 0
    out[(t > 3) & (t <= 6)] = 1
    out[(t > 6) & (t <= 10)] = 2
    return out


def row_mid(truth_ttc: NDArray[Any], prediction_ttc: NDArray[Any]) -> FloatArray:
    """Per-row MiD exactly as the official scorer (NaN where its log ratio is undefined)."""
    gt = np.asarray(truth_ttc, dtype=np.float64)
    pred = np.asarray(prediction_ttc, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pred_ratio = np.where(pred == 0.0, np.nan, 1.0 - DT / pred)
        gt_ratio = 1.0 - DT / gt
        ok = (pred_ratio > 0) & (gt_ratio > 0) & np.isfinite(pred_ratio)
        mid = np.abs(np.log(np.where(ok, gt_ratio, 1.0)) - np.log(np.where(ok, pred_ratio, 1.0)))
    return np.where(ok, mid * 1e4, np.nan)


def failed(prediction_ttc: NDArray[Any]) -> NDArray[np.bool_]:
    """Official failure flag: non-finite or |T| < 0.1 s."""
    pred = np.asarray(prediction_ttc, dtype=np.float64)
    return ~np.isfinite(pred) | (np.abs(pred) < DT)


def summarize(truth_ttc: NDArray[Any], prediction_ttc: NDArray[Any]) -> dict[str, float]:
    """Official-style band means (finite only), failure rates, Overall and mean MiD."""
    mid = row_mid(truth_ttc, prediction_ttc)
    bands = band_index(truth_ttc)
    fail = failed(prediction_ttc)
    out: dict[str, float] = {}
    for index, name in enumerate(BANDS):
        member = bands == index
        values = mid[member & np.isfinite(mid)]
        out[f"MiD{name}"] = float(values.mean()) if len(values) else float("nan")
        out[f"FR{name}"] = float(fail[member].mean() * 100.0) if member.any() else float("nan")
        out[f"n_{name}"] = float(member.sum())
    out["overall_MiD"] = float(sum(out[f"MiD{b}"] * BAND_WEIGHTS[b] for b in BANDS))
    finite = mid[np.isfinite(mid)]
    out["mean_MiD"] = float(finite.mean()) if len(finite) else float("nan")
    out["failed_rate"] = float(fail.mean() * 100.0)
    out["invalid_mid"] = float((~np.isfinite(mid)).sum())
    out["num_samples"] = float(len(mid))
    return out


def overall_from_mid(mid: NDArray[Any], bands: NDArray[Any]) -> float:
    """Band-weighted Overall of a per-row MiD vector (finite rows only)."""
    m = np.asarray(mid, dtype=np.float64)
    total = 0.0
    for index, name in enumerate(BANDS):
        values = m[(np.asarray(bands) == index) & np.isfinite(m)]
        total += BAND_WEIGHTS[name] * (float(values.mean()) if len(values) else float("nan"))
    return total


def official_check(
    scorer: ModuleType, truth_ttc: NDArray[Any], prediction_ttc: NDArray[Any]
) -> float:
    """Return |official overall - vectorized overall| using the pinned scorer module."""
    reference = [{"sample_token": str(i), "ttc": float(t)} for i, t in enumerate(truth_ttc)]
    values = {str(i): float(p) for i, p in enumerate(prediction_ttc)}
    official = scorer.evaluate_rows(reference, values, dT=DT)
    mine = summarize(truth_ttc, prediction_ttc)
    return float(abs(official["overall_MiD"] - mine["overall_MiD"]))


def cluster_sums(
    values: NDArray[Any], bands: NDArray[Any], clusters: NDArray[Any]
) -> tuple[FloatArray, FloatArray, NDArray[Any]]:
    """Per-cluster, per-band finite sums and counts ``[n_clusters, 4]``."""
    v = np.asarray(values, dtype=np.float64)
    b = np.asarray(bands)
    names, inverse = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    sums = np.zeros((len(names), len(BANDS)))
    counts = np.zeros((len(names), len(BANDS)))
    ok = np.isfinite(v) & (b >= 0)
    np.add.at(sums, (inverse[ok], b[ok]), v[ok])
    np.add.at(counts, (inverse[ok], b[ok]), 1.0)
    return sums, counts, names


def bootstrap_overall(
    values: NDArray[Any],
    bands: NDArray[Any],
    clusters: NDArray[Any],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Cluster (sequence) bootstrap of the band-weighted Overall of ``values``.

    ``values`` is a per-row MiD or a paired per-row MiD difference.  Clusters are resampled
    with replacement; band means are recomputed from the pooled resampled rows.
    """
    sums, counts, names = cluster_sums(values, bands, clusters)
    draws = np.random.default_rng(seed).integers(0, len(names), (replicates, len(names)))
    pooled_sums = sums[draws].sum(axis=1)
    pooled_counts = counts[draws].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        band_means = pooled_sums / pooled_counts
    distribution = band_means @ WEIGHT_VECTOR
    point_band = sums.sum(0) / counts.sum(0)
    low, high = np.nanquantile(distribution, [0.025, 0.975])
    band_ci = np.nanquantile(band_means, [0.025, 0.975], axis=0)
    return {
        "point": float(point_band @ WEIGHT_VECTOR),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "band_point": {b: float(point_band[i]) for i, b in enumerate(BANDS)},
        "band_ci95": {b: [float(band_ci[0, i]), float(band_ci[1, i])] for i, b in enumerate(BANDS)},
        "clusters": len(names),
        "replicates": replicates,
        "seed": seed,
        "undefined_replicates": int((~np.isfinite(distribution)).sum()),
    }


def weighted_median(values: NDArray[Any], weights: NDArray[Any]) -> float:
    """Lower weighted median (minimizer of the weighted L1 loss)."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    order = np.argsort(v, kind="stable")
    cumulative = np.cumsum(w[order])
    return float(v[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def json_safe(value: object) -> object:
    """Replace non-finite floats with None recursively for strict JSON."""
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


__all__ = [
    "BANDS",
    "BAND_WEIGHTS",
    "band_index",
    "bootstrap_overall",
    "cluster_sums",
    "failed",
    "json_safe",
    "official_check",
    "overall_from_mid",
    "phase_to_ttc",
    "row_mid",
    "summarize",
    "ttc_to_phase",
    "weighted_median",
]
