"""Official-form GarlTTC MiD, banded aggregation and the paired sequence bootstrap."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

DT = 0.1
BANDS = ("crucial", "small", "large", "negative")
BAND_WEIGHTS = {"crucial": 0.5, "small": 0.3, "large": 0.1, "negative": 0.1}


def band_index(target_ttc: np.ndarray) -> np.ndarray:
    """Return 0..3 for crucial/small/large/negative bands and -1 outside all bands."""
    t = np.asarray(target_ttc, dtype=np.float64)
    out = np.full(t.shape, -1, dtype=np.int64)
    out[(t > 0) & (t <= 3)] = 0
    out[(t > 3) & (t <= 6)] = 1
    out[(t > 6) & (t <= 10)] = 2
    out[(t > -10) & (t <= 0)] = 3
    return out


def mid_per_query(target_ttc: np.ndarray, predicted_ttc: np.ndarray) -> np.ndarray:
    """Per-query MiD as in the official ``evaluate_rows`` (NaN where it is undefined)."""
    gt = np.asarray(target_ttc, dtype=np.float64)
    pred = np.asarray(predicted_ttc, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pred_ratio = np.where(pred != 0.0, 1.0 - DT / pred, np.nan)
        gt_ratio = 1.0 - DT / gt
        ok = (pred_ratio > 0) & (gt_ratio > 0) & np.isfinite(pred_ratio)
        mid = np.where(
            ok,
            np.abs(np.log(np.where(ok, gt_ratio, 1.0)) - np.log(np.where(ok, pred_ratio, 1.0)))
            * 1e4,
            np.nan,
        )
    return mid


def summarize(target_ttc: np.ndarray, predicted_ttc: np.ndarray) -> dict[str, float]:
    """Overall/banded MiD (official pooling), seconds MAE overall and per band."""
    gt = np.asarray(target_ttc, dtype=np.float64)
    pred = np.asarray(predicted_ttc, dtype=np.float64)
    mid = mid_per_query(gt, pred)
    bands = band_index(gt)
    abs_err = np.abs(pred - gt)
    out: dict[str, float] = {}
    overall = 0.0
    for index, name in enumerate(BANDS):
        selected = bands == index
        values = mid[selected]
        values = values[np.isfinite(values)]
        out[f"MiD_{name}"] = float(values.mean()) if values.size else math.nan
        out[f"MAE_s_{name}"] = float(abs_err[selected].mean()) if selected.any() else math.nan
        out[f"n_{name}"] = int(selected.sum())
        overall += BAND_WEIGHTS[name] * out[f"MiD_{name}"]
    out["Overall_MiD"] = overall
    finite = mid[np.isfinite(mid)]
    out["mean_MiD"] = float(finite.mean()) if finite.size else math.nan
    out["MAE_s"] = float(abs_err.mean())
    out["n"] = int(len(gt))
    out["n_nan_MiD"] = int((~np.isfinite(mid)).sum())
    return out


def sequence_sums(
    target_ttc: np.ndarray, predicted_ttc: np.ndarray, group: np.ndarray, groups: Sequence[str]
) -> dict[str, np.ndarray]:
    """Per-sequence, per-band sums of finite MiD, finite counts and absolute seconds error.

    Shapes are ``[G, 4]`` (bands) plus ``[G]`` totals for MAE. Rows outside all bands
    contribute only to the MAE totals.
    """
    gt = np.asarray(target_ttc, dtype=np.float64)
    pred = np.asarray(predicted_ttc, dtype=np.float64)
    mid = mid_per_query(gt, pred)
    bands = band_index(gt)
    lookup = {name: index for index, name in enumerate(groups)}
    gidx = np.asarray([lookup[str(name)] for name in np.asarray(group).astype(str)])
    g = len(groups)
    mid_sum = np.zeros((g, 4))
    mid_count = np.zeros((g, 4))
    abs_sum = np.zeros((g, 4))
    row_count = np.zeros((g, 4))
    finite = np.isfinite(mid)
    for index in range(4):
        selected = bands == index
        np.add.at(mid_sum[:, index], gidx[selected & finite], mid[selected & finite])
        np.add.at(mid_count[:, index], gidx[selected & finite], 1.0)
        np.add.at(abs_sum[:, index], gidx[selected], np.abs(pred - gt)[selected])
        np.add.at(row_count[:, index], gidx[selected], 1.0)
    total_abs = np.zeros(g)
    total_rows = np.zeros(g)
    np.add.at(total_abs, gidx, np.abs(pred - gt))
    np.add.at(total_rows, gidx, 1.0)
    return {
        "mid_sum": mid_sum,
        "mid_count": mid_count,
        "abs_sum": abs_sum,
        "row_count": row_count,
        "total_abs": total_abs,
        "total_rows": total_rows,
    }


def bootstrap_weights(n_groups: int, resamples: int, seed: int) -> np.ndarray:
    """Multiplicity matrix ``[B, G]`` of a with-replacement sequence bootstrap."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(resamples, n_groups))
    weights = np.zeros((resamples, n_groups))
    np.add.at(weights, (np.repeat(np.arange(resamples), n_groups), draws.reshape(-1)), 1.0)
    return weights


def resampled_metrics(sums: Mapping[str, np.ndarray], weights: np.ndarray) -> dict[str, np.ndarray]:
    """Metric vectors ``[B]`` for every bootstrap replicate (ratio estimators)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        band_mid = (weights @ sums["mid_sum"]) / (weights @ sums["mid_count"])
        band_mae = (weights @ sums["abs_sum"]) / (weights @ sums["row_count"])
        mae = (weights @ sums["total_abs"]) / (weights @ sums["total_rows"])
    out = {"Overall_MiD": sum(BAND_WEIGHTS[b] * band_mid[:, i] for i, b in enumerate(BANDS))}
    for index, name in enumerate(BANDS):
        out[f"MiD_{name}"] = band_mid[:, index]
        out[f"MAE_s_{name}"] = band_mae[:, index]
    out["MAE_s"] = mae
    return out


def point_metrics(sums: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Full-sample metrics from sequence sums (identical to the pooled official form)."""
    replicate = resampled_metrics(sums, np.ones((1, len(sums["total_rows"]))))
    return {key: float(value[0]) for key, value in replicate.items()}


def interval(values: np.ndarray, level: float = 0.95) -> dict[str, float]:
    """Percentile interval over finite replicates, with the dropped-replicate count."""
    finite = values[np.isfinite(values)]
    alpha = (1 - level) / 2
    return {
        "lo": float(np.quantile(finite, alpha)),
        "hi": float(np.quantile(finite, 1 - alpha)),
        "replicates": int(finite.size),
        "dropped_nonfinite": int(values.size - finite.size),
    }


__all__ = [
    "BANDS",
    "BAND_WEIGHTS",
    "band_index",
    "bootstrap_weights",
    "interval",
    "mid_per_query",
    "point_metrics",
    "resampled_metrics",
    "sequence_sums",
    "summarize",
]
