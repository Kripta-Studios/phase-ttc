"""Pure numerical helpers of the level-gap study (numpy only, no file access)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from operational.phase_geometry.metrics import BAND_WEIGHTS, BANDS, WEIGHT_VECTOR

FloatArray = NDArray[np.float64]
PRED_BAND_NAMES = ("pc", "ps", "pl", "pfar", "pn", "pfarneg")


def predicted_band(ttc: NDArray[Any]) -> NDArray[np.int64]:
    """Six predicted-TTC strata: (0,3], (3,6], (6,10], (10,inf), (-10,0], (-inf,-10]."""
    t = np.asarray(ttc, dtype=np.float64)
    out = np.full(t.shape, 5, dtype=np.int64)
    out[(t > 0) & (t <= 3)] = 0
    out[(t > 3) & (t <= 6)] = 1
    out[(t > 6) & (t <= 10)] = 2
    out[t > 10] = 3
    out[(t > -10) & (t <= 0)] = 4
    return out


def weighted_overall(
    mid: NDArray[Any], bands: NDArray[Any], weights: NDArray[Any] | None = None
) -> dict[str, float]:
    """Band means (row weights within each target band) and the official Overall."""
    m = np.asarray(mid, dtype=np.float64)
    b = np.asarray(bands)
    w = np.ones_like(m) if weights is None else np.asarray(weights, dtype=np.float64)
    out: dict[str, float] = {}
    for index, name in enumerate(BANDS):
        keep = (b == index) & np.isfinite(m) & (w > 0)
        total = float(w[keep].sum())
        out[f"MiD{name}"] = float((w[keep] * m[keep]).sum() / total) if total > 0 else np.nan
    out["overall"] = float(sum(BAND_WEIGHTS[n] * out[f"MiD{n}"] for n in BANDS))
    return out


def effective_sample_size(weights: NDArray[Any]) -> float:
    """Kish effective sample size of non-negative weights."""
    w = np.asarray(weights, dtype=np.float64)
    return float(w.sum() ** 2 / (w * w).sum()) if (w * w).sum() > 0 else 0.0


def cell_keys(columns: Sequence[NDArray[Any]], radices: Sequence[int]) -> NDArray[np.int64]:
    """Integer cell id of several small-integer columns with fixed radices (shared by samples)."""
    if len(columns) != len(radices):
        raise ValueError("one radix per column")
    key = np.zeros(len(columns[0]), dtype=np.int64)
    for column, radix in zip(columns, radices, strict=True):
        c = np.asarray(column, dtype=np.int64)
        if c.min(initial=0) < 0 or c.max(initial=0) >= radix:
            raise ValueError("cell value outside its radix")
        key = key * radix + c
    return key


def stratified_weights(
    train_cells: NDArray[Any], test_cells: NDArray[Any], test_mass: NDArray[Any] | None = None
) -> tuple[FloatArray, dict[str, float]]:
    """Per-train-row weights p_test(cell)/p_train(cell), normalized to mean 1 over train rows.

    Test cells without any train row cannot be represented; their mass is reported.
    """
    train = np.asarray(train_cells)
    test = np.asarray(test_cells)
    mass = np.ones(len(test)) if test_mass is None else np.asarray(test_mass, dtype=np.float64)
    cells, train_inverse, train_counts = np.unique(train, return_inverse=True, return_counts=True)
    test_share = {c: 0.0 for c in cells.tolist()}
    missing = 0.0
    for cell, value in zip(test.tolist(), mass.tolist(), strict=True):
        if cell in test_share:
            test_share[cell] += value
        else:
            missing += value
    total = mass.sum()
    test_p = np.asarray([test_share[c] / total for c in cells.tolist()])
    ratio = test_p / (train_counts / len(train))
    weights = ratio[train_inverse]
    weights = weights / weights.mean()
    info = {
        "cells_train": float(len(cells)),
        "cells_test": float(len(np.unique(test))),
        "test_mass_unrepresented": float(missing / total),
        "ess": effective_sample_size(weights),
        "max_weight": float(weights.max()),
    }
    return weights, info


def logistic_weights(
    x_train: NDArray[Any],
    x_test: NDArray[Any],
    test_mass: NDArray[Any] | None = None,
    *,
    clip_quantile: float = 0.99,
    seed: int = 20261118,
) -> tuple[FloatArray, dict[str, float]]:
    """Domain-classifier density ratio p(test|x)/p(train|x), clipped and mean-normalized."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    xtr = np.asarray(x_train, dtype=np.float64)
    xte = np.asarray(x_test, dtype=np.float64)
    mass = np.ones(len(xte)) if test_mass is None else np.asarray(test_mass, dtype=np.float64)
    scaler = StandardScaler().fit(np.concatenate((xtr, xte)))
    x = scaler.transform(np.concatenate((xtr, xte)))
    y = np.concatenate((np.zeros(len(xtr)), np.ones(len(xte))))
    sample = np.concatenate((np.full(len(xtr), 1.0 / len(xtr)), mass / mass.sum()))
    sample = sample * len(y) / sample.sum()
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    model.fit(x, y, sample_weight=sample)
    p = model.predict_proba(scaler.transform(xtr))[:, 1]
    raw = p / np.clip(1.0 - p, 1e-9, None)
    cap = float(np.quantile(raw, clip_quantile))
    weights = np.minimum(raw, cap)
    weights = weights / weights.mean()
    auc = float(roc_auc_score(y, model.predict_proba(x)[:, 1], sample_weight=sample))
    return weights, {"auc": auc, "ess": effective_sample_size(weights), "clip": cap}


def cluster_band_sums(
    values: NDArray[Any], weights: NDArray[Any], bands: NDArray[Any], clusters: NDArray[Any]
) -> tuple[FloatArray, FloatArray]:
    """Per-cluster weighted sums of ``values`` and of weights by target band ``[C,4]``."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    b = np.asarray(bands)
    _, inverse = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    n = int(inverse.max()) + 1
    sums = np.zeros((n, len(BANDS)))
    mass = np.zeros((n, len(BANDS)))
    ok = np.isfinite(v) & (b >= 0)
    np.add.at(sums, (inverse[ok], b[ok]), (w * v)[ok])
    np.add.at(mass, (inverse[ok], b[ok]), w[ok])
    return sums, mass


def bootstrap_band_means(
    named_values: Mapping[str, NDArray[Any]],
    weights: NDArray[Any],
    bands: NDArray[Any],
    clusters: NDArray[Any],
    *,
    replicates: int = 10_000,
    seed: int = 20261118,
) -> dict[str, FloatArray]:
    """Joint cluster bootstrap: band-mean replicates ``[R,4]`` for every named row vector."""
    n = len(np.unique(np.asarray(clusters).astype(str)))
    draws = np.random.default_rng(seed).integers(0, n, (replicates, n))
    counts = np.zeros((replicates, n))
    np.add.at(counts, (np.repeat(np.arange(replicates), n), draws.ravel()), 1.0)
    out = {}
    for name, values in named_values.items():
        sums, mass = cluster_band_sums(values, weights, bands, clusters)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = (counts @ sums) / (counts @ mass)
    return out


def ci(values: NDArray[Any]) -> list[float]:
    """Percentile 95 % interval ignoring non-finite replicates."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def overall_of(band_means: NDArray[Any]) -> FloatArray:
    """Official Overall of band-mean rows ``[...,4]``."""
    return np.asarray(band_means, dtype=np.float64) @ WEIGHT_VECTOR


def project_multiplicative(
    reference_bands: Mapping[str, float], ratio_bands: NDArray[Any]
) -> FloatArray:
    """Projected Overall = sum_b w_b * reference_b * ratio_b (rows of ``ratio_bands``)."""
    reference = np.asarray([reference_bands[b] for b in BANDS], dtype=np.float64)
    return overall_of(np.asarray(ratio_bands, dtype=np.float64) * reference)


def blend_phase(phase_a: NDArray[Any], phase_b: NDArray[Any], weight_a: float) -> FloatArray:
    """Convex phase-space blend ``w*a + (1-w)*b``."""
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError("blend weight must be in [0,1]")
    a = np.asarray(phase_a, dtype=np.float64)
    b = np.asarray(phase_b, dtype=np.float64)
    return weight_a * a + (1.0 - weight_a) * b


def three_cornered_hat(
    a: NDArray[Any], b: NDArray[Any], c: NDArray[Any], *, clip: float
) -> dict[str, float]:
    """Error variances of three estimators from pairwise differences (independent errors).

    Differences are winsorized to ``[-clip, clip]``; returns variances and their square roots
    (negative variance estimates are kept as signed values and flagged by a NaN root).
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    z = np.asarray(c, dtype=np.float64)
    vab = float(np.var(np.clip(x - y, -clip, clip)))
    vac = float(np.var(np.clip(x - z, -clip, clip)))
    vbc = float(np.var(np.clip(y - z, -clip, clip)))
    var = {"a": 0.5 * (vab + vac - vbc), "b": 0.5 * (vab + vbc - vac), "c": 0.5 * (vac + vbc - vab)}
    out: dict[str, float] = {}
    for key, value in var.items():
        out[f"var_{key}"] = value
        out[f"sd_{key}"] = float(np.sqrt(value)) if value > 0 else float("nan")
    return out


def winsorized_sd(error: NDArray[Any], *, clip: float) -> float:
    """Standard deviation of an error vector winsorized to ``[-clip, clip]``."""
    return float(np.std(np.clip(np.asarray(error, dtype=np.float64), -clip, clip)))


def conditional_wins(
    error_new: NDArray[Any],
    error_old: NDArray[Any],
    disagreement: NDArray[Any],
    edges: Sequence[float],
) -> list[dict[str, float]]:
    """Win rate (|e_new| < |e_old|) and mean MiD change per disagreement bin."""
    en = np.abs(np.asarray(error_new, dtype=np.float64))
    eo = np.abs(np.asarray(error_old, dtype=np.float64))
    d = np.asarray(disagreement, dtype=np.float64)
    rows = []
    bounds = [*edges, np.inf]
    for low, high in zip(bounds[:-1], bounds[1:], strict=True):
        keep = (d >= low) & (d < high) & np.isfinite(en) & np.isfinite(eo)
        n = int(keep.sum())
        rows.append(
            {
                "low": float(low),
                "high": float(high),
                "rows": float(n),
                "win_rate": float((en[keep] < eo[keep]).mean()) if n else float("nan"),
                "tie_rate": float((en[keep] == eo[keep]).mean()) if n else float("nan"),
                "mean_change": float((en[keep] - eo[keep]).mean()) if n else float("nan"),
            }
        )
    return rows


def concentration_share(contributions: NDArray[Any], top_fraction: float) -> float:
    """Share of the total carried by the largest ``top_fraction`` of units."""
    v = np.sort(np.asarray(contributions, dtype=np.float64))[::-1]
    if not len(v) or v.sum() <= 0:
        return float("nan")
    k = max(1, int(np.ceil(top_fraction * len(v))))
    return float(v[:k].sum() / v.sum())


def describe(values: NDArray[Any], weights: NDArray[Any] | None = None) -> dict[str, float]:
    """Weighted mean and 5/25/50/75/95 % quantiles."""
    v = np.asarray(values, dtype=np.float64)
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(v)
    v, w = v[keep], w[keep]
    order = np.argsort(v, kind="stable")
    cumulative = np.cumsum(w[order]) / w.sum()
    out = {"mean": float((v * w).sum() / w.sum())}
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        out[f"q{int(q * 100):02d}"] = float(v[order][np.searchsorted(cumulative, q)])
    return out


def shares(labels: NDArray[Any], names: Sequence[str], weights: NDArray[Any] | None = None) -> dict:
    """Weighted share of each integer label 0..len(names)-1."""
    lab = np.asarray(labels)
    w = np.ones(len(lab)) if weights is None else np.asarray(weights, dtype=np.float64)
    return {name: float(w[lab == i].sum() / w.sum()) for i, name in enumerate(names)}


__all__ = [
    "PRED_BAND_NAMES",
    "blend_phase",
    "bootstrap_band_means",
    "cell_keys",
    "ci",
    "cluster_band_sums",
    "concentration_share",
    "conditional_wins",
    "describe",
    "effective_sample_size",
    "logistic_weights",
    "overall_of",
    "predicted_band",
    "project_multiplicative",
    "shares",
    "stratified_weights",
    "three_cornered_hat",
    "weighted_overall",
    "winsorized_sd",
]
