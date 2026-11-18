"""Label-free Test12 projections (level_gap method) and 12-sequence resolution (numpy only).

All fold quantities are band means of per-row MiD over the 40 TRAIN40 sequences; replicates
resample whole sequences jointly for every arm so ratios are paired.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

BANDS = ("c", "s", "l", "n")
WEIGHTS = np.asarray([0.5, 0.3, 0.1, 0.1])
EDGES = (0.0, 10.0, 20.0, 40.0, 80.0)  # level_gap.addenda disagreement bins (MiD)
OFFICIAL = {
    "REL": {"c": 43.52, "s": 28.26, "l": 25.72, "n": 29.93, "overall": 35.80},
    "RGB_GEO_BAND": {"c": 24.37, "s": 23.95, "l": 22.89, "n": 18.13, "overall": 23.47},
    "GARL_PAPER": {"c": 53.1, "s": 37.6, "l": 40.6, "n": 31.3, "overall": 45.03},
}
FloatArray = NDArray[np.float64]


def cluster_band_sums(
    mid: NDArray[Any], bands: NDArray[Any], clusters: NDArray[Any]
) -> tuple[FloatArray, FloatArray]:
    """Per-cluster finite MiD sums and counts by target band ``[C, 4]`` (sorted cluster ids)."""
    _, inverse = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    v = np.asarray(mid, dtype=np.float64)
    b = np.asarray(bands)
    ok = np.isfinite(v) & (b >= 0)
    n_clusters = int(inverse.max()) + 1
    sums = np.zeros((n_clusters, 4))
    counts = np.zeros((n_clusters, 4))
    np.add.at(sums, (inverse[ok], b[ok]), v[ok])
    np.add.at(counts, (inverse[ok], b[ok]), 1.0)
    return sums, counts


def draws(n_clusters: int, replicates: int, seed: int) -> FloatArray:
    """Multiplicity matrix ``[R, C]`` of a cluster bootstrap with replacement."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_clusters, (replicates, n_clusters))
    counts = np.zeros((replicates, n_clusters))
    np.add.at(counts, (np.repeat(np.arange(replicates), n_clusters), picks.ravel()), 1.0)
    return counts


def band_means(
    sums: FloatArray, counts: FloatArray, weights: FloatArray | None = None
) -> FloatArray:
    """Band means of the pooled clusters (``weights`` ``[R, C]`` gives replicates ``[R, 4]``)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        if weights is None:
            return sums.sum(0) / counts.sum(0)
        return (weights @ sums) / (weights @ counts)


def overall(bands: FloatArray) -> FloatArray:
    """Official Overall of band-mean rows ``[..., 4]``."""
    return np.asarray(bands) @ WEIGHTS


def multiplicative(
    variant_mid: NDArray[Any],
    anchor_mid: NDArray[Any],
    bands: NDArray[Any],
    clusters: NDArray[Any],
    anchor: str,
    *,
    replicates: int = 2000,
    seed: int = 20261118,
) -> dict[str, Any]:
    """Projected Overall = sum_b w_b * official_anchor_b * fold_variant_b / fold_anchor_b."""
    official = np.asarray([OFFICIAL[anchor][b] for b in BANDS])
    vs, vc = cluster_band_sums(variant_mid, bands, clusters)
    as_, ac = cluster_band_sums(anchor_mid, bands, clusters)
    point_ratio = band_means(vs, vc) / band_means(as_, ac)
    weights = draws(len(vs), replicates, seed)
    ratio = band_means(vs, vc, weights) / band_means(as_, ac, weights)
    projected = overall(official * ratio)
    finite = projected[np.isfinite(projected)]
    return {
        "anchor": anchor,
        "point": float(overall(official * point_ratio)),
        "ci95": [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))],
        "band_point": {b: float(official[i] * point_ratio[i]) for i, b in enumerate(BANDS)},
        "ratio_point": {b: float(point_ratio[i]) for i, b in enumerate(BANDS)},
        "replicates": replicates,
    }


def predicted_group(ttc: NDArray[Any]) -> NDArray[np.int64]:
    """level_gap ``group4(predicted_band)``: (0,3]->c, (3,6]->s, (6,inf)->l, <=0->n."""
    t = np.asarray(ttc, dtype=np.float64)
    out = np.full(t.shape, 3, dtype=np.int64)
    out[(t > 0) & (t <= 3)] = 0
    out[(t > 3) & (t <= 6)] = 1
    out[t > 6] = 2
    return out


def phase_mid(a_ttc: NDArray[Any], b_ttc: NDArray[Any]) -> FloatArray:
    """|phase(a) - phase(b)| x 1e4 with phase = -log(1 - 0.1/T) (NaN where undefined)."""
    a = np.asarray(a_ttc, dtype=np.float64)
    b = np.asarray(b_ttc, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        pa = -np.log(1.0 - 0.1 / a)
        pb = -np.log(1.0 - 0.1 / b)
    return np.abs(pa - pb) * 1e4


def additive_disagreement(
    fold_variant_ttc: NDArray[Any],
    fold_anchor_ttc: NDArray[Any],
    truth_ttc: NDArray[Any],
    bands: NDArray[Any],
    clusters: NDArray[Any],
    test_variant_ttc: NDArray[Any],
    test_anchor_ttc: NDArray[Any],
    test_mass: NDArray[Any],
    anchor: str = "RGB_GEO_BAND",
    *,
    replicates: int = 2000,
    seed: int = 20261118,
) -> dict[str, Any]:
    """level_gap A3: official_b + sum_bin P_test(bin | anchor-predicted group b) * fold change.

    The fold change is the mean |error| change (MiD) of the variant versus the anchor inside
    (target band b, disagreement bin) cells; the test weights use the disagreement between the
    two test prediction sets within each group predicted by the anchor package.
    """
    phase_true = -np.log(1.0 - 0.1 / np.asarray(truth_ttc, dtype=np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        e_new = np.abs(-np.log(1.0 - 0.1 / np.asarray(fold_variant_ttc)) - phase_true) * 1e4
        e_old = np.abs(-np.log(1.0 - 0.1 / np.asarray(fold_anchor_ttc)) - phase_true) * 1e4
    change = e_new - e_old
    bins = np.digitize(
        np.nan_to_num(phase_mid(fold_variant_ttc, fold_anchor_ttc), nan=1e9), EDGES[1:]
    )
    b = np.asarray(bands)
    ok = np.isfinite(change) & (b >= 0)
    n_cells = 4 * len(EDGES)
    cell = b * len(EDGES) + bins
    _, inverse = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    sums = np.zeros((n_clusters, n_cells))
    mass = np.zeros((n_clusters, n_cells))
    np.add.at(sums, (inverse[ok], cell[ok]), change[ok])
    np.add.at(mass, (inverse[ok], cell[ok]), 1.0)
    groups = predicted_group(test_anchor_ttc)
    test_bins = np.digitize(
        np.nan_to_num(phase_mid(test_variant_ttc, test_anchor_ttc), nan=1e9), EDGES[1:]
    )
    weight = np.asarray(test_mass, dtype=np.float64)
    p_test = np.zeros((4, len(EDGES)))
    for g in range(4):
        keep = groups == g
        for i in range(len(EDGES)):
            p_test[g, i] = weight[keep & (test_bins == i)].sum() / max(weight[keep].sum(), 1e-12)
    official = np.asarray([OFFICIAL[anchor][x] for x in BANDS])

    def project(cell_sum: FloatArray, cell_mass: FloatArray) -> FloatArray:
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.nan_to_num(cell_sum / cell_mass).reshape(*cell_sum.shape[:-1], 4, len(EDGES))
        return overall(np.maximum(official + (means * p_test).sum(-1), 0.0))

    weights = draws(n_clusters, replicates, seed)
    reps = project(weights @ sums, weights @ mass)
    return {
        "anchor": anchor,
        "point": float(project(sums.sum(0), mass.sum(0))),
        "ci95": [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))],
        "test_bin_share_by_group": {x: p_test[i].tolist() for i, x in enumerate(BANDS)},
        "replicates": replicates,
    }


def resolution12(
    delta_mid: NDArray[Any],
    bands: NDArray[Any],
    clusters: NDArray[Any],
    *,
    size: int = 12,
    replicates: int = 10_000,
    seed: int = 20261118,
) -> dict[str, float]:
    """SD and 95 % range of the paired Overall difference over random ``size``-sequence tests."""
    sums, counts = cluster_band_sums(delta_mid, bands, clusters)
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(sums)))
    for r in range(replicates):
        weights[r, rng.choice(len(sums), size=size, replace=False)] = 1.0
    values = overall(band_means(sums, counts, weights))
    values = values[np.isfinite(values)]
    return {
        "sd": float(values.std(ddof=1)),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
        "defined_replicates": int(len(values)),
    }


__all__ = [
    "EDGES",
    "OFFICIAL",
    "additive_disagreement",
    "band_means",
    "cluster_band_sums",
    "draws",
    "multiplicative",
    "overall",
    "phase_mid",
    "predicted_group",
    "resolution12",
]
