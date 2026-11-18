"""Numpy-only protocol logic shared by every baseline environment.

Conventions (see ``PROTOCOL.json``): image 1 is the query frame ``t`` and image 2 the frame
``t - delta``; ``rho = log(s(t) / s(t - delta))`` is the log image-scale change, positive when
the object grows (approaches).  The benchmark phase is ``-log(1 - 0.1 / T)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from operational.phase_geometry.metrics import (
    BAND_WEIGHTS,
    BANDS,
    DT,
    MAX_PHASE,
    MIN_PHASE,
    phase_to_ttc,
    weighted_median,
)

CROP_SIZE = 384
MARGIN_FRACTION = 0.25
MIN_EDGE = 8.0
SUBSET_SEED = 20261120
PER_SEQUENCE = 150
INNER_FRACTION = 0.5
TRIM_KEEP = 0.8
SYNTHETIC_SIGMAS = (0.95, 0.98, 1.0, 1.02, 1.05)

FloatArray = NDArray[np.float64]
Box = Sequence[float] | NDArray[Any]
Square = tuple[float, float, float, float]


# --------------------------------------------------------------------------------------------
# Subset
# --------------------------------------------------------------------------------------------


def allocate(counts: Sequence[int], total: int) -> list[int]:
    """Largest-remainder proportional allocation of ``total`` over ``counts`` (ties by order)."""
    available = int(sum(counts))
    if total > available or total < 0:
        raise ValueError(f"Cannot allocate {total} draws from {available} rows")
    if available == 0:
        return [0 for _ in counts]
    quotas = [total * int(c) / available for c in counts]
    base = [math.floor(q) for q in quotas]
    remainder = total - sum(base)
    order = sorted(range(len(counts)), key=lambda i: (-(quotas[i] - base[i]), i))
    for index in order[:remainder]:
        base[index] += 1
    if any(b > int(c) for b, c in zip(base, counts, strict=True)):
        raise AssertionError("Allocation exceeds a stratum")
    return base


def select_subset(
    tokens: Sequence[str],
    sequences: Sequence[str],
    bands: Sequence[int] | NDArray[Any],
    *,
    per_sequence: int = PER_SEQUENCE,
    seed: int = SUBSET_SEED,
) -> NDArray[np.int64]:
    """Row indices of the sequence-stratified, band-proportional subset (manifest order).

    One generator ``default_rng([seed, k])`` per sorted sequence ``k``; a permutation is drawn
    for every band (in order c,s,l,n) whatever its allocation, so smaller ``per_sequence``
    values select prefixes of the same permutations.
    """
    token_array = np.asarray(tokens).astype(str)
    sequence_array = np.asarray(sequences).astype(str)
    band_array = np.asarray(bands, dtype=np.int64)
    if not (len(token_array) == len(sequence_array) == len(band_array)):
        raise ValueError("tokens, sequences and bands must be aligned")
    if len(set(token_array.tolist())) != len(token_array):
        raise ValueError("sample tokens must be unique")
    if ((band_array < 0) | (band_array >= len(BANDS))).any():
        raise ValueError("every row must belong to a scored band")
    selected: list[int] = []
    for k, sequence in enumerate(sorted(set(sequence_array.tolist()))):
        rng = np.random.default_rng([seed, k])
        members = np.flatnonzero(sequence_array == sequence)
        strata = []
        for band in range(len(BANDS)):
            rows = members[band_array[members] == band]
            strata.append(rows[np.argsort(token_array[rows], kind="stable")])
        counts = allocate([len(s) for s in strata], per_sequence)
        for rows, count in zip(strata, counts, strict=True):
            permutation = rng.permutation(len(rows))
            selected.extend(int(i) for i in rows[permutation[:count]])
    return np.asarray(selected, dtype=np.int64)


# --------------------------------------------------------------------------------------------
# Crop geometry
# --------------------------------------------------------------------------------------------


def edge_box(box: Box) -> tuple[float, float, float, float]:
    """Inclusive pixel box ``[x1, y1, x2, y2]`` as edge coordinates ``[x1, y1, x2+1, y2+1)``."""
    x1, y1, x2, y2 = (float(v) for v in box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 < x1 or y2 < y1:
        raise ValueError(f"Invalid box {box}")
    return x1, y1, x2 + 1.0, y2 + 1.0


def common_square(
    boxes: Sequence[Box], *, margin_fraction: float = MARGIN_FRACTION, min_edge: float = MIN_EDGE
) -> Square:
    """Square centred on the union of ``boxes`` with side ``max(w, h, min_edge)*(1+2m)``."""
    edges = [edge_box(b) for b in boxes]
    x0 = min(e[0] for e in edges)
    y0 = min(e[1] for e in edges)
    x1 = max(e[2] for e in edges)
    y1 = max(e[3] for e in edges)
    side = max(x1 - x0, y1 - y0, min_edge) * (1.0 + 2.0 * margin_fraction)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.5 * side
    return cx - half, cy - half, cx + half, cy + half


def zoom_square(square: Square, centre: tuple[float, float], sigma: float) -> Square:
    """Square ``sigma`` times smaller about ``centre``: content appears ``sigma`` times larger."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    cx, cy = centre
    x0, y0, x1, y1 = square
    return (
        cx + (x0 - cx) / sigma,
        cy + (y0 - cy) / sigma,
        cx + (x1 - cx) / sigma,
        cy + (y1 - cy) / sigma,
    )


def box_in_crop(box: Box, square: Square, size: int = CROP_SIZE) -> FloatArray:
    """Edge-coordinate box mapped to crop pixel-edge coordinates (``size`` px per square side)."""
    x0, y0, x1, y1 = edge_box(box)
    sx0, sy0, sx1, sy1 = square
    kx = size / (sx1 - sx0)
    ky = size / (sy1 - sy0)
    return np.asarray([(x0 - sx0) * kx, (y0 - sy0) * ky, (x1 - sx0) * kx, (y1 - sy0) * ky])


def region_mask(
    crop_box: Sequence[float] | NDArray[Any], size: int = CROP_SIZE, *, inner: bool = False
) -> NDArray[np.bool_]:
    """Crop pixels whose centre lies inside ``crop_box`` (optionally its central 50 %)."""
    u0, v0, u1, v1 = (float(v) for v in crop_box)
    if inner:
        cu, cv = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        hu, hv = 0.5 * INNER_FRACTION * (u1 - u0), 0.5 * INNER_FRACTION * (v1 - v0)
        u0, u1, v0, v1 = cu - hu, cu + hu, cv - hv, cv + hv
    centres = np.arange(size, dtype=np.float64) + 0.5
    cols = (centres >= u0) & (centres < u1)
    rows = (centres >= v0) & (centres < v1)
    return rows[:, None] & cols[None, :]


def truncated(box: Box, width: int = 1280, height: int = 720) -> bool:
    """Box touching the image border (same rule as the geometry study)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return bool(x1 <= 0 or y1 <= 0 or x2 >= width - 1 or y2 >= height - 1)


def box_scale(box: Box) -> float:
    """Geometric-mean box size ``sqrt(w*h)`` with inclusive widths."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return math.sqrt((x2 - x1 + 1.0) * (y2 - y1 + 1.0))


# --------------------------------------------------------------------------------------------
# Native output -> rho -> phase
# --------------------------------------------------------------------------------------------


def box_ratio_rho(box_previous: Box, box_current: Box) -> float:
    """Two-frame box scale ratio ``log(s(t)/s(t-delta))``."""
    return math.log(box_scale(box_current) / box_scale(box_previous))


def rho_from_depth_ratio(depth_ratio: NDArray[Any]) -> FloatArray:
    """``Z2/Z1`` with image 2 = earlier frame equals ``s(t)/s(t-delta)``: ``rho = log(Z2/Z1)``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.asarray(depth_ratio, dtype=np.float64))


def rho_to_phase_cv(rho: NDArray[Any] | float, delta_s: NDArray[Any] | float) -> FloatArray:
    """Constant-velocity conversion: ``r = 1 + delta/T`` -> ``-log(1 - (0.1/delta)(r - 1))``."""
    r_minus_one = np.expm1(np.asarray(rho, dtype=np.float64))
    argument = 1.0 - (DT / np.asarray(delta_s, dtype=np.float64)) * r_minus_one
    with np.errstate(divide="ignore", invalid="ignore"):
        phase = -np.log(np.where(argument > 0, argument, 1.0))
    return np.where(argument > 0, phase, MAX_PHASE)


def rho_to_phase_exp(rho: NDArray[Any] | float, delta_s: NDArray[Any] | float) -> FloatArray:
    """Constant scale-rate conversion: ``phase = (0.1/delta) * rho``."""
    return (DT / np.asarray(delta_s, dtype=np.float64)) * np.asarray(rho, dtype=np.float64)


def ttc_to_rho_cv(ttc: float, delta_s: float) -> float:
    """Inverse of the constant-velocity model, for tests: ``rho = log(1 + delta/T)``."""
    return math.log1p(delta_s / ttc)


def clean_rho(rho: NDArray[Any]) -> tuple[NDArray[Any], NDArray[np.bool_]]:
    """Replace non-finite rho by 0 (T = +60 s after decoding) and flag those failures."""
    values = np.asarray(rho, dtype=np.float64)
    failed = ~np.isfinite(values)
    return np.where(failed, 0.0, values), failed


# --------------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------------


def aggregate(values: NDArray[Any], mask: NDArray[np.bool_], how: str = "median") -> float:
    """Robust statistic of a dense map inside ``mask`` (finite values only; NaN if empty)."""
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return float("nan")
    if how == "median":
        return float(np.median(selected))
    if how == "mean":
        return float(selected.mean())
    raise ValueError(f"Unknown aggregation {how}")


def _affine_fit(source: NDArray[Any], target: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    design = np.column_stack([source, np.ones(len(source))])
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = np.linalg.norm(design @ solution - target, axis=1)
    return solution[:2].T, residual


def affine_rho(flow: NDArray[Any], mask: NDArray[np.bool_], keep: float = TRIM_KEEP) -> float:
    """``rho = -0.5 log|det M|`` of a trimmed affine fit ``x2 = M x1 + c`` to a flow field.

    ``flow`` is ``[2, H, W]`` (dx, dy) from image 1 (query frame) to image 2 (earlier frame), so
    ``det M = (s(t-delta)/s(t))**2``.
    """
    field = np.asarray(flow, dtype=np.float64)
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size < 6:
        return float("nan")
    source = np.column_stack([cols + 0.5, rows + 0.5])
    target = source + np.column_stack([field[0, rows, cols], field[1, rows, cols]])
    finite = np.isfinite(target).all(axis=1)
    source, target = source[finite], target[finite]
    if len(source) < 6:
        return float("nan")
    matrix, residual = _affine_fit(source, target)
    retained = residual <= np.quantile(residual, keep)
    if retained.sum() >= 6:
        matrix, _ = _affine_fit(source[retained], target[retained])
    determinant = float(np.linalg.det(matrix))
    if not math.isfinite(determinant) or determinant <= 0:
        return float("nan")
    return -0.5 * math.log(determinant)


# --------------------------------------------------------------------------------------------
# One-parameter calibration
# --------------------------------------------------------------------------------------------


def band_mass(bands: NDArray[Any]) -> FloatArray:
    """Per-row weight ``w_band / n_band`` with counts from the given rows."""
    b = np.asarray(bands, dtype=np.int64)
    weights = np.zeros(len(b))
    for index, name in enumerate(BANDS):
        member = b == index
        if member.any():
            weights[member] = BAND_WEIGHTS[name] / member.sum()
    return weights


def fit_scale(phase_true: NDArray[Any], phase_hat: NDArray[Any], bands: NDArray[Any]) -> float:
    """``argmin_a sum w_i |phi_i - a phi_hat_i|`` (exact weighted median of the ratios)."""
    phi = np.asarray(phase_true, dtype=np.float64)
    hat = np.asarray(phase_hat, dtype=np.float64)
    weights = band_mass(bands) * np.abs(hat)
    usable = (hat != 0) & np.isfinite(hat) & np.isfinite(phi)
    if not usable.any():
        return 1.0
    return weighted_median(phi[usable] / hat[usable], weights[usable])


def cross_fitted_scale(
    phase_true: NDArray[Any], phase_hat: NDArray[Any], bands: NDArray[Any], folds: NDArray[Any]
) -> tuple[FloatArray, dict[int, float]]:
    """Grouped cross-fitting: each fold is scaled by ``a`` fitted on the other folds."""
    fold_array = np.asarray(folds, dtype=np.int64)
    hat = np.asarray(phase_hat, dtype=np.float64)
    calibrated = np.full(len(hat), np.nan)
    scales: dict[int, float] = {}
    for fold in sorted(set(fold_array.tolist())):
        train = fold_array != fold
        a = fit_scale(np.asarray(phase_true)[train], hat[train], np.asarray(bands)[train])
        scales[int(fold)] = a
        calibrated[~train] = a * hat[~train]
    return calibrated, scales


__all__ = [
    "CROP_SIZE",
    "DT",
    "MAX_PHASE",
    "MIN_PHASE",
    "PER_SEQUENCE",
    "SUBSET_SEED",
    "SYNTHETIC_SIGMAS",
    "affine_rho",
    "aggregate",
    "allocate",
    "band_mass",
    "box_in_crop",
    "box_ratio_rho",
    "box_scale",
    "clean_rho",
    "common_square",
    "cross_fitted_scale",
    "edge_box",
    "fit_scale",
    "phase_to_ttc",
    "region_mask",
    "rho_from_depth_ratio",
    "rho_to_phase_cv",
    "rho_to_phase_exp",
    "select_subset",
    "truncated",
    "ttc_to_rho_cv",
    "zoom_square",
]
