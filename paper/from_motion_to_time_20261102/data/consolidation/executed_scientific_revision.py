"""Matched frozen-prediction diagnostics with explicit support and recording units."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def align_records(records: list[dict[str, Any]], tokens: Sequence[str]) -> list[dict[str, Any]]:
    """Join exact query tokens, rejecting missing, extra and duplicate identities."""
    lookup = {str(row["token"]): row for row in records}
    if len(lookup) != len(records) or len(set(tokens)) != len(tokens):
        raise ValueError("duplicate query identity")
    if set(lookup) != set(tokens):
        raise ValueError("query identity sets differ")
    return [lookup[token] for token in tokens]


def _emit_rate(rate: np.ndarray) -> np.ndarray:
    sign = np.where(rate < 0, -1.0, 1.0)
    return sign / np.maximum(np.abs(rate), 1 / 60)


def simple_fusion(rgb: np.ndarray, event: np.ndarray, coordinate: str) -> np.ndarray:
    """Combine the same two emitted predictions with a fixed half/half weight."""
    rgb, event = np.asarray(rgb, dtype=float), np.asarray(event, dtype=float)
    if rgb.shape != event.shape or not np.isfinite([rgb, event]).all():
        raise ValueError("Expected matching finite predictions")
    if np.any(rgb == 0) or np.any(event == 0):
        raise ValueError("Zero TTC is outside the prediction domain")
    if coordinate == "ttc":
        return (rgb + event) / 2
    if coordinate == "inverse_rate":
        return _emit_rate((1 / rgb + 1 / event) / 2)
    if coordinate == "phase":
        if np.any((rgb > 0) & (rgb <= 0.1)) or np.any((event > 0) & (event <= 0.1)):
            raise ValueError("Positive TTC must exceed the phase domain boundary")
        phase = (-np.log1p(-0.1 / rgb) - np.log1p(-0.1 / event)) / 2
        phase = np.clip(phase, -np.log(2) + 1e-6, 8)
        return _emit_rate(-np.expm1(-phase) / 0.1)
    raise ValueError(f"Unknown fusion coordinate: {coordinate}")


def geometry_arms(original: np.ndarray, geometry: np.ndarray) -> dict[str, np.ndarray]:
    """Keep geometry abstentions visible and use the same original for both fallbacks."""
    original, geometry = np.asarray(original, float), np.asarray(geometry, float)
    if original.shape != geometry.shape or not np.isfinite(original).all():
        raise ValueError("Expected matching geometry and finite original predictions")
    available = np.isfinite(geometry) & (geometry != 0)
    alone = np.where(available, geometry, np.nan)
    fallback = np.where(available, geometry, original)
    fixed = original.copy()
    fixed[available] = simple_fusion(original[available], geometry[available], "inverse_rate")
    return {
        "geometry_only": alone,
        "geometry_with_original_fallback": fallback,
        "fixed_half_inverse_rate": fixed,
    }


def prediction_summary(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Report errors together with coverage; never silently drop missing predictions."""
    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    if (
        target.ndim != 1
        or target.shape != prediction.shape
        or not np.isfinite(target).all()
        or np.any(target <= 0)
    ):
        raise ValueError("Expected matching predictions and positive finite targets")
    valid = np.isfinite(prediction)
    result: dict[str, Any] = {
        "population": len(target),
        "scored": int(valid.sum()),
        "coverage": float(valid.mean()) if len(valid) else None,
        "negative_predictions": int(np.sum(prediction[valid] < 0)),
    }
    if not valid.any():
        return {**result, **dict.fromkeys(("MAE_s", "median_AE_s", "p95_AE_s", "RTE_pct"))}
    error = np.abs(prediction[valid] - target[valid])
    return {
        **result,
        "MAE_s": float(error.mean()),
        "median_AE_s": float(np.median(error)),
        "p95_AE_s": float(np.quantile(error, 0.95)),
        "RTE_pct": float(np.mean(error / target[valid]) * 100),
    }


def paired_mae(
    target: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int = 10000,
    seed: int = 20261111,
) -> dict[str, Any]:
    """Bootstrap whole recordings and preserve paired query-weighted MAE differences."""
    target, candidate, reference = (np.asarray(a, float) for a in (target, candidate, reference))
    groups = np.asarray(groups).astype(str)
    if (
        target.ndim != 1
        or any(a.shape != target.shape for a in (candidate, reference, groups))
        or not len(target)
        or not np.isfinite([target, candidate, reference]).all()
        or np.any(target <= 0)
    ):
        raise ValueError("Paired comparison requires finite predictions on common support")
    if draws < 1:
        raise ValueError("Bootstrap requires positive draw count")
    delta = np.abs(candidate - target) - np.abs(reference - target)
    names, membership = np.unique(groups, return_inverse=True)
    counts = np.bincount(membership)
    sums = np.bincount(membership, weights=delta)
    result: dict[str, Any] = {
        "delta_MAE_s": float(delta.mean()),
        "recording_macro_delta_MAE_s": float((sums / counts).mean()),
        "recordings": len(names),
        "queries": len(target),
        "resampling_unit": "recording",
        "draws": draws,
        "seed": seed,
        "ci95": None,
        "interpretation": "Descriptive interval on historically inspected recordings",
    }
    if len(names) > 1:
        samples = np.random.default_rng(seed).integers(0, len(names), (draws, len(names)))
        values = sums[samples].sum(1) / counts[samples].sum(1)
        result["ci95"] = np.quantile(values, [0.025, 0.975]).tolist()
    return result
