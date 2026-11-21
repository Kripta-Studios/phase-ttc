"""Utilities shared by the A/B/C PHASE-TTC-Geo reproduction runs.

The helpers deliberately report prediction drift only.  They never infer an
official score because Test12 targets are private.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def compare_predictions(
    candidate: ArrayLike,
    reference: ArrayLike,
    *,
    sequence: ArrayLike,
    canonical: ArrayLike,
    tolerance_seconds: float,
) -> dict[str, Any]:
    """Summarize row drift, alias coverage, and the sequence with the worst row."""
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    sequences = np.asarray(sequence).astype(str)
    identities = np.asarray(canonical).astype(str)
    if not (left.shape == right.shape == sequences.shape == identities.shape):
        raise ValueError("Prediction, sequence, and canonical arrays must have one shape")
    if left.ndim != 1 or tolerance_seconds < 0:
        raise ValueError("Expected one-dimensional rows and a non-negative tolerance")
    finite = np.isfinite(left) & np.isfinite(right)
    drift = np.full(left.shape, np.inf, dtype=np.float64)
    drift[finite] = np.abs(left[finite] - right[finite])
    worst = int(np.argmax(drift)) if len(drift) else -1
    quantiles = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
    finite_drift = drift[np.isfinite(drift)]
    values = np.quantile(finite_drift, quantiles) if len(finite_drift) else np.full(6, np.nan)
    return {
        "rows": int(len(left)),
        "physical_queries": int(len(set(identities.tolist()))),
        "aliases": int(len(left) - len(set(identities.tolist()))),
        "finite_pairs": int(finite.sum()),
        "failures": int(np.sum(drift > tolerance_seconds)),
        "tolerance_seconds": float(tolerance_seconds),
        "max_abs_ttc_seconds": (
            float(drift[worst]) if worst >= 0 and np.isfinite(drift[worst]) else None
        ),
        "worst_row": worst if worst >= 0 else None,
        "worst_sequence": str(sequences[worst]) if worst >= 0 else None,
        "drift_quantiles_seconds": {
            str(q): float(value) if np.isfinite(value) else None
            for q, value in zip(quantiles, values, strict=True)
        },
    }


__all__ = ["compare_predictions"]
