"""Metrics for the author-informed EvTTC zero-shot reproduction.

The paper's Table VI accuracy measure is plain relative TTC error (RTE). It
must not inherit the signed eAP development scorer's TTC bands or paper MiD
weights. Invalid predictions remain visible as failures and invalidate the
mean rather than disappearing from its denominator.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from e_jepa_ttc.data.evttc_garl_adapter import reject_labels_from_predict_payload


def score_zero_shot_predictions(
    y_true_ttc_s: Iterable[float],
    y_pred_ttc_s: Iterable[float],
) -> dict[str, Any]:
    """Compute strict-coverage plain RTE for an already aligned cohort.

    Zero and negative predictions are numeric model outputs and are scored by
    the same formula as positive predictions. NaN or infinite predictions are
    failures: the aggregate RTE is unavailable, never recomputed on a subset.
    """

    target = np.asarray(list(y_true_ttc_s), dtype=np.float64).reshape(-1)
    prediction = np.asarray(list(y_pred_ttc_s), dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("EvTTC RTE arrays must be non-empty and shape matched.")
    if not np.isfinite(target).all():
        raise ValueError("EvTTC TTC targets must be finite.")
    if np.any(target == 0.0):
        raise ValueError("EvTTC plain RTE is undefined for a zero TTC target.")

    failed = ~np.isfinite(prediction)
    # Divide each operand by |target| before subtraction. This is algebraically
    # identical to |prediction-target|/|target|, while avoiding overflow for
    # large finite values with opposite signs (for example +/-1e308).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        per_sample = np.abs(prediction / np.abs(target) - np.sign(target)) * 100.0
    failed |= ~np.isfinite(per_sample)
    failure_count = int(np.count_nonzero(failed))
    if failure_count:
        rte = None
    else:
        maximum = float(np.max(per_sample))
        # RTE is non-negative. Scaling before reduction prevents np.mean from
        # overflowing its intermediate sum for very large finite errors.
        rte = 0.0 if maximum == 0.0 else float(maximum * np.mean(per_sample / maximum))
        if not np.isfinite(rte):
            rte = None
            failed[:] = True
            failure_count = int(target.size)
    return {
        "protocol": "evttc_plain_rte_strict_coverage_v1",
        "formula": "abs(predicted_ttc_s-target_ttc_s)/abs(target_ttc_s)*100",
        "sample_count": int(target.size),
        "scored_sample_count": int(target.size - failure_count),
        "failure_count": failure_count,
        "failure_rate_pct": float(np.mean(failed) * 100.0),
        "mean_RTE_pct": rte,
        "status": "complete" if failure_count == 0 else "invalid_predictions",
    }


__all__ = ["reject_labels_from_predict_payload", "score_zero_shot_predictions"]
