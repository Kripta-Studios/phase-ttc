"""Signed Garl-TTC metrics and validation-only checkpoint selection."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

PAPER_MID_WEIGHTS = {"crucial": 0.5, "small": 0.3, "large": 0.1, "negative": 0.1}
BUCKETS = (
    ("crucial", 0.0, 3.0),
    ("small", 3.0, 6.0),
    ("large", 6.0, 10.0),
    ("negative", -10.0, 0.0),
)


def _bucket_mask(values: np.ndarray, lower: float, upper: float, name: str) -> np.ndarray:
    if name == "negative":
        # This is the exact right-closed ``pd.cut`` interval used by the
        # official release: (-10, 0].  In particular, zero is negative in
        # the signed protocol and -10 is outside the protocol domain.
        return (values > lower) & (values <= upper)
    return (values > lower) & (values <= upper)


def signed_garl_metrics(
    y_true_ttc_s: Iterable[float] | np.ndarray,
    y_pred_ttc_s: Iterable[float] | np.ndarray,
    *,
    delta_t_s: float = 0.1,
) -> dict[str, Any]:
    """Compute MiD/RTE/FR with the official signed TTC domains."""

    if not np.isfinite(delta_t_s) or delta_t_s <= 0.0:
        raise ValueError("delta_t_s must be finite and positive.")
    target = np.asarray(y_true_ttc_s, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred_ttc_s, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("Signed Garl TTC arrays must be non-empty and shape matched.")
    if not np.isfinite(target).all():
        raise ValueError("Signed Garl TTC targets must be finite.")
    failed_prediction = ~np.isfinite(prediction) | (np.abs(prediction) < 0.1)
    failure_count = int(np.count_nonzero(failed_prediction))
    bins: dict[str, dict[str, float | int]] = {}
    weighted_mid = 0.0
    weighted_rte = 0.0
    missing_bucket_count = 0
    invalid_mid_count = 0
    assigned = np.zeros(target.shape, dtype=bool)
    invalid_mid = np.zeros(target.shape, dtype=bool)
    for name, lower, upper in BUCKETS:
        selected = _bucket_mask(target, lower, upper, name)
        assigned |= selected
        count = int(np.count_nonzero(selected))
        if count == 0:
            bins[name] = {
                "count": 0,
                "mid": float("nan"),
                "rte_pct": float("nan"),
                "failure_count": 0,
                "failure_rate_pct": float("nan"),
                "invalid_mid_count": 0,
            }
            missing_bucket_count += 1
            continue
        truth = target[selected]
        estimate = prediction[selected]
        with np.errstate(divide="ignore", invalid="ignore"):
            truth_eta = 1.0 - delta_t_s / truth
            estimate_eta = np.where(estimate != 0.0, 1.0 - delta_t_s / estimate, np.nan)
        failed = ~np.isfinite(estimate) | (np.abs(estimate) < 0.1)
        # Preserve the release-compatible finite-only diagnostic mean.  The
        # explicit invalid counter below prevents selection from treating a
        # conditionally finite bucket mean as complete validation evidence.
        with np.errstate(divide="ignore", invalid="ignore"):
            mid_per_sample = np.abs(np.log(truth_eta) - np.log(estimate_eta)) * 1e4
            rte_per_sample = np.abs(estimate - truth) / np.abs(truth) * 100.0
        finite_mid = mid_per_sample[np.isfinite(mid_per_sample)]
        mid = float(np.mean(finite_mid)) if finite_mid.size else float("nan")
        bucket_invalid_mid_count = int(np.count_nonzero(~np.isfinite(mid_per_sample)))
        invalid_mid[selected] = ~np.isfinite(mid_per_sample)
        invalid_mid_count += bucket_invalid_mid_count
        valid_rte = ~failed
        rte = float(np.mean(rte_per_sample[valid_rte])) if np.any(valid_rte) else float("nan")
        bins[name] = {
            "count": count,
            "mid": mid,
            "rte_pct": rte,
            "failure_count": int(np.count_nonzero(failed)),
            "failure_rate_pct": float(np.mean(failed) * 100.0),
            "invalid_mid_count": bucket_invalid_mid_count,
        }
        weighted_mid += PAPER_MID_WEIGHTS[name] * mid
        weighted_rte += PAPER_MID_WEIGHTS[name] * rte
    complete_bucket_coverage = missing_bucket_count == 0
    assigned_sample_count = int(np.count_nonzero(assigned))
    out_of_domain_count = int(target.size - assigned_sample_count)
    complete_cohort_accounting = complete_bucket_coverage and out_of_domain_count == 0
    invalid_row = failed_prediction | invalid_mid
    invalid_row_count = int(np.count_nonzero(invalid_row))
    if not complete_bucket_coverage:
        weighted_mid = float("nan")
        weighted_rte = float("nan")
    return {
        "protocol": "garl_signed_v1",
        "delta_t_s": delta_t_s,
        "num_samples": int(target.size),
        "bins": bins,
        # Missing buckets make the weighted score unavailable. Invalid rows
        # remain visible through the counters while the release-compatible
        # finite-only bucket means stay available as diagnostics.
        "paper_MiD_overall": weighted_mid,
        "weighted_RTE_pct": weighted_rte,
        "missing_bucket_count": missing_bucket_count,
        "complete_bucket_coverage": complete_bucket_coverage,
        "assigned_sample_count": assigned_sample_count,
        "out_of_domain_count": out_of_domain_count,
        "complete_cohort_accounting": complete_cohort_accounting,
        "invalid_mid_count": invalid_mid_count,
        "invalid_row_count": invalid_row_count,
        "invalid_row_rate_pct": float(np.mean(invalid_row) * 100.0),
        "strict_metric_coverage": complete_cohort_accounting and invalid_row_count == 0,
        "failure_count": failure_count,
        "failure_rate_pct": float(np.mean(failed_prediction) * 100.0),
    }


def sequence_macro_signed_metrics(
    y_true_ttc_s: np.ndarray,
    y_pred_ttc_s: np.ndarray,
    sequence_ids: Iterable[str],
    *,
    delta_t_s: float = 0.1,
) -> dict[str, Any]:
    """Aggregate the signed protocol by complete sequences, not windows."""

    target = np.asarray(y_true_ttc_s).reshape(-1)
    prediction = np.asarray(y_pred_ttc_s).reshape(-1)
    groups = np.asarray(list(sequence_ids)).astype(str).reshape(-1)
    if target.shape != prediction.shape or target.shape != groups.shape:
        raise ValueError("Sequence-macro inputs must have matching shapes.")
    per_sequence = {
        sequence: signed_garl_metrics(
            target[groups == sequence], prediction[groups == sequence], delta_t_s=delta_t_s
        )
        for sequence in sorted(np.unique(groups).tolist())
    }
    values = [float(item["paper_MiD_overall"]) for item in per_sequence.values()]
    finite = [value for value in values if np.isfinite(value)]
    invalid_mid_count = sum(int(item["invalid_mid_count"]) for item in per_sequence.values())
    invalid_row_count = sum(int(item["invalid_row_count"]) for item in per_sequence.values())
    sample_count = sum(int(item["num_samples"]) for item in per_sequence.values())
    assigned_sample_count = sum(
        int(item["assigned_sample_count"]) for item in per_sequence.values()
    )
    out_of_domain_count = sum(int(item["out_of_domain_count"]) for item in per_sequence.values())
    finite_sequence_count = len(finite)
    complete_sequence_coverage = finite_sequence_count == len(per_sequence)
    strict_metric_coverage = complete_sequence_coverage and all(
        bool(item["strict_metric_coverage"]) for item in per_sequence.values()
    )
    complete_cohort_accounting = complete_sequence_coverage and all(
        bool(item["complete_cohort_accounting"]) for item in per_sequence.values()
    )
    return {
        "protocol": "garl_signed_v1",
        "sequence_macro_paper_MiD_overall": float(np.mean(finite)) if finite else float("nan"),
        "per_sequence": per_sequence,
        "finite_sequence_count": finite_sequence_count,
        "sequence_count": len(per_sequence),
        "complete_sequence_coverage": complete_sequence_coverage,
        "invalid_mid_count": invalid_mid_count,
        "invalid_row_count": invalid_row_count,
        "invalid_row_rate_pct": float(invalid_row_count / sample_count * 100.0),
        "assigned_sample_count": assigned_sample_count,
        "out_of_domain_count": out_of_domain_count,
        "complete_cohort_accounting": complete_cohort_accounting,
        "strict_metric_coverage": strict_metric_coverage,
    }


def _has_complete_selection_accounting(item: dict[str, Any]) -> bool:
    """Return whether a metric row accounts for every validation sample."""

    try:
        sample_count = int(item["num_samples"])
        assigned_count = int(item["assigned_sample_count"])
        failure_count = int(item["failure_count"])
        invalid_mid_count = int(item["invalid_mid_count"])
        invalid_row_count = int(item["invalid_row_count"])
        invalid_row_rate = float(item["invalid_row_rate_pct"])
        failure_rate = float(item["failure_rate_pct"])
        mid = float(item["paper_MiD_overall"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    counts_are_consistent = (
        sample_count > 0
        and assigned_count == sample_count
        and item.get("out_of_domain_count") == 0
        and item.get("missing_bucket_count") == 0
        and 0 <= failure_count <= invalid_row_count <= sample_count
        and 0 <= invalid_mid_count <= invalid_row_count
        and np.isclose(invalid_row_rate, invalid_row_count / sample_count * 100.0)
        and np.isclose(failure_rate, failure_count / sample_count * 100.0)
    )
    return bool(
        counts_are_consistent
        and item.get("complete_bucket_coverage") is True
        and item.get("complete_cohort_accounting") is True
        and isinstance(item.get("strict_metric_coverage"), bool)
        and np.isfinite(mid)
        and np.isfinite(invalid_row_rate)
        and np.isfinite(failure_rate)
    )


def select_checkpoint(
    metrics: list[dict[str, Any]],
    *,
    protocol: str = "garl_signed_v1",
) -> dict[str, Any]:
    """Select a checkpoint using validation signed metrics only."""

    if protocol != "garl_signed_v1":
        raise ValueError(f"Unsupported checkpoint-selection protocol: {protocol!r}.")
    if not metrics:
        raise ValueError("Checkpoint selection requires validation metrics.")
    usable = [item for item in metrics if item.get("protocol") == protocol]
    if len(usable) != len(metrics):
        raise ValueError("Checkpoint metrics contain a different protocol.")
    finite_candidates = [
        (index, item)
        for index, item in enumerate(usable)
        if np.isfinite(float(item.get("paper_MiD_overall", np.nan)))
        and np.isfinite(float(item.get("failure_rate_pct", np.nan)))
    ]
    candidates = [
        (index, item)
        for index, item in finite_candidates
        if _has_complete_selection_accounting(item)
    ]
    if not candidates:
        raise ValueError("Checkpoint selection has no completely accounted validation metrics.")
    ranked = sorted(
        candidates,
        key=lambda pair: (
            float(pair[1]["invalid_row_rate_pct"]),
            float(pair[1]["paper_MiD_overall"]),
        ),
    )
    index, selected = ranked[0]
    return {
        "protocol": protocol,
        "selected_index": index,
        "selection_rule": "validation_invalid_row_rate_then_paper_MiD_overall",
        "excluded_non_finite_indices": sorted(
            set(range(len(usable))) - {i for i, _ in finite_candidates}
        ),
        "excluded_incomplete_indices": sorted(
            {i for i, _ in finite_candidates} - {i for i, _ in candidates}
        ),
        "selected": selected,
    }


__all__ = [
    "BUCKETS",
    "PAPER_MID_WEIGHTS",
    "select_checkpoint",
    "sequence_macro_signed_metrics",
    "signed_garl_metrics",
]
