from __future__ import annotations

import math

import numpy as np
import pytest

from e_jepa_ttc.evaluation.garl_ttc_protocol import (
    select_checkpoint,
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)
from e_jepa_ttc.training.causal_scale_eap import _is_better, _selection


def test_missing_signed_buckets_make_weighted_overall_unavailable() -> None:
    metrics = signed_garl_metrics(np.asarray([1.0]), np.asarray([2.0]))

    assert math.isnan(metrics["paper_MiD_overall"])
    assert math.isnan(metrics["weighted_RTE_pct"])
    assert metrics["missing_bucket_count"] == 3
    assert metrics["complete_bucket_coverage"] is False
    assert metrics["invalid_mid_count"] == 0


def test_finite_skip_diagnostic_retains_explicit_invalid_mid_count() -> None:
    target = np.asarray([-1.0, 1.0, 2.0, 4.0, 8.0])
    prediction = np.asarray([-1.0, 0.05, 2.0, 4.0, 8.0])

    metrics = signed_garl_metrics(target, prediction)

    # The release-compatible bucket mean still skips the invalid row.
    assert metrics["bins"]["crucial"]["mid"] == pytest.approx(0.0)
    assert metrics["paper_MiD_overall"] == pytest.approx(0.0)
    assert metrics["invalid_mid_count"] == 1
    assert metrics["bins"]["crucial"]["invalid_mid_count"] == 1
    assert metrics["complete_bucket_coverage"] is True
    assert metrics["strict_metric_coverage"] is False


def test_sequence_macro_reports_skipped_sequences_and_invalid_rows() -> None:
    target = np.asarray([-1.0, 1.0, 4.0, 8.0, 1.0])
    prediction = target.copy()
    groups = ["complete"] * 4 + ["incomplete"]

    metrics = sequence_macro_signed_metrics(target, prediction, groups)

    assert metrics["sequence_macro_paper_MiD_overall"] == pytest.approx(0.0)
    assert metrics["finite_sequence_count"] == 1
    assert metrics["sequence_count"] == 2
    assert metrics["complete_sequence_coverage"] is False
    assert metrics["invalid_mid_count"] == 0
    assert metrics["strict_metric_coverage"] is False


def test_failure_with_finite_mid_is_still_not_strictly_valid() -> None:
    metrics = signed_garl_metrics(
        np.asarray([-1.0, 1.0, 4.0, 8.0]),
        np.asarray([-0.05, 1.0, 4.0, 8.0]),
    )

    assert metrics["invalid_mid_count"] == 0
    assert metrics["failure_count"] == 1
    assert metrics["strict_metric_coverage"] is False


@pytest.mark.parametrize("delta_t_s", [float("nan"), float("inf"), float("-inf")])
def test_delta_t_must_be_finite(delta_t_s: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        signed_garl_metrics(np.asarray([1.0]), np.asarray([1.0]), delta_t_s=delta_t_s)


def test_out_of_domain_targets_are_counted_and_not_strict() -> None:
    metrics = signed_garl_metrics(
        np.asarray([-1.0, 1.0, 4.0, 8.0, 20.0]),
        np.asarray([-1.0, 1.0, 4.0, 8.0, 999.0]),
    )

    assert metrics["assigned_sample_count"] == 4
    assert metrics["out_of_domain_count"] == 1
    assert metrics["complete_bucket_coverage"] is True
    assert metrics["complete_cohort_accounting"] is False
    assert metrics["strict_metric_coverage"] is False
    with pytest.raises(ValueError, match="no completely accounted validation metrics"):
        select_checkpoint([metrics])


def _selection_for(prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray([-1.0, -2.0, 1.0, 2.0, 4.0, 8.0])
    signed = signed_garl_metrics(target, prediction)
    macro = sequence_macro_signed_metrics(target, prediction, ["seq"] * len(target))
    return _selection({"signed": signed, "sequence_macro": macro})


def test_dev_best_ranks_invalid_rows_before_finite_skip_mid() -> None:
    failed = _selection_for(np.asarray([-1.0, -2.0, 0.05, 2.0, 4.0, 8.0]))
    valid = _selection_for(np.asarray([-2.0, -3.0, 2.0, 3.0, 5.0, 9.0]))

    assert failed["invalid_row_count"] == 1.0
    assert failed["invalid_row_rate_pct"] == pytest.approx(100.0 / 6.0)
    assert failed["strict_metric_coverage"] == 0.0
    assert _is_better(failed, None) is True
    assert _is_better(valid, failed) is True
    assert _is_better(failed, valid) is False


def test_dev_best_prefers_lower_known_invalid_row_rate() -> None:
    one_failed = _selection_for(np.asarray([-1.0, -2.0, 0.05, 2.0, 4.0, 8.0]))
    two_failed = _selection_for(np.asarray([-0.05, -2.0, 0.05, 2.0, 4.0, 8.0]))

    assert one_failed["invalid_row_rate_pct"] == pytest.approx(100.0 / 6.0)
    assert two_failed["invalid_row_rate_pct"] == pytest.approx(200.0 / 6.0)
    assert _is_better(one_failed, two_failed) is True
    assert _is_better(two_failed, one_failed) is False


def test_public_checkpoint_selector_ranks_invalid_rows_before_mid() -> None:
    valid = signed_garl_metrics(
        np.asarray([-1.0, 1.0, 4.0, 8.0]),
        np.asarray([-1.0, 1.0, 4.0, 8.0]),
    )
    invalid = signed_garl_metrics(
        np.asarray([-1.0, 1.0, 2.0, 4.0, 8.0]),
        np.asarray([-1.0, 0.05, 2.0, 4.0, 8.0]),
    )
    invalid["paper_MiD_overall"] = -1.0

    selected = select_checkpoint([invalid, valid])

    assert selected["selected_index"] == 1
    assert selected["excluded_incomplete_indices"] == []
    with pytest.raises(ValueError, match="no completely accounted validation metrics"):
        select_checkpoint(
            [
                {
                    "protocol": "garl_signed_v1",
                    "paper_MiD_overall": 0.0,
                    "failure_rate_pct": 0.0,
                }
            ]
        )


def test_public_checkpoint_selector_prefers_lower_invalid_row_rate() -> None:
    target = np.asarray([-1.0, -2.0, 1.0, 2.0, 4.0, 8.0])
    one_failed = signed_garl_metrics(
        target,
        np.asarray([-1.0, -2.0, 0.05, 2.0, 4.0, 8.0]),
    )
    two_failed = signed_garl_metrics(
        target,
        np.asarray([-0.05, -2.0, 0.05, 2.0, 4.0, 8.0]),
    )

    selected = select_checkpoint([two_failed, one_failed])

    assert selected["selected_index"] == 1
    assert selected["selection_rule"] == "validation_invalid_row_rate_then_paper_MiD_overall"
