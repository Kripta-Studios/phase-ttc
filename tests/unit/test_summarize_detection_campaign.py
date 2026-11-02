"""Focused fixtures for the campaign interim summarizer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/summarize_detection_campaign.py"
SPEC = importlib.util.spec_from_file_location("summarize_detection_campaign", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_training_curves_are_derived_from_every_contiguous_update():
    rows = [
        {
            "update": update,
            "loss": float(10 - update),
            "loss_center": float(update),
            "gradient_norm_before_clip": float(update * 2),
            "preprocessing_seconds": 0.5,
            "forward_backward_update_seconds": 0.25,
            "empty_target_exposures": update % 2,
            "valid_velocity_object_exposures": update,
        }
        for update in range(1, 5)
    ]
    result = MODULE.derive_training_curves(rows)
    assert result["update"] == [1, 2, 3, 4]
    assert result["loss"] == [9.0, 8.0, 7.0, 6.0]
    assert result["preprocessing_seconds_sum"] == 2.0
    assert result["forward_backward_update_seconds_sum"] == 1.0
    assert result["empty_target_exposures_sum"] == 2


def test_class_diagnostic_separates_no_proposals_from_no_loose_matches():
    metrics = {
        "per_class": {
            "absent": {
                "eligible_ground_truth_count": 2,
                "prediction_count_after_domain_filter": 0,
                "AP101_mean_thresholds": 0.0,
                "thresholds": {"4.0": {"TP": 0, "FP": 0}},
            },
            "missed": {
                "eligible_ground_truth_count": 3,
                "prediction_count_after_domain_filter": 10,
                "AP101_mean_thresholds": 0.0,
                "thresholds": {"4.0": {"TP": 0, "FP": 10}},
            },
            "positive": {
                "eligible_ground_truth_count": 4,
                "prediction_count_after_domain_filter": 8,
                "AP101_mean_thresholds": 0.2,
                "thresholds": {"4.0": {"TP": 3, "FP": 5}},
            },
        }
    }
    result = MODULE.analyze_classes(metrics)
    assert result["absent"]["observed_state"] == "no_class_predictions"
    assert result["missed"]["observed_state"] == "class_predictions_but_no_tp_within_4m"
    assert result["positive"]["observed_state"] == "positive_ap"


def test_noncontiguous_training_updates_fail_closed():
    rows = [{"update": 1}, {"update": 3}]
    # The contiguous check belongs to the JSONL boundary. The curve helper still
    # rejects malformed numeric rows rather than silently fabricating points.
    with pytest.raises(ValueError, match="loss@0"):
        MODULE.derive_training_curves(rows)
