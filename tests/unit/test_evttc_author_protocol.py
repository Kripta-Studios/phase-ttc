from __future__ import annotations

from pathlib import Path

import pytest

from e_jepa_ttc.data.evttc_author_protocol import (
    ScenarioInterval,
    TimedValue,
    load_author_protocol,
    mapped_square_roi,
    nearest_timed_value,
    protocol_intervals,
    scenario_for_timestamp,
    select_tracked_bbox,
)
from e_jepa_ttc.evaluation.garl_evttc_zero_shot import score_zero_shot_predictions

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "data" / "protocols" / "evttc_author_20261011.json"


def test_versioned_protocol_preserves_author_evidence_limits() -> None:
    protocol = load_author_protocol(PROTOCOL)
    intervals = protocol_intervals(protocol)
    assert [interval.scenario_id for interval in intervals] == [
        "CCRs2-medium",
        "CCRs2-high",
        "CCRm-medium",
    ]
    assert protocol["official_query_population_verified"] is False
    assert protocol["official_table_vi_checkpoint_sha256_verified"] is False
    assert scenario_for_timestamp(9.4, intervals, scenario_id="CCRs2-medium") == "CCRs2-medium"
    assert scenario_for_timestamp(11.6, intervals, scenario_id="CCRs2-medium") == "CCRs2-medium"
    assert scenario_for_timestamp(11.600001, intervals, scenario_id="CCRs2-medium") is None


def test_nearest_ttc_match_includes_exact_tolerance_and_resolves_ties_earlier() -> None:
    candidates = [TimedValue(1.05, 5.0), TimedValue(0.95, 4.0)]
    assert nearest_timed_value(1.0, candidates) == TimedValue(0.95, 4.0)
    assert nearest_timed_value(1.0, [TimedValue(1.050001, 5.0)]) is None


def test_nearest_ttc_rejects_conflicting_duplicate_timestamps() -> None:
    with pytest.raises(ValueError, match="Conflicting TTC"):
        nearest_timed_value(1.0, [TimedValue(1.0, 2.0), TimedValue(1.0, 3.0)])


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), float("-inf")])
def test_nearest_ttc_rejects_nonfinite_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        nearest_timed_value(1.0, [TimedValue(1.0, 2.0)], tolerance_s=tolerance)


def test_bbox_tracking_is_deterministic_and_initial_ambiguity_is_explicit() -> None:
    boxes = [(8.0, 0.0, 10.0, 2.0), (0.0, 0.0, 2.0, 2.0)]
    assert select_tracked_bbox(reversed(boxes), (5.0, 1.0)) == boxes[1]
    with pytest.raises(ValueError, match="Initial target is unresolved"):
        select_tracked_bbox(boxes, None)
    assert select_tracked_bbox([boxes[0]], None) == boxes[0]


def test_roi_maps_all_four_corners_and_squares_enclosure() -> None:
    observed = []

    def mapper(points):
        observed.extend(points)
        return [(x + 10.0, y - 2.0) for x, y in points]

    roi = mapped_square_roi((0.0, 0.0, 4.0, 2.0), calibration_mapper=mapper)
    assert observed == [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]
    assert roi == pytest.approx((10.0, -3.0, 14.0, 1.0))
    with pytest.raises(ValueError, match="mapping is unresolved"):
        mapped_square_roi((0.0, 0.0, 4.0, 2.0), calibration_mapper=None)


def test_plain_rte_scores_zero_and_negative_predictions_without_filtering() -> None:
    metrics = score_zero_shot_predictions([2.0, 2.0], [0.0, -2.0])
    assert metrics["mean_RTE_pct"] == pytest.approx(150.0)
    assert metrics["failure_count"] == 0


def test_nonfinite_prediction_invalidates_rte_instead_of_improving_subset_mean() -> None:
    metrics = score_zero_shot_predictions([2.0, 2.0], [2.0, float("nan")])
    assert metrics["mean_RTE_pct"] is None
    assert metrics["status"] == "invalid_predictions"
    assert metrics["sample_count"] == 2
    assert metrics["failure_count"] == 1


def test_plain_rte_mean_does_not_overflow_for_large_finite_per_sample_errors() -> None:
    metrics = score_zero_shot_predictions([1.0e-306, 1.0e-306], [1.0, 1.0])
    assert metrics["status"] == "complete"
    assert metrics["mean_RTE_pct"] == pytest.approx(1.0e308)


def test_plain_rte_avoids_subtraction_overflow_for_opposite_large_values() -> None:
    metrics = score_zero_shot_predictions([-1.0e308], [1.0e308])
    assert metrics["status"] == "complete"
    assert metrics["mean_RTE_pct"] == pytest.approx(200.0)


def test_distinct_sequence_intervals_may_overlap_and_are_selected_explicitly() -> None:
    intervals = protocol_intervals(
        {
            "configured_intervals": [
                {
                    "scenario_id": "a",
                    "canonical_sequence_id": "sequence-a",
                    "start_s": 0.0,
                    "end_s": 1.0,
                },
                {
                    "scenario_id": "b",
                    "canonical_sequence_id": "sequence-b",
                    "start_s": 0.5,
                    "end_s": 2.0,
                },
            ]
        }
    )
    assert scenario_for_timestamp(0.75, intervals, scenario_id="a") == "a"
    assert scenario_for_timestamp(0.75, intervals, scenario_id="b") == "b"
    assert ScenarioInterval("x", 0.0, 1.0).contains(1.0)
