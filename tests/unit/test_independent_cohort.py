"""Independent-cohort preregistration and target-blind evaluator contracts."""

from __future__ import annotations

import hashlib

import pytest

from operational.independent_cohort import audit, evaluator, protocol


def _planned_manifest() -> dict[str, object]:
    return {
        "schema": "phase_ttc_independent_cohort_v1",
        "status": "planned",
        "created_utc": "2026-11-21T00:00:00Z",
        "known_session_ids": ["FCWD1", "FCWD2", "FCWD3", "eap:2cyv0Oedzg"],
        "sessions": [],
        "freeze": {
            "model_sha256": "1" * 64,
            "detector_sha256": "2" * 64,
            "tracker_sha256": "3" * 64,
            "evaluator_sha256": "4" * 64,
            "delay_policy_sha256": "5" * 64,
        },
        "design": {
            "primary_metric": "session_macro_rte",
            "practical_margin_rte_points": 3.0,
            "ci95_half_width_target_rte_points": 2.0,
            "minimum_independent_sessions": 12,
            "estimated_independent_sessions": 64,
            "sampling_strata": ["approach", "recede", "occlusion", "saturation"],
            "box_modes": ["supplied_immediate", "supplied_delayed_50ms", "detector_real"],
        },
    }


def test_planned_manifest_is_valid_but_reserves_no_sessions() -> None:
    checked = protocol.validate_manifest(_planned_manifest())
    assert checked["status"] == "planned"
    assert checked["sessions"] == []


def test_reserved_manifest_rejects_overlap_or_unverifiable_session() -> None:
    manifest = _planned_manifest()
    manifest["status"] = "reserved"
    manifest["sessions"] = [{"session_id": "FCWD1"}]
    with pytest.raises(ValueError, match="already known"):
        protocol.validate_manifest(manifest)

    manifest["sessions"] = [{"session_id": "NEW-001"}]
    with pytest.raises(ValueError, match="provenance"):
        protocol.validate_manifest(manifest)


def test_three_box_policies_obey_capture_and_availability_times() -> None:
    query = evaluator.Query("q1", "new-session", "car-1", reference_us=1_000_000)
    boxes = [
        evaluator.BoxObservation(
            "new-session", "car-1", 900_000, 900_000, (0, 0, 10, 10), "supplied"
        ),
        evaluator.BoxObservation(
            "new-session", "car-1", 980_000, 980_000, (0, 0, 12, 12), "supplied"
        ),
        evaluator.BoxObservation(
            "new-session", "car-1", 900_000, 940_000, (0, 0, 9, 9), "detector"
        ),
        evaluator.BoxObservation(
            "new-session", "car-1", 970_000, 1_020_000, (0, 0, 11, 11), "detector"
        ),
    ]

    immediate = evaluator.available_history(query, boxes, "supplied_immediate")
    delayed = evaluator.available_history(query, boxes, "supplied_delayed_50ms")
    detected = evaluator.available_history(query, boxes, "detector_real")

    assert [b.capture_us for b in immediate] == [900_000, 980_000]
    assert [b.capture_us for b in delayed] == [900_000]
    assert [b.capture_us for b in detected] == [900_000]


def test_prediction_rows_seal_the_effective_availability_clock_for_each_policy() -> None:
    query = evaluator.Query("q1", "s1", "car", 1_000_000)
    boxes = [
        evaluator.BoxObservation("s1", "car", 800_000, 810_000, (0, 0, 10, 10), "supplied"),
        evaluator.BoxObservation("s1", "car", 900_000, 920_000, (0, 0, 12, 12), "supplied"),
        evaluator.BoxObservation("s1", "car", 800_000, 830_000, (0, 0, 10, 10), "detector"),
        evaluator.BoxObservation("s1", "car", 900_000, 940_000, (0, 0, 12, 12), "detector"),
    ]

    rows = evaluator.run_predictions([query], boxes, lambda _history: 2.0)
    effective = {row.mode: row.last_box_effective_available_us for row in rows}

    assert effective == {
        "supplied_immediate": 900_000,
        "supplied_delayed_50ms": 950_000,
        "detector_real": 940_000,
    }


def test_sealed_evaluator_reports_common_support_coverage_and_failures() -> None:
    queries = [
        evaluator.Query("q1", "s1", "car", 1_000_000),
        evaluator.Query("q2", "s1", "car", 2_000_000),
    ]
    boxes = [
        evaluator.BoxObservation("s1", "car", 800_000, 800_000, (0, 0, 10, 10), "supplied"),
        evaluator.BoxObservation("s1", "car", 950_000, 950_000, (0, 0, 12, 12), "supplied"),
        evaluator.BoxObservation("s1", "car", 800_000, 850_000, (0, 0, 10, 10), "detector"),
        evaluator.BoxObservation("s1", "car", 950_000, 970_000, (0, 0, 12, 12), "detector"),
        evaluator.BoxObservation("s1", "car", 1_900_000, 1_900_000, (0, 0, 15, 15), "supplied"),
        evaluator.BoxObservation("s1", "car", 1_950_000, 1_950_000, (0, 0, 16, 16), "supplied"),
    ]

    def predictor(history: tuple[evaluator.BoxObservation, ...]) -> float:
        return 2.0 if len(history) >= 2 else float("nan")

    rows = evaluator.run_predictions(queries, boxes, predictor, history_us=300_000)
    payload, receipt = evaluator.seal_predictions(
        rows,
        manifest_sha256="1" * 64,
        model_sha256="2" * 64,
        evaluator_sha256="3" * 64,
        delay_policy_sha256="4" * 64,
    )
    assert receipt["prediction_sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["targets_read"] is False

    result = evaluator.score_sealed(
        payload, receipt, {"q1": 2.5, "q2": 1.0}, truth_file_sha256="5" * 64
    )
    assert result["eligible_queries"] == 2
    assert result["common_support_queries"] == 1
    assert result["modes"]["supplied_immediate"]["coverage"] == 1.0
    assert result["modes"]["detector_real"]["coverage"] == 0.5
    assert result["modes"]["detector_real"]["failures"] == {"insufficient_history": 1}
    assert result["modes"]["supplied_immediate"]["common_support_rte"] == pytest.approx(20.0)
    assert result["truth_file_sha256"] == "5" * 64
    assert result["prediction_seal"]["manifest_sha256"] == "1" * 64


def test_scoring_rejects_a_modified_prediction_seal() -> None:
    payload, receipt = evaluator.seal_predictions(
        [],
        manifest_sha256="1" * 64,
        model_sha256="2" * 64,
        evaluator_sha256="3" * 64,
        delay_policy_sha256="4" * 64,
    )
    with pytest.raises(ValueError, match="seal"):
        evaluator.score_sealed(payload + b" ", receipt, {}, truth_file_sha256="5" * 64)


def test_scoring_rejects_an_unbound_or_label_exposed_seal_receipt() -> None:
    payload, receipt = evaluator.seal_predictions(
        [],
        manifest_sha256="1" * 64,
        model_sha256="2" * 64,
        evaluator_sha256="3" * 64,
        delay_policy_sha256="4" * 64,
    )
    receipt["targets_read"] = True
    with pytest.raises(ValueError, match="target-blind"):
        evaluator.score_sealed(payload, receipt, {}, truth_file_sha256="5" * 64)

    receipt["targets_read"] = False
    receipt.pop("sealed_utc")
    with pytest.raises(ValueError, match="sealed UTC"):
        evaluator.score_sealed(payload, receipt, {}, truth_file_sha256="5" * 64)


def test_scoring_rejects_a_sealed_row_with_a_noncausal_effective_clock() -> None:
    rows = [
        evaluator.PredictionRow(
            "q1",
            "s1",
            1_000_000,
            mode,
            True,
            2.0,
            None,
            2,
            980_000,
            980_000,
            1_030_000 if mode == "supplied_delayed_50ms" else 980_000,
        )
        for mode in evaluator.BOX_MODES
    ]
    payload, receipt = evaluator.seal_predictions(
        rows,
        manifest_sha256="1" * 64,
        model_sha256="2" * 64,
        evaluator_sha256="3" * 64,
        delay_policy_sha256="4" * 64,
    )

    with pytest.raises(ValueError, match="effective availability"):
        evaluator.score_sealed(payload, receipt, {"q1": 2.5}, truth_file_sha256="5" * 64)


def test_scoring_rejects_modes_with_different_query_populations() -> None:
    rows = [
        evaluator.PredictionRow(
            "q1",
            "s1",
            1_000_000,
            mode,
            True,
            2.0,
            None,
            2,
            900_000,
            950_000,
            900_000 if mode == "supplied_immediate" else 950_000,
        )
        for mode in evaluator.BOX_MODES[:-1]
    ]
    payload, receipt = evaluator.seal_predictions(
        rows,
        manifest_sha256="1" * 64,
        model_sha256="2" * 64,
        evaluator_sha256="3" * 64,
        delay_policy_sha256="4" * 64,
    )
    with pytest.raises(ValueError, match="identical query populations"):
        evaluator.score_sealed(payload, receipt, {"q1": 2.5}, truth_file_sha256="5" * 64)


def test_fcwd_audit_preserves_630_597_33_and_content_failures() -> None:
    report = audit.fcwd_audit(
        "docs/sota_20261118/second_benchmark/RESULTS.json",
        "docs/sota_20261118/saturation_scale/FCWD_RESULTS.json",
    )
    assert report["anchors"] == 630
    assert report["scorable"] == 597
    assert report["outside_truth_support"] == 33
    assert report["content_invalid"] == 14
    assert report["content_scale_mean_per_sequence_rte_rounded_1dp"] == {
        "FCWD1": 25.4,
        "FCWD2": 12.4,
        "FCWD3": 10.4,
    }
    assert all(
        delta > 0 for delta in report["content_invalid_delta_vs_no_content_by_seed"].values()
    )
    checked_in = protocol.load_json("docs/sota_20261121/independent_cohort/FCWD_AUDIT.json")
    assert checked_in == report


def test_checked_in_manifest_and_inventory_do_not_claim_an_independent_session() -> None:
    manifest = protocol.load_manifest("docs/sota_20261121/independent_cohort/PREREGISTRATION.json")
    assert manifest["status"] == "planned" and manifest["sessions"] == []
    inventory = protocol.load_json("docs/sota_20261121/independent_cohort/DATASET_AUDIT.json")
    assert not any(item["independent_cohort_eligible"] for item in inventory["datasets"])


def test_boreas_shortlist_is_one_session_not_37_independent_sessions() -> None:
    candidate = protocol.load_json("docs/sota_20261121/independent_cohort/BOREAS_CANDIDATE.json")
    assert candidate["selection"]["obj_train_scene_intervals"] == 37
    assert candidate["selection"]["physical_session_count"] == 1
    assert candidate["status"] == "metadata_shortlisted_not_reserved"
    assert protocol.canonical_sha256(candidate["selection"]) == candidate["selection_sha256"]
