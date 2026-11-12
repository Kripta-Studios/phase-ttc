"""Integrity tests for expanded inference cohorts and deferred comparison."""

import pytest

from operational.local_comparison.extended import (
    compare_reference,
    longest_interval,
    validate_run_population,
)


def test_controller_rejects_a_scheduler_absent_from_the_freeze(tmp_path, monkeypatch):
    from scripts.run_extended_oracles import main

    (tmp_path / "input_inventory.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["oracle-controller", "--campaign", str(tmp_path)])
    with pytest.raises(ValueError, match="prospectively frozen"):
        main()
    assert not (tmp_path / "oracle_jobs.json").exists()


def test_longest_interval_preserves_gaps_and_earliest_tie():
    rows = [{"anchor_us": x} for x in [100, 20100, 80100, 100100]]
    assert longest_interval(rows) == rows[:2]
    with pytest.raises(ValueError, match="Duplicate"):
        longest_interval(rows + [rows[0]])


def test_garl_reference_requires_exact_input_and_finite_output():
    reference = {
        "update_id": "update-0",
        "sensor_sha256": "a",
        "contract": {},
        "heights": [[1.0, 2.0]],
        "ttc": 0.2,
    }
    assert compare_reference("garl", [reference], [reference])["all_delivered_passed"]
    missing = compare_reference("garl", [reference], [])
    assert not missing["complete"]
    bad = compare_reference("garl", [reference], [{**reference, "sensor_sha256": "b"}])
    assert not bad["all_delivered_passed"]
    with pytest.raises(ValueError, match="Duplicate"):
        compare_reference("garl", [reference], [reference, reference])


def test_declared_population_rejects_rate_and_missing_attempts():
    protocol = {"workloads": [{"id": i} for i in range(6)], "rate_hz": 25, "commands": []}
    with pytest.raises(ValueError, match="rate"):
        validate_run_population(protocol, [])
    protocol["rate_hz"] = 50
    with pytest.raises(ValueError, match="population"):
        validate_run_population(protocol, [])


def test_phase_metadata_parity_cannot_hide_prediction_change():
    obj = {
        "object_id": "roi_00",
        "history_lengths": {"event": 4},
        "rgb_available": True,
        "rgb_unavailable_reason": "",
        "input_receipt": {"query_us": 100},
        "predictions": {"PHASE_ONLY_TRAIN40_S7": 2.0, "E_CTX_TRAIN40": 3.0, "R_CTX_TRAIN40": 4.0},
    }
    reference = {"update_id": "update-0", "objects": [obj]}
    assert compare_reference("phase", [reference], [reference])["all_delivered_passed"]
    changed = {**obj, "predictions": {**obj["predictions"], "PHASE_ONLY_TRAIN40_S7": 3.0}}
    result = compare_reference(
        "phase", [reference], [{"update_id": "update-0", "objects": [changed]}]
    )
    assert not result["all_delivered_passed"]


def test_reference_contract_compares_json_identity_not_tuple_container():
    reference = {
        "update_id": "update-0",
        "sensor_sha256": "a",
        "contract": {"box": (1, 2)},
        "heights": [[1.0, 2.0]],
        "ttc": 0.2,
    }
    actual = {**reference, "contract": {"box": [1, 2]}}
    assert compare_reference("garl", [reference], [actual])["all_delivered_passed"]
