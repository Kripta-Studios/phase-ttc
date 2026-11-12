"""Persistent publication must use scheduled response and retain incomplete passes."""

import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[2] / (
        "paper/from_motion_to_time_20261102/scripts/build_persistent_comparison.py"
    )
    spec = importlib.util.spec_from_file_location("persistent_report", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def paced():
    rows = []
    arrivals = []
    for index in range(3):
        start = 1_000_000_000 + index * 20_000_000
        arrivals.append(dict(update_id=f"update-{index}", status="ENQUEUED", scheduled_ns=start))
        rows.append(
            dict(
                update_id=f"update-{index}",
                object_id="roi_00",
                scheduled_ns=start,
                observation_available_ns=start + 5_000_000,
                started_ns=start + 8_000_000,
                completed_ns=start + 25_000_000,
                success=True,
            )
        )
    return {
        "offered_updates": 3,
        "rate_hz": 50.0,
        "queue_capacity": 2,
        "arrival_receipts": arrivals,
        "object_records": rows,
    }


def test_scheduled_clock_includes_acquisition_and_queue_wait():
    result = module().summarize_paced(paced(), expected=3)
    assert result["complete"]
    assert result["deadline"]["p95_ms"] == 25.0
    assert result["availability"]["p95_ms"] == 20.0
    assert result["service"]["p95_ms"] == 17.0
    assert result["under_20ms"] == 0


def test_drop_cannot_become_fast_primary_p95():
    data = paced()
    data["arrival_receipts"][-1]["status"] = "DROPPED_QUEUE_FULL"
    data["object_records"].pop()
    result = module().summarize_paced(data, expected=3)
    assert not result["complete"]
    assert result["primary_p95_ms"] is None
    assert result["completed"] == 2
    assert result["drops"] == 1
    assert result["offered"] == 3


def test_duplicate_clock_population_is_rejected():
    data = paced()
    data["object_records"][-1]["update_id"] = "update-0"
    with pytest.raises(ValueError, match="duplicate"):
        module().summarize_paced(data, expected=3)


def test_inverted_clock_is_rejected():
    data = paced()
    data["object_records"][0]["observation_available_ns"] = 0
    with pytest.raises(ValueError, match="clock"):
        module().summarize_paced(data, expected=3)


def test_missing_enqueued_result_is_rejected():
    data = paced()
    data["object_records"].pop()
    with pytest.raises(ValueError, match="accounting"):
        module().summarize_paced(data, expected=3)
