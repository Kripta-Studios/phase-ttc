"""The manuscript must reject stale measurements and preserve failed queries."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / (
    "paper/from_motion_to_time_20261102/scripts/build_local_comparison.py"
)


def module():
    spec = importlib.util.spec_from_file_location("local_report", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_report_refuses_modified_trial_ledger(tmp_path):
    (tmp_path / "protocol.json").write_bytes(b"{}")
    (tmp_path / "trials.jsonl").write_bytes(b"{}\n")
    summary = {
        "status": "complete",
        "protocol_sha256": hashlib.sha256(b"{}").hexdigest(),
        "trials_sha256": hashlib.sha256(b"{}\n").hexdigest(),
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert module().load_run(tmp_path)[2] == [{}]
    (tmp_path / "trials.jsonl").write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="ledger"):
        module().load_run(tmp_path)


def test_report_complete_rectangular_population_required():
    rows = [
        dict(model=m, query=q, repeat=0, status="ok", total_ms=1.0)
        for m in ("a", "b")
        for q in ("x", "y")
    ]
    module().validate_population(rows, ["a", "b"], ["x", "y"], 1)
    with pytest.raises(ValueError, match="population"):
        module().validate_population(rows[:-1], ["a", "b"], ["x", "y"], 1)


def test_bypass_validator_rejects_different_inputs_even_with_equal_predictions():
    rows = [
        dict(
            model=m,
            query="x",
            repeat=0,
            status="ok",
            reason="",
            ttc_seconds=2.0,
            prepared_input_sha256="same",
            diagnostics={"heights": [1.0, 2.0]},
        )
        for m in ("garl_full", "garl_ttc_only")
    ]
    module().validate_parity(rows, "garl_full", "garl_ttc_only")
    rows[1]["prepared_input_sha256"] = "changed"
    with pytest.raises(ValueError, match="input"):
        module().validate_parity(rows, "garl_full", "garl_ttc_only")


def test_pairwise_matching_does_not_drop_queries_due_to_third_model():
    rows = [
        dict(model=m, query=q, repeat=0, status="ok")
        for m in ("phase", "full", "event")
        for q in ("x", "y")
    ]
    rows[-1]["status"] = "unavailable"
    assert len(module().paired(rows, ["phase", "full"])) == 4
    assert len(module().paired(rows, ["phase", "full", "event"])) == 3


def test_publication_refuses_smoke_cohort_and_wrong_native_cadence():
    protocol = {
        "models": ["phase_native_raw", "garl_full_raw", "garl_ttc_only_raw"],
        "repeats": 3,
        "queries": [{"sequence_id": str(i)} for i in range(12) for _ in range(10)],
        "garl_rows": {"q": {"native_delta_t_s": 0.1}},
    }
    module().validate_protocol(protocol, "eap")
    protocol["queries"].pop()
    with pytest.raises(ValueError, match="120 eAP"):
        module().validate_protocol(protocol, "eap")
    protocol["queries"].append({"sequence_id": "11"})
    protocol["garl_rows"]["q"]["native_delta_t_s"] = 0.100001
    with pytest.raises(ValueError, match="native Garl"):
        module().validate_protocol(protocol, "eap")


def test_profile_accounting_refuses_nonfinite_memory_or_excess_executed_count():
    profiles = {
        "p": dict(
            query="q",
            total_parameters=10,
            executed_parameters=5,
            parameter_bytes=40,
            peak_allocated_bytes=100,
            peak_reserved_bytes=100,
            baseline_allocated_bytes=40,
            counted_operator_flops=200,
        )
    }
    args = {"models": ["p"]}, [{"model": "p", "query": "q", "status": "ok"}]
    module().validate_profiles(profiles, *args)
    profiles["p"]["executed_parameters"] = 11
    with pytest.raises(ValueError, match="exceeds"):
        module().validate_profiles(profiles, *args)
    profiles["p"]["executed_parameters"] = 5
    profiles["p"]["peak_allocated_bytes"] = float("nan")
    with pytest.raises(ValueError, match="memory"):
        module().validate_profiles(profiles, *args)
