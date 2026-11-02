from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest


def _module() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "build_detection_progress_status.py"
    spec = importlib.util.spec_from_file_location("progress_status", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load progress status script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    evidence = tmp_path / "evidence"
    _write(
        evidence / "original_r0_stop_v1" / "TRAINING_TAIL_ACCOUNTING.json",
        {
            "retained_update": 3840,
            "last_logged_update": 4068,
            "discarded_completed_updates": 228,
            "discarded_logged_exposures": 912,
            "scientific_continuation_must_resume_update": 3840,
            "training_completion_claimed": False,
        },
    )
    summary = {
        "status": "BUDGET_STOPPED",
        "start_update": 1024,
        "final_update": 1040,
        "unscheduled_stop_evaluation": {"selection_used": False},
    }
    _write(evidence / "gpu_admission_1024_v1" / "old_A" / "summary.json", summary)
    _write(evidence / "gpu_admission_1024_v1" / "old_B" / "summary.json", summary)
    _write(
        evidence / "batched_evaluation_profile_v1" / "receipt.json",
        {
            "status": "PASS",
            "summary": {
                "1": {"passes_preregistered_tolerance": True},
                "2": {"passes_preregistered_tolerance": False},
                "4": {"passes_preregistered_tolerance": False},
            },
            "recommendation": "REJECT_BATCHED_EVALUATION",
        },
    )
    _write(
        evidence / "original_fallback_plan_v1" / "launch_argv_v2.json",
        {
            "status": "PREPARED_NOT_EXECUTED",
            "reason": "OPTIMIZED_READER_FAILED_PREREGISTERED_NUMERICAL_PARITY_GATE",
            "start_update": 3840,
        },
    )
    _write(
        repo / "output" / "performance" / "performance_diagnosis_v5.json",
        {
            "measured": {
                "ap_progression": {"update3072_dev_ap101_supported_classes": 0.07669186858179068}
            }
        },
    )
    plan = (
        repo
        / "docs"
        / "detection_sota_20261029"
        / "evidence"
        / "intake"
        / "SOTA_DEVELOPMENT_PLAN.md"
    )
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("# plan\n", encoding="utf-8")
    return repo, evidence


def test_status_separates_durable_work_from_discarded_tail(tmp_path: Path) -> None:
    repo, evidence = _fixture(tmp_path)
    result = cast(Any, _module()).build_status(repo, evidence)
    state = result["current_state"]
    assert state["durable_resume_update"] == 3840
    assert state["remaining_r0_seed7_updates"] == 4352
    assert state["original_logged_tail"] == {
        "last_logged_update": 4068,
        "discarded_updates": 228,
        "discarded_exposures": 912,
        "reduces_remaining_work": False,
    }


def test_eta_is_unknown_and_cross_host_speed_is_not_claimed(tmp_path: Path) -> None:
    repo, evidence = _fixture(tmp_path)
    result = cast(Any, _module()).build_status(repo, evidence)
    assert result["eta"]["hours"] is None
    assert (
        result["eta"]["total_eta_status"] == "UNKNOWN_PENDING_EVALUATION_AND_REMAINING_EXPERIMENTS"
    )
    rendered = cast(Any, _module()).render_markdown(result)
    assert "No se publica una ETA total en horas" in rendered
    assert "no una comparación causal de velocidad entre hosts" in rendered
    assert "Esta cifra no representa todo S1" in rendered


def test_recovery_batches_and_eta_window_are_explicit(tmp_path: Path) -> None:
    repo, evidence = _fixture(tmp_path)
    result = cast(Any, _module()).build_status(repo, evidence)
    strict = result["current_state"]["strict_suite"]
    assert strict["recovery_training_batch_size"] == 4
    assert strict["recovery_evaluation_batch_size"] == 1
    assert strict["external_stop_claimed"] is False
    window = result["eta"]["preregistered_passive_window"]
    assert window["completed_updates"] == 100
    assert window["report_evaluation_seconds_separately"] is True
    assert window["observations_available"] is False
    assert window["update_range_inclusive"] == [3841, 3940]
    clean = result["eta"]["future_clean_passive_window"]
    assert clean["start_update"] == 3901
    assert clean["end_update"] == 4000
    assert clean["status"] == "PENDING"
    assert clean["eligible_for_conditional_r0_eta"] is False


def test_optimized_runtime_is_rejected_and_fallback_is_not_pass(tmp_path: Path) -> None:
    repo, evidence = _fixture(tmp_path)
    result = cast(Any, _module()).build_status(repo, evidence)
    assert result["current_state"]["optimized_runtime"]["adopted"] is False
    fallback = result["current_state"]["original_fallback"]
    assert fallback["status"] == "LAUNCHED_UNVERIFIED_INCOMPLETE"
    assert fallback["pass_claimed"] is False


def test_complete_clean_window_uses_aggregate_means(tmp_path: Path) -> None:
    repo, evidence = _fixture(tmp_path)
    metrics = evidence / "original_fallback_plan_v1" / "run" / "metrics.jsonl"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(
        "".join(
            json.dumps(
                {
                    "update": update,
                    "preprocessing_seconds": 1.0,
                    "forward_backward_update_seconds": 2.0,
                }
            )
            + "\n"
            for update in range(3901, 4001)
        ),
        encoding="utf-8",
    )
    result = cast(Any, _module()).build_status(repo, evidence)
    clean = result["eta"]["future_clean_passive_window"]
    assert clean["eligible_for_conditional_r0_eta"] is True
    observed = clean["observed_metrics"]
    assert observed["aggregate_component_seconds"] == 300.0
    assert observed["aggregate_component_seconds_mean_per_update"] == 3.0
    assert observed["evaluation_seconds"] is None
    projection = result["eta"]["conditional_r0_component_projection"]
    assert projection["remaining_updates"] == 4192
    assert projection["projected_preprocessing_plus_compute_seconds"] == 12576.0
    assert projection["is_total_eta"] is False


def test_failed_batched_parity_remains_rejected(tmp_path: Path) -> None:
    repo, evidence = _fixture(tmp_path)
    result = cast(Any, _module()).build_status(repo, evidence)
    batched = result["current_state"]["batched_evaluation"]
    assert batched["parity_by_batch"] == {"1": True, "2": False, "4": False}
    assert batched["adopted"] is False


def test_inconsistent_tail_fails_closed(tmp_path: Path) -> None:
    repo, evidence = _fixture(tmp_path)
    tail_path = evidence / "original_r0_stop_v1" / "TRAINING_TAIL_ACCOUNTING.json"
    tail = json.loads(tail_path.read_text(encoding="utf-8"))
    tail["discarded_logged_exposures"] = 911
    _write(tail_path, tail)
    with pytest.raises(ValueError, match="batch size four"):
        cast(Any, _module()).build_status(repo, evidence)


def test_markdown_has_all_phase_rows(tmp_path: Path) -> None:
    repo, evidence = _fixture(tmp_path)
    result = cast(Any, _module()).build_status(repo, evidence)
    rendered = cast(Any, _module()).render_markdown(result)
    for phase in ("S1", "S2", "S3", "S4", "S5", "S6"):
        assert f"| {phase} |" in rendered
