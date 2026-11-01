"""Scientific scope and exclusive-writer guards for the new technical GPU smoke."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import psutil
import pytest

from e_jepa_ttc.detection.benchmark_evaluation import PROFILE_SHA256, BenchmarkEvaluator
from e_jepa_ttc.detection.monocop_smoke import (
    INITIAL_SHA256,
    POPULATION_SHA256,
    PORT_SHA256,
    MonoCoPSmokeConfig,
    verify_admission,
    verify_replay,
)
from e_jepa_ttc.detection.range_campaign import digest


def test_exact_replay_accepts_actual_profile_json_roundtrip() -> None:
    evaluator = BenchmarkEvaluator()
    evaluator.add_frame("fixture:001", [], [])
    replay = {"predictions": {"fixture:001": []}, "metrics": evaluator.summarize()}
    verify_replay(replay, json.loads(json.dumps(replay)))


def test_exact_replay_rejects_even_a_small_numeric_change() -> None:
    evaluator = BenchmarkEvaluator()
    evaluator.add_frame("fixture:001", [], [])
    replay = {"predictions": {"fixture:001": []}, "metrics": evaluator.summarize()}
    expected = json.loads(json.dumps(replay))
    expected["metrics"]["mAP"] += 1e-15
    with pytest.raises(ValueError, match="metrics differ"):
        verify_replay(replay, expected)


def config() -> MonoCoPSmokeConfig:
    """A valid recipe whose paths remain unused in these admission-only tests."""
    return MonoCoPSmokeConfig(
        "phase_monocop_gpu_smoke_v1",
        "fixture",
        "data",
        "population",
        POPULATION_SHA256,
        "source",
        PORT_SHA256,
        "weights",
        INITIAL_SHA256,
        "cpu-result",
        "a" * 64,
        PROFILE_SHA256,
        7,
        640,
        384,
        300.0,
        2,
        32,
        [0, 16, 32],
        5e-5,
        0.1,
        1e-4,
        1.0,
        600.0,
        9 * 2**30,
        4,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"updates": 2048},
        {"depth_max_m": 400.0},
        {"maximum_seconds": 7200.0},
        {"batch_size": True},
        {"width": 640.0},
        {"evaluation_updates": [0, 32]},
    ],
)
def test_smoke_cannot_silently_expand_into_a_quality_campaign(change: dict) -> None:
    with pytest.raises(ValueError):
        replace(config(), **change).validate()


def test_unknown_config_setting_is_not_ignored(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**asdict(config()), "use_test_for_selection": True}))
    with pytest.raises(ValueError, match="keys differ"):
        MonoCoPSmokeConfig.load(path)


def admission_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Bind terminal receipts while leaving process identities independently testable."""
    plan = tmp_path / "PLAN.json"
    queue = tmp_path / "QUEUE.json"
    queue.write_text("{}")
    current = psutil.Process()
    processes = [
        {"pid": current.pid, "created_epoch": current.create_time()},
        {"pid": 124, "created_epoch": 101.0},
    ]
    plan.write_text(
        json.dumps(
            {
                "predecessor_queue": str(queue),
                "predecessor_queue_sha256": digest(queue),
                "predecessor_processes": processes,
            }
        )
    )
    completed = {}
    for stage, relative in (
        ("smoke", "smoke/STATUS.json"),
        ("smoke_gate", "SMOKE_GATE.json"),
        ("screen", "screen/STATUS.json"),
    ):
        receipt = tmp_path / relative
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"status": "COMPLETE"}))
        completed[stage] = {"returncode": 0, "receipt_sha256": digest(receipt)}
    state = tmp_path / "QUEUE.status.json"
    state.write_text(json.dumps({"status": "COMPLETE", "completed": completed}))
    checkpoint = tmp_path / "screen/checkpoints/selected.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint bytes for provenance-only fixture")
    review = tmp_path / "REVIEW.json"
    review.write_text(
        json.dumps(
            {
                "schema": "phase_completed_pretrained_screen_v1",
                "runs": {"kitti": {"status": {"status": "COMPLETE"}}},
                "selected_checkpoint": str(checkpoint),
                "selected_checkpoint_sha256": digest(checkpoint),
            }
        )
    )
    value = {
        "schema": "phase_monocop_gpu_smoke_admission_v1",
        "decision": "GPU_TECHNICAL_SMOKE_ONLY",
        "plan_sha256": digest(plan),
        "config_sha256": config().sha256,
        "maximum_seconds": 600.0,
        "new_cloud_spend": False,
        "predecessor_processes": processes,
        "predecessor_queue": str(queue),
        "predecessor_queue_sha256": digest(queue),
        "predecessor_status_sha256": digest(state),
        "review_result": str(review),
        "review_result_sha256": digest(review),
    }
    path = tmp_path / "ADMISSION.json"
    return path, plan, value


def test_terminal_json_does_not_override_a_live_writer(tmp_path: Path) -> None:
    path, plan, value = admission_fixture(tmp_path)
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="still alive"):
        verify_admission(path, plan, config())


def test_completion_receipt_cannot_change_after_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, plan, value = admission_fixture(tmp_path)

    def absent(pid: int) -> None:
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", absent)
    path.write_text(json.dumps(value))
    assert verify_admission(path, plan, config())["decision"] == "GPU_TECHNICAL_SMOKE_ONLY"
    (tmp_path / "SMOKE_GATE.json").write_text('{"status":"FAILED"}')
    with pytest.raises(ValueError, match="completion receipt differs"):
        verify_admission(path, plan, config())


def test_process_inventory_is_required(tmp_path: Path) -> None:
    path, plan, value = admission_fixture(tmp_path)
    value["predecessor_processes"] = []
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="both"):
        verify_admission(path, plan, config())


def test_admission_cannot_replace_the_registered_predecessor(tmp_path: Path) -> None:
    path, plan, value = admission_fixture(tmp_path)
    value["predecessor_processes"][0]["pid"] += 10000
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity differs"):
        verify_admission(path, plan, config())


def test_review_cannot_bind_a_different_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, plan, value = admission_fixture(tmp_path)

    def absent(pid: int) -> None:
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", absent)
    path.write_text(json.dumps(value))
    (tmp_path / "screen/checkpoints/selected.pt").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="completed predecessor checkpoint"):
        verify_admission(path, plan, config())
