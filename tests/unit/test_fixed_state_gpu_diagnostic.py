from __future__ import annotations

import json
from pathlib import Path

import pytest

from e_jepa_ttc.detection.fixed_state_gpu_diagnostic import build_plan, validate_plan


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        {
            "update": 1025 + index,
            "sample_tokens": [f"token-{index}-{item}" for item in range(4)],
            "learning_rates": [0.0002, 0.0002],
        }
        for index in range(16)
    ]
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    paths = {"old_metrics_path": metrics}
    for name in (
        "checkpoint_path",
        "selected_path",
        "original_producer",
        "optimized_producer",
        "original_runtime_preregistration",
        "optimized_runtime_preregistration",
        "performance_config",
        "config_path",
    ):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    return paths


def test_plan_fixes_equidistant_batches_and_reset_policy(tmp_path: Path) -> None:
    plan = build_plan(**_artifacts(tmp_path))
    validate_plan(plan)
    assert plan["relative_batches"] == [1, 6, 11, 16]
    assert [batch["historical_update"] for batch in plan["batches"]] == [1025, 1030, 1035, 1040]
    assert plan["control_repeats_per_batch"] == plan["candidate_repeats_per_batch"] == 5
    assert plan["pre_backward_parity"]["on_control_nonexact"].startswith("INCONCLUSIVE")


def test_plan_rejects_artifact_or_policy_drift(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    plan = build_plan(**artifacts)
    artifacts["config_path"].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        validate_plan(plan)
    plan["relative_batches"] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="plan hash differs"):
        validate_plan(plan)


def test_plan_rejects_nonunique_fixed_batch(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    rows = [json.loads(line) for line in artifacts["old_metrics_path"].read_text().splitlines()]
    rows[5]["sample_tokens"] = ["same"] * 4
    artifacts["old_metrics_path"].write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unique sample tokens"):
        build_plan(**artifacts)
