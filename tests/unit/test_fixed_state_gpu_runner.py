from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from e_jepa_ttc.detection.fixed_state_gpu_diagnostic import build_plan

ROOT = Path(__file__).parents[2]


def _runner() -> Any:
    path = ROOT / "scripts/run_fixed_state_gpu_diagnostic.py"
    spec = importlib.util.spec_from_file_location("fixed_state_gpu_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalized_distances_use_declared_denominator_and_epsilon() -> None:
    runner = _runner()
    zero = torch.zeros(2)
    tiny = torch.tensor([1e-13, 0.0])
    assert runner._normalized(zero, zero, infinity=False) == 0.0
    assert runner._normalized(zero, tiny, infinity=True) == pytest.approx(0.1)


def test_symmetric_hausdorff_detects_candidate_set_outlier() -> None:
    runner = _runner()
    control = [{"vectors": {"x": torch.tensor([0.0])}} for _ in range(5)]
    candidate = [{"vectors": {"x": torch.tensor([0.0])}} for _ in range(4)]
    candidate.append({"vectors": {"x": torch.tensor([1.0])}})
    assert runner._hausdorff(control, candidate, "x", False) == pytest.approx(1.0)


def test_vector_key_sets_must_be_exact() -> None:
    runner = _runner()
    trials = [{"vectors": {"gradient/a": torch.ones(1)}} for _ in range(5)]
    assert runner._same_vector_keys(trials)
    trials[-1]["vectors"]["optimizer/extra"] = torch.ones(1)
    assert not runner._same_vector_keys(trials)


def test_empty_full_evaluation_is_rejected(tmp_path: Path) -> None:
    runner = _runner()
    evaluation = tmp_path / "full_selection_update_008192.json"
    evaluation.write_text("{}", encoding="utf-8")
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps({"sample_tokens": ["a"]}), encoding="utf-8")
    config = SimpleNamespace(
        evaluation=SimpleNamespace(protocol="default"),
        data=SimpleNamespace(full_selection_tokens=str(tokens)),
    )
    ordinary = SimpleNamespace(load_protocol=lambda value: SimpleNamespace(sha256="p" * 64))
    with pytest.raises(ValueError, match="evaluation contract differs"):
        runner._validate_full_evaluation(evaluation, config, ordinary)


def test_incomplete_r0_rejected_before_any_cuda_probe(tmp_path: Path, monkeypatch: Any) -> None:
    runner = _runner()
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "".join(
            json.dumps(
                {
                    "update": 1025 + index,
                    "sample_tokens": [f"t{index}-{item}" for item in range(4)],
                    "learning_rates": [0.0002, 0.0002],
                }
            )
            + "\n"
            for index in range(16)
        ),
        encoding="utf-8",
    )
    checkpoint, selected = tmp_path / "last.pt", tmp_path / "selected.pt"
    checkpoint.write_bytes(b"checkpoint")
    selected.write_bytes(b"selected")
    # This gate precedes parsing campaign inputs or executing the producer. Seal
    # minimal local artifacts so the test exercises that ordering in a clean clone.
    artifacts = {}
    for name in (
        "original_producer",
        "optimized_producer",
        "original_runtime_preregistration",
        "optimized_runtime_preregistration",
        "performance_config",
        "config_path",
    ):
        suffix = ".py" if name.endswith("producer") else ".json"
        path = tmp_path / (name + suffix)
        path.write_text("# Inert unit-test artifact.\n" if suffix == ".py" else "{}\n")
        artifacts[name] = path
    performance = artifacts["performance_config"]
    plan = build_plan(
        old_metrics_path=metrics,
        checkpoint_path=checkpoint,
        selected_path=selected,
        **artifacts,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"status": "RUNNING", "final_update": 4000}))
    monkeypatch.setattr(
        runner.torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA probed")),
    )
    monkeypatch.setattr(runner.sys, "path", list(runner.sys.path))
    with pytest.raises(ValueError, match="complete update 8192"):
        runner.run(plan_path, performance, summary, tmp_path / "output")
