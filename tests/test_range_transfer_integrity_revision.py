"""Regression tests for V-cohort ancestry and historical trace terminal states."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.rgb_port.features import EVENT_PHASE17_SHA256
from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE
from operational.range_transfer import analyze
from operational.range_transfer.common import read, sha, write
from operational.range_transfer.trace_status import (
    begin_trace,
    complete_trace,
    fail_trace,
    finish_trace,
    read_trace,
)


def _phase(ttc: np.ndarray) -> np.ndarray:
    return (-np.log1p(-0.1 / ttc)).astype(np.float32)


def _save_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    savez = cast(Any, np.savez)
    savez(path, **values)


def _v_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, dict[str, str]]]:
    owner = tmp_path / "owner"
    run = tmp_path / "run"
    feature_dir = owner / "features/V"
    parent = owner / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json"
    write(parent, {"endpoints": {"frozen": {"checkpoint_sha256": "parent"}}})
    parent_sha = sha(parent)

    normalizer = owner / "features/H/E_NORMALIZER.json"
    write(
        normalizer,
        {
            "fit_role": "H",
            "modality": "event",
            "schema_sha256": EVENT_PHASE17_SHA256,
            "producer_sha256": parent_sha,
        },
    )
    observations = feature_dir / "E_OBSERVATIONS.npz"
    _save_npz(
        observations,
        observation_id=np.asarray(["o0", "o1", "o2", "o3"]),
        features=np.zeros((4, 17), np.float32),
        role=np.asarray(["V"] * 4),
        schema_sha256=np.asarray(EVENT_PHASE17_SHA256),
        endpoint_freeze_sha256=np.asarray(parent_sha),
    )
    write(
        observations.with_suffix(".json"),
        {
            "schema": "rgb_port_observation_features_v1",
            "status": "COMPLETE",
            "role": "V",
            "modality": "event",
            "schema_sha256": EVENT_PHASE17_SHA256,
            "endpoint_hashes": {"frozen": "parent"},
            "output_sha256": sha(observations),
        },
    )
    tokens = np.asarray(["a", "b", "c", "d"])
    groups = np.asarray(["g1", "g1", "g2", "g2"])
    ttc = np.asarray([1.0, 4.0, 7.0, -2.0])
    cache = feature_dir / "E_HISTORY.npz"
    _save_npz(
        cache,
        features=np.zeros((4, 1, 17), np.float32),
        timing=np.zeros((4, 1, 4), np.float32),
        valid=np.ones((4, 1), np.bool_),
        expert_phase=np.zeros((4, 3), np.float32),
        target_phase=_phase(ttc),
        mass=np.full(4, 0.25, np.float32),
        sample_token=tokens,
        group_id=groups,
        role=np.asarray(["V"] * 4),
    )
    write(
        cache.with_suffix(".json"),
        {
            "schema": "rgb_port_history_features_v1",
            "status": "COMPLETE",
            "role": "V",
            "modality": "event",
            "output_sha256": sha(cache),
            "source_sha256": sha(observations),
            "normalizer_sha256": sha(normalizer),
        },
    )
    rows = owner / "V_ROWS.parquet"
    pd.DataFrame(
        {
            "sample_token": tokens,
            "group_id": groups,
            "target_phase": _phase(ttc),
            "rgb_available": [True] * 4,
        }
    ).to_parquet(rows, index=False)
    write(
        owner / "V_MANIFEST.json",
        {
            "group_ids": ["g1", "g2"],
            "rows_path": str(rows),
            "rows_sha256": sha(rows),
        },
    )
    checkpoint = tmp_path / "endpoint.pt"
    checkpoint.write_bytes(b"endpoint")
    endpoints = {
        "RT-M0-L0-S7": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint),
        }
    }
    write(run / "SCREEN_ENDPOINTS.json", endpoints)
    return owner, run, endpoints


def _rewrite_cache_groups(owner: Path, groups: np.ndarray) -> None:
    cache = owner / "features/V/E_HISTORY.npz"
    with np.load(cache, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["group_id"] = groups
    _save_npz(cache, **values)
    receipt = read(cache.with_suffix(".json"))
    receipt["output_sha256"] = sha(cache)
    write(cache.with_suffix(".json"), receipt)


def test_rejects_per_token_group_permutation_with_same_group_set(tmp_path: Path) -> None:
    owner, run, endpoints = _v_fixture(tmp_path)
    _rewrite_cache_groups(owner, np.asarray(["g2", "g1", "g2", "g1"]))
    with pytest.raises(ValueError, match="per sample token"):
        analyze.evaluate_v(owner, run, endpoints, stage="screen")


def test_rejects_missing_history_source_provenance(tmp_path: Path) -> None:
    owner, run, endpoints = _v_fixture(tmp_path)
    receipt_path = owner / "features/V/E_HISTORY.json"
    receipt = read(receipt_path)
    del receipt["source_sha256"]
    write(receipt_path, receipt)
    with pytest.raises(ValueError, match="source provenance"):
        analyze.evaluate_v(owner, run, endpoints, stage="screen")


def test_rejects_observations_from_different_parent(tmp_path: Path) -> None:
    owner, run, endpoints = _v_fixture(tmp_path)
    observations = owner / "features/V/E_OBSERVATIONS.npz"
    with np.load(observations, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["endpoint_freeze_sha256"] = np.asarray("wrong-parent")
    _save_npz(observations, **values)
    observation_receipt = read(observations.with_suffix(".json"))
    observation_receipt["output_sha256"] = sha(observations)
    write(observations.with_suffix(".json"), observation_receipt)
    history_receipt_path = owner / "features/V/E_HISTORY.json"
    history_receipt = read(history_receipt_path)
    history_receipt["source_sha256"] = sha(observations)
    write(history_receipt_path, history_receipt)
    with pytest.raises(ValueError, match="parent ancestry"):
        analyze.evaluate_v(owner, run, endpoints, stage="screen")


def test_cached_metrics_require_current_receipt_and_ancestry(tmp_path: Path, monkeypatch) -> None:
    owner, run, endpoints = _v_fixture(tmp_path)
    monkeypatch.setattr(
        analyze, "predict", lambda endpoint, values: (np.array([1, 4, 7, -2.0]), 0.1)
    )
    first = analyze.evaluate_v(owner, run, endpoints, stage="screen")
    assert first == analyze.evaluate_v(owner, run, endpoints, stage="screen")
    metrics_path = run / "evaluation/screen/RT-M0-L0-S7/METRICS.json"
    metrics = read(metrics_path)
    metrics["overall_MiD"] = 0.0
    write(metrics_path, metrics)
    with pytest.raises(ValueError, match="Cached V metrics binding"):
        analyze.evaluate_v(owner, run, endpoints, stage="screen")


def test_v_target_phase_uses_canonical_zero_boundary() -> None:
    phase = analyze._validated_target_phase(np.asarray([0.0], np.float32))
    assert phase.dtype.is_floating_point
    assert float(analyze.phase_to_ttc(phase)[0]) == 60.0


@pytest.mark.parametrize(
    "phase",
    [
        np.asarray([np.nan], np.float32),
        np.asarray([MIN_PHASE - 1e-3], np.float32),
        np.asarray([MAX_PHASE + 1e-3], np.float32),
    ],
)
def test_v_target_phase_rejects_nonfinite_or_out_of_support(phase: np.ndarray) -> None:
    with pytest.raises(ValueError, match="canonical finite support"):
        analyze._validated_target_phase(phase)


def test_trace_status_preserves_failures_and_stale_result_is_not_complete(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    write(root / "RESULT.json", {"status": "COMPLETE", "old": True})
    first = begin_trace(root)
    assert read(root / "STATUS.json")["status"] == "RUNNING"
    assert read_trace(root)["status"] == "RUNNING"
    fail_trace(root, first, RuntimeError("first failure"))
    failure = root / "attempts" / first / "FAILURE.json"
    assert failure.is_file()
    second = begin_trace(root)
    status = read(root / "STATUS.json")
    assert status["status"] == "RUNNING" and status["result_sha256"] is None
    assert failure.is_file()
    result = {"status": "COMPLETE", "attempt": second}
    write(root / "RESULT.json", result)
    complete_trace(root, second, result)
    final = read(root / "STATUS.json")
    assert final["status"] == "COMPLETE"
    assert read_trace(root) == result
    assert [item["status"] for item in final["attempts"]] == ["FAILED", "COMPLETE"]


def test_trace_status_archives_legacy_failure(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    write(root / "FAILURE.json", {"type": "OldError", "optimizer_updates": 0})
    begin_trace(root)
    assert not (root / "FAILURE.json").exists()
    legacy = [
        item for item in read(root / "STATUS.json")["attempts"] if item["status"] == "FAILED_LEGACY"
    ]
    assert len(legacy) == 1
    assert (root / legacy[0]["failure_path"]).is_file()


def test_paused_trace_is_terminal_but_never_reported_complete(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    attempt = begin_trace(root)
    result = {"status": "PAUSED_RESOURCE", "completed_queries": 2, "requested_queries": 32}
    write(root / "RESULT.json", result)
    finish_trace(root, attempt, result)
    status = read(root / "STATUS.json")
    assert status["status"] == "PAUSED_RESOURCE"
    assert status["result_sha256"] == sha(root / "RESULT.json")
    assert read_trace(root) == result


@pytest.mark.parametrize("waiting", ["WAITING_RAW_MANIFEST", "WAITING_RAW_ASSETS"])
def test_waiting_trace_result_stays_waiting(tmp_path: Path, waiting: str) -> None:
    root = tmp_path / "trace"
    attempt = begin_trace(root)
    result = {"status": waiting, "paths": ["missing-a", "missing-b"]}
    write(root / "RESULT.json", result)
    finish_trace(root, attempt, result)
    assert read_trace(root) == result


def test_terminal_trace_rejects_changed_result(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    attempt = begin_trace(root)
    result = {"status": "PAUSED_RESOURCE", "completed_queries": 2}
    write(root / "RESULT.json", result)
    finish_trace(root, attempt, result)
    write(root / "RESULT.json", {**result, "completed_queries": 32})
    with pytest.raises(ValueError, match="terminal trace result"):
        read_trace(root)
