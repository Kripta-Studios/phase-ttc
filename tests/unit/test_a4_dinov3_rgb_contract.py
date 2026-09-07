"""Unit tests for the A4 RGB↔event endpoint binding contract."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.data.event_v4_geometry import box_in_common_roi
from scripts.materialize_dinov3_relational_teacher import (
    _capacity_teacher_scope,
    _load_offline_teacher,
    _resolve_rgb_endpoints,
)


def _teacher_scope_fixture(tmp_path: Path, *, forbidden: bool = False) -> dict[str, Any]:
    def write(name: str, value: dict[str, Any]) -> dict[str, Any]:
        path = tmp_path / name
        atomic_json(path, {**value, "artifact_sha256": object_digest(value)})
        return binding(path)

    metadata = tmp_path / "train_metadata.parquet"
    metadata.write_bytes(b"synthetic metadata binding; no reader invoked")
    cohort = write(
        "cohort.json",
        {
            "kind": "CAPACITY_DEVELOPMENT_COHORT",
            "selected_sequences": ["extra_sequence"],
            "ordered_tokens": ["extra_token"],
            "confirmation_sequences": [],
            "protected_sequences": [],
        },
    )
    roles = write(
        "roles.json",
        {
            "roles": {
                "sequence_to_group": [["extra_sequence", "acquisition"]],
                "confirmation": ["acquisition"] if forbidden else [],
                "protected": [],
            }
        },
    )
    partitions = write("partitions.json", {"outputs": {"train_metadata": binding(metadata)}})
    return write(
        "preparation.json",
        {
            "cohort": cohort,
            "fixed_original_roles": roles,
            "source_partitions": partitions,
        },
    )


def test_expanded_teacher_scope_uses_signed_train_cohort(tmp_path: Path) -> None:
    source = _teacher_scope_fixture(tmp_path)
    actual, sequences, tokens = _capacity_teacher_scope(
        Path(source["path"]), source["sha256"], tmp_path / "train_metadata.parquet"
    )
    assert actual == source
    assert sequences == {"extra_sequence"} and tokens == {"extra_token"}


@pytest.mark.parametrize("violation", ["acquisition", "hash", "path", "changed_metadata"])
def test_expanded_teacher_scope_rejects_invalid_bindings(tmp_path: Path, violation: str) -> None:
    source = _teacher_scope_fixture(tmp_path, forbidden=violation == "acquisition")
    metadata = tmp_path / "train_metadata.parquet"
    if violation == "changed_metadata":
        metadata.write_bytes(b"changed")
    if violation == "path":
        metadata = tmp_path / "confirmation.parquet"
    with pytest.raises((ValueError, RuntimeError)):
        _capacity_teacher_scope(
            Path(source["path"]), "0" * 64 if violation == "hash" else source["sha256"], metadata
        )


def test_teacher_loading_pins_both_local_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    model_loader = Mock(return_value=object())
    processor_loader = Mock(return_value=object())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=model_loader),
            AutoImageProcessor=SimpleNamespace(from_pretrained=processor_loader),
        ),
    )
    model, processor = _load_offline_teacher()
    assert model is model_loader.return_value
    assert processor is processor_loader.return_value
    for loader in (model_loader, processor_loader):
        loader.assert_called_once_with(
            "facebook/dinov3-convnext-large-pretrain-lvd1689m",
            revision="e959efa74c867491dcfe3ec3e4f97382e39025b3",
            local_files_only=True,
            trust_remote_code=False,
        )


@pytest.mark.parametrize("missing", ["model", "processor"])
def test_teacher_missing_local_dependency_has_no_remote_fallback(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    model_loader, processor_loader = Mock(), Mock()
    loaders = {"model": model_loader, "processor": processor_loader}
    loaders[missing].side_effect = OSError("local dependency missing")
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=model_loader),
            AutoImageProcessor=SimpleNamespace(from_pretrained=processor_loader),
        ),
    )
    with pytest.raises(OSError, match="local dependency missing"):
        _load_offline_teacher()
    assert model_loader.call_count == 1
    assert processor_loader.call_count == (0 if missing == "model" else 1)


def _record_and_row() -> tuple[dict[str, object], dict[str, object]]:
    boxes = [
        (10.0, 10.0, 20.0, 20.0),
        (20.0, 20.0, 40.0, 40.0),  # t1 selected by event cache
        (30.0, 30.0, 50.0, 50.0),
        (35.0, 35.0, 60.0, 60.0),
        (40.0, 40.0, 80.0, 80.0),  # t2 selected by event cache
        (45.0, 45.0, 90.0, 90.0),
    ]
    square = (0.0, 0.0, 100.0, 100.0)
    mapped_t1 = box_in_common_roi(boxes[1], square, roi_size=128)
    mapped_t2 = box_in_common_roi(boxes[4], square, roi_size=128)
    record: dict[str, object] = {
        "event_v4_common_square_xyxy": np.asarray(square, dtype=np.float32),
        "event_v4_boxes_xyxy": np.asarray(
            [[0.0, 0.0, 1.0, 1.0], mapped_t1, mapped_t2], dtype=np.float32
        ),
        "event_v4_common_roi": np.zeros((3, 12, 128, 128), dtype=np.float32),
        "garl_delta_t_s": 0.3,
    }
    timestamps = [1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000, 1_500_000]
    row: dict[str, object] = {
        "rgb_shard_paths": [f"s{i}.tar" for i in range(6)],
        "rgb_member_paths": [f"f{i}.jpg" for i in range(6)],
        "frame_timestamps_us": timestamps,
        "event_windows_us": [[t - 50_000, t] for t in timestamps],
        "boxes_xyxy": boxes,
    }
    return record, row


def test_resolve_rgb_endpoints_binds_exact_event_cache_pair() -> None:
    record, row = _record_and_row()
    resolved = _resolve_rgb_endpoints(record, row)
    assert resolved["indices"] == (1, 4)
    assert resolved["rgb_shards"] == ("s1.tar", "s4.tar")
    assert resolved["rgb_members"] == ("f1.jpg", "f4.jpg")
    assert resolved["frame_timestamps_us"] == (1_100_000, 1_400_000)


def test_resolve_rgb_endpoints_rejects_unaligned_metadata() -> None:
    record, row = _record_and_row()
    row["rgb_member_paths"] = list(row["rgb_member_paths"])[:-1]  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unaligned RGB/event metadata lengths"):
        _resolve_rgb_endpoints(record, row)


def test_resolve_rgb_endpoints_rejects_delta_t_mismatch() -> None:
    record, row = _record_and_row()
    record["garl_delta_t_s"] = 0.2
    with pytest.raises(ValueError, match="could not uniquely bind RGB endpoints"):
        _resolve_rgb_endpoints(record, row)
