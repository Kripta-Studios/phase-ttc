"""Comparator replay blocks preserve exact rows/features across interruption."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, object_digest
from scripts.train_scientific_recovery_v8_router_expert import _recoverable_replay_blocks


def fixture_rows(count=300):
    return pd.DataFrame(
        {
            "sample_token": [f"t{i}" for i in range(count)],
            "track_id": ["001"] * count,
            "prediction_ttc_s": np.linspace(0.1, 9, count, dtype=np.float32),
        }
    )


def test_interrupted_replay_resumes_without_recomputing_complete_block(tmp_path: Path):
    metadata = fixture_rows()
    pair = np.arange(300 * 133, dtype=np.float32).reshape(300, 133) / 1000
    calls = []

    def interrupted(first, last):
        calls.append((first, last))
        if first:
            raise OSError("power interruption")
        return metadata.iloc[first:last].copy(), pair[first:last].copy()

    with pytest.raises(OSError, match="power interruption"):
        _recoverable_replay_blocks(
            tmp_path, metadata, {"producer": "A5", "role": "train"}, 16, interrupted, lambda: None
        )
    assert calls == [(0, 256), (256, 300)]
    calls.clear()

    def complete(first, last):
        calls.append((first, last))
        return metadata.iloc[first:last].copy(), pair[first:last].copy()

    rows, features = _recoverable_replay_blocks(
        tmp_path, metadata, {"producer": "A5", "role": "train"}, 16, complete, lambda: None
    )
    assert calls == [(256, 300)]
    pd.testing.assert_frame_equal(rows, metadata)
    np.testing.assert_array_equal(features, pair)
    calls.clear()
    _recoverable_replay_blocks(
        tmp_path, metadata, {"producer": "A5", "role": "train"}, 16, complete, lambda: None
    )
    assert not calls


def test_role_change_and_resigned_payload_escape_rejected(tmp_path: Path):
    metadata = fixture_rows(2)

    def evaluate(first, last):
        return metadata.iloc[first:last].copy(), None

    _recoverable_replay_blocks(tmp_path, metadata, {"role": "train"}, 16, evaluate, lambda: None)
    with pytest.raises(ValueError, match="producer/role/tokens"):
        _recoverable_replay_blocks(
            tmp_path, metadata, {"role": "prediction"}, 16, evaluate, lambda: None
        )
    receipt = tmp_path / "block_000000/BLOCK.json"
    record = json.loads(receipt.read_text())
    record.pop("artifact_sha256")
    record["arrays"]["path"] = str(tmp_path.parent / "protected.npz")
    atomic_json(receipt, {**record, "artifact_sha256": object_digest(record)})
    with pytest.raises(ValueError, match="escaped"):
        _recoverable_replay_blocks(
            tmp_path, metadata, {"role": "train"}, 16, evaluate, lambda: None
        )


def test_source_reorder_and_bad_feature_rows_rejected(tmp_path: Path):
    metadata = fixture_rows(2)
    with pytest.raises(ValueError, match="reordered"):
        _recoverable_replay_blocks(
            tmp_path, metadata, {}, 16, lambda a, b: (metadata.iloc[::-1], None), lambda: None
        )
    invalid = tmp_path / "invalid"
    with pytest.raises(ValueError, match="misaligned"):
        _recoverable_replay_blocks(
            invalid,
            metadata,
            {"collect_pair": True},
            16,
            lambda a, b: (metadata.copy(), np.zeros((1, 133), np.float32)),
            lambda: None,
        )
    assert not (invalid / "block_000000/BLOCK.json").exists()
