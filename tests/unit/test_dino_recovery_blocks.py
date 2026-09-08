"""DINO recovery survives interrupted publication and rejects contaminated blocks."""

import json
from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, object_digest
from e_jepa_ttc.data.dinov3_relational_teacher_cache import DinoRecoveryBlocks


def arrays(tokens):
    n = len(tokens)
    return {
        "tokens": np.asarray(tokens),
        "tracks": np.asarray(["a"] * n),
        "targets": np.arange(n * 2 * 6 * 32 * 32, dtype=np.float32)
        .reshape(n, 2, 6, 32, 32)
        .astype(np.float16)
        / 1000,
        "valid": np.ones((n, 2, 6, 32, 32), dtype=np.uint8),
    }


def test_continuous_and_resume_identical(tmp_path: Path):
    block = DinoRecoveryBlocks(tmp_path, {"weights": "fixed"})
    first = arrays(["a", "b"])
    block.publish(first)
    (tmp_path / "block_000002.npz.tmp").write_bytes(b"interrupted")
    resumed = DinoRecoveryBlocks(tmp_path, {"weights": "fixed"})
    assert resumed.rows == 2
    second = arrays(["c"])
    resumed.publish(second)
    actual = resumed.assemble("targets", 3)
    np.testing.assert_array_equal(actual, np.concatenate([first["targets"], second["targets"]]))
    del actual


def test_incomplete_and_duplicate_block_rejected(tmp_path: Path):
    block = DinoRecoveryBlocks(tmp_path, {})
    block.publish(arrays(["a"]))
    with pytest.raises(ValueError, match="incomplete"):
        block.assemble("targets", 2)
    with pytest.raises(ValueError, match="duplicate"):
        block.publish(arrays(["a"]))


def test_changed_identity_or_payload_rejected(tmp_path: Path):
    block = DinoRecoveryBlocks(tmp_path, {"weights": "fixed"})
    block.publish(arrays(["a"]))
    with pytest.raises(ValueError, match="identity"):
        DinoRecoveryBlocks(tmp_path, {"weights": "different"})
    with (tmp_path / "block_000000.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError):
        DinoRecoveryBlocks(tmp_path, {"weights": "fixed"})


def test_resigned_path_escape_rejected_before_read(tmp_path: Path):
    block = DinoRecoveryBlocks(tmp_path, {})
    block.publish(arrays(["a"]))
    path = tmp_path / "block_000000.json"
    receipt = json.loads(path.read_text())
    receipt.pop("artifact_sha256")
    receipt["payload"]["path"] = str(tmp_path.parent / "protected.npz")
    atomic_json(path, {**receipt, "artifact_sha256": object_digest(receipt)})
    with pytest.raises(ValueError, match="escaped"):
        DinoRecoveryBlocks(tmp_path, {})


def test_materializer_recovers_without_recomputing_rgb(tmp_path: Path, monkeypatch):
    """Interrupt final publication after durable inference, then resume production flow."""
    import hashlib
    from types import SimpleNamespace

    import pandas as pd
    import torch

    from scripts import materialize_dinov3_relational_teacher as m

    sequences = [
        "2cyv0Oedzg",
        "5ilM1PX2vz",
        "6h5yRW2LGc",
        "OBneIVg4Cw",
        "OYgB6RGWcq",
        "WbCh1DRerJ",
        "mHGFBekt7X",
        "qGsgzl4Q8B",
        "t79dBxj1WS",
    ]
    records = [
        {
            "sample_token": s + "_token",
            "track_id": "a",
            "sequence_id": s,
            "event_v4_common_square_xyxy": np.asarray([0, 0, 2, 2], np.float32),
        }
        for s in sequences
    ]
    manifest = tmp_path / "events.json"
    manifest.write_text("{}")
    metadata = tmp_path / "train.parquet"
    metadata.write_bytes(b"synthetic fixture")
    frame = pd.DataFrame(records)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: frame)
    monkeypatch.setattr(m, "GarlTTCObjectEventV4Dataset", lambda *a, **kw: records)
    monkeypatch.setattr(m, "observe_git_identity", lambda *a: None)
    monkeypatch.setattr(
        m, "serialize_git_identity", lambda *a: {"git_commit": "synthetic", "git_dirty": False}
    )
    monkeypatch.setattr(m, "require_clean_scientific_worktree", lambda *a: None)
    model = torch.nn.Identity()
    model.config = SimpleNamespace(to_dict=lambda: {}, _commit_hash="fixed")
    monkeypatch.setattr(
        m,
        "_load_offline_teacher",
        lambda: (model, SimpleNamespace(image_mean=[0] * 3, image_std=[1] * 3)),
    )
    monkeypatch.setattr(m, "_EXPECTED_WEIGHTS_SHA256", hashlib.sha256().hexdigest())
    monkeypatch.setattr(m, "_find_32x32_feature", lambda *a: (2, "hidden_states"))
    monkeypatch.setattr(
        m, "_extract_features", lambda model, x, *a: torch.ones(len(x), 384, 32, 32)
    )
    endpoints = {
        "rgb_shards": ["a", "b"],
        "rgb_members": ["a", "b"],
        "indices": [0, 1],
        "frame_timestamps_us": [0, 1],
        "event_windows_us": [[0, 1], [1, 2]],
    }
    monkeypatch.setattr(m, "_resolve_rgb_endpoints", lambda *a: endpoints)
    calls = []

    def crop(**kw):
        calls.append(kw)
        return torch.zeros(1, 3, 256, 256), "a" * 64

    monkeypatch.setattr(m, "_load_and_crop_rgb", crop)
    actual_writer = m.write_complete_mmap_cache

    def interrupted(*a, **kw):
        raise OSError("simulated interruption before final publication")

    monkeypatch.setattr(m, "write_complete_mmap_cache", interrupted)
    arguments = (manifest, metadata, tmp_path, m._EXPECTED_MODEL_ID, tmp_path / "output", "cpu", 9)
    with pytest.raises(OSError, match="simulated interruption"):
        m.run(*arguments)
    assert len(calls) == 18
    assert not (tmp_path / "output/manifest.json").exists()
    monkeypatch.setattr(m, "write_complete_mmap_cache", actual_writer)
    result = m.run(*arguments)
    assert len(calls) == 18
    assert result["row_count_observed"] == 9
    with pytest.raises(FileExistsError, match="Complete DINO cache"):
        m.run(*arguments)
