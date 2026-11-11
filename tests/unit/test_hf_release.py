"""Release boundaries: reject corrupted packages and noncausal example inputs."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from release.huggingface.predict import load_input, verify_files


def sample(tmp_path, **changes):
    meta = {
        "schema": "phase_ttc_prepared_input_v1",
        "query_us": 300000,
        "observations": [
            {
                "modality": "event",
                "pixels": "e0",
                "timestamps_us": [100000, 200000, 300000],
                "first_sensor_ts_us": 0,
                "anchor_us": 300000,
                "available_us": 300000,
            }
        ],
    }
    meta.update(changes)
    path = tmp_path / "input.npz"
    np.savez_compressed(
        path, metadata=np.array(json.dumps(meta)), e0=np.zeros((3, 12, 128, 128), np.float32)
    )
    return path


def test_manifest_rejects_changed_bytes(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"original")
    manifest = {"model.pt": hashlib.sha256(b"original").hexdigest()}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    verify_files(tmp_path)
    (tmp_path / "model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="model.pt"):
        verify_files(tmp_path)


def test_manifest_rejects_escaping_paths(tmp_path):
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"../outside": "bad"}))
    with pytest.raises(ValueError, match="path"):
        verify_files(tmp_path)


@pytest.mark.parametrize(
    "extra", ["source/torch.py", "source/__pycache__/torch.pyc", "model/extra.pt"]
)
def test_manifest_rejects_unlisted_executable_or_model(tmp_path, extra):
    (tmp_path / "README.md").write_bytes(b"release")
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps({"README.md": hashlib.sha256(b"release").hexdigest()})
    )
    path = tmp_path / extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="Unexpected release file"):
        verify_files(tmp_path)


def test_extractor_verifies_source_before_import(tmp_path):
    source = Path(__file__).resolve().parents[2] / "release/huggingface"
    manifest = {}
    for name in ["predict.py", "extract_example.py"]:
        raw = (source / name).read_bytes()
        (tmp_path / name).write_bytes(raw)
        manifest[name] = hashlib.sha256(raw).hexdigest()
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    (tmp_path / "source").mkdir()
    (tmp_path / "source/torch.py").write_text("raise RuntimeError('must not execute')")
    result = subprocess.run(
        [
            sys.executable,
            str(tmp_path / "extract_example.py"),
            "--data-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "input.npz"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Unexpected release file: source/torch.py" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_input_accepts_declared_sensor_arrays(tmp_path):
    meta, arrays = load_input(sample(tmp_path))
    assert meta["query_us"] == 300000 and arrays["e0"].shape == (3, 12, 128, 128)


def test_input_rejects_targets(tmp_path):
    with pytest.raises(ValueError, match="field"):
        load_input(sample(tmp_path, target_ttc=2.0))


def test_input_rejects_future_observations(tmp_path):
    with pytest.raises(ValueError, match="causal"):
        load_input(sample(tmp_path, query_us=200000))


def test_input_rejects_future_boxes(tmp_path):
    with pytest.raises(ValueError, match="causal"):
        load_input(sample(tmp_path, boxes_xyxy=[[0, 0, 10, 10]], box_timestamps_us=[400000]))
