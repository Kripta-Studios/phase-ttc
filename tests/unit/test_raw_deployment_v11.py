"""Raw deployment preserves fixed clocks, roles, calibration and full count representation."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding
from e_jepa_ttc.data import raw_deployment_v11 as module
from e_jepa_ttc.data.event_context128 import RateNormalization


def fixture(tmp_path, monkeypatch):
    rows = pd.DataFrame(
        [
            {
                "sample_token": "train",
                "window_id": i,
                "sequence_id": "sequence",
                "window_start_us": 100000 + i * 100000,
                "window_end_us": 200000 + i * 100000,
                "frame_to_event_clock_offset_us": 0,
                "target_anchor_us": 300000,
                "roi_x0": 5.0,
                "roi_y0": 0.0,
                "roi_x1": 133.0,
                "roi_y1": 128.0,
                "event_x_offset_px": 5.0,
                "polarity_encoding": "zero_one",
                "h5_file_sha256": "c" * 64,
            }
            for i in (0, 1)
        ]
    )
    csv = tmp_path / "bindings.csv"
    rows.to_csv(csv, index=False)
    charter = tmp_path / "charter.json"
    atomic_json(
        charter,
        {
            "status": "ORIGINAL_8192_CLOCK_AND_ROI_VERIFIED",
            "raw_files": {"sequence": {"sha256": "c" * 64}},
        },
    )
    seen = []

    class Reader:
        def __init__(self, access, hashes):
            self.source_hashes, self.memo = hashes, {"stale": "must be cleared"}

        def count(self, row, window):
            assert not self.memo
            seen.append(window)
            return np.full((8, 2, 128, 128), row.name[1], np.uint32), {}

        def close(self):
            seen.append("closed")

    monkeypatch.setattr(module, "WindowReader", Reader)
    view = SimpleNamespace(
        tokens=("train",),
        identity="a" * 64,
        reader=SimpleNamespace(
            metadata=pd.DataFrame({"sample_token": ["train"], "sequence_id": ["sequence"]})
        ),
    )
    raw = module.RawDeploymentInputs(binding(csv), binding(charter), None, view)
    return raw, RateNormalization((0.0, 0.0), (1.0, 1.0), "a" * 64, 10), seen


def test_raw_input_keeps_full_shape_actual_relative_clock_and_handle_closure(tmp_path, monkeypatch):
    raw, norm, seen = fixture(tmp_path, monkeypatch)
    sample = raw.inputs("train", norm)
    assert sample.contexts.shape == (4, 8, 128, 128)
    torch.testing.assert_close(sample.times, torch.tensor([-0.175, -0.125, -0.075, -0.025]))
    assert sample.contexts[:2].count_nonzero() == 0
    assert (sample.contexts[2:] > 0).all()
    assert len(seen) == 2
    raw.close()
    assert seen[-1] == "closed"


def test_unexposed_token_rejected_before_any_raw_read(tmp_path, monkeypatch):
    raw, norm, seen = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="outside its TRAIN"):
        raw.inputs("held_out", norm)
    assert not seen


def test_changed_window_hash_rejected_even_if_hdf5_handle_is_already_open(tmp_path, monkeypatch):
    raw, norm, seen = fixture(tmp_path, monkeypatch)
    raw.index.loc[("train", 1), "h5_file_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="physical source hash"):
        raw.inputs("train", norm)
    assert not seen
