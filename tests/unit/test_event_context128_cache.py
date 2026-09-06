"""Interrupted cache publication resumes verified shards and preserves identity."""

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import digest
from e_jepa_ttc.data import event_context128_cache as cache_module
from e_jepa_ttc.data.raw_event_binding import ReadOnlyTrainAccess


def fixture_inputs(tmp_path):
    raw_root = tmp_path / "train"
    sequence = "2cyv0Oedzg"
    (raw_root / sequence).mkdir(parents=True)
    hdf5 = raw_root / sequence / "events.h5"
    with h5py.File(hdf5, "w") as handle:
        for key, data in {
            "x": np.zeros(32, dtype=np.int16),
            "y": np.zeros(32, dtype=np.int16),
            "t": np.arange(32, dtype=np.int64),
            "p": np.arange(32, dtype=np.uint8) % 2,
        }.items():
            handle.create_dataset(f"events/{key}", data=data)
        handle.create_dataset("ms_to_idx", data=np.array([0], dtype=np.int64))
    sha = digest(hdf5)
    rows = []
    for token in ("alpha", "beta"):
        for window in range(2):
            rows.append(
                {
                    "sample_token": token,
                    "sequence_id": sequence,
                    "track_id": token,
                    "outer_fold": 0,
                    "window_id": window,
                    "window_start_us": window * 16,
                    "window_end_us": (window + 1) * 16,
                    "start_event_index": window * 16,
                    "stop_event_index": (window + 1) * 16,
                    "events_path_relative": f"{sequence}/events.h5",
                    "h5_file_sha256": sha,
                    "roi_x0": 5.0,
                    "roi_y0": 0.0,
                    "roi_x1": 21.0,
                    "roi_y1": 16.0,
                    "event_x_offset_px": 5.0,
                    "polarity_encoding": "zero_one",
                    "frame_to_event_clock_offset_us": 100,
                    "target_anchor_us": 132,
                }
            )
    binding_csv = tmp_path / "binding.csv"
    pd.DataFrame(rows).to_csv(binding_csv, index=False)
    metadata = tmp_path / "train.parquet"
    metadata.write_bytes(b"unused fixture metadata path")
    source = {
        sequence: {
            "path": str(hdf5.resolve()),
            "bytes": hdf5.stat().st_size,
            "mtime_ns": hdf5.stat().st_mtime_ns,
            "sha256": sha,
        }
    }
    return binding_csv, source, ReadOnlyTrainAccess(raw_root, metadata)


def test_cache_resume_keeps_complete_shard_and_exact_row_identity(tmp_path, monkeypatch):
    binding_csv, sources, access = fixture_inputs(tmp_path)
    output = tmp_path / "cache"
    original = cache_module.rasterize128
    calls = 0

    def interrupt_after_first_shard(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(cache_module, "rasterize128", interrupt_after_first_shard)
    with pytest.raises(RuntimeError, match="injected interruption"):
        cache_module.build_count_cache(binding_csv, sources, access, output, shard_rows=1)
    assert not (output / "CACHE_MANIFEST.json").exists()
    progress = json.loads((output / "CACHE_PROGRESS.json").read_text())
    assert progress["completed_rows"] == 1
    first = output / "counts_00000.npy"
    first_identity = digest(first), first.stat().st_mtime_ns
    monkeypatch.setattr(cache_module, "rasterize128", original)
    result = cache_module.build_count_cache(binding_csv, sources, access, output, shard_rows=1)
    assert result["status"] == "COMPLETE_COUNTS_ONLY"
    assert (digest(first), first.stat().st_mtime_ns) == first_identity
    counts = np.load(first, allow_pickle=False)
    assert counts.shape == (1, 2, 8, 2, 128, 128) and counts.sum() == 32
    assert (
        cache_module.build_count_cache(binding_csv, sources, access, output, shard_rows=1) == result
    )
    (output / "metadata.csv").write_text("tampered row identity")
    with pytest.raises(ValueError, match="component"):
        cache_module.build_count_cache(binding_csv, sources, access, output, shard_rows=1)
