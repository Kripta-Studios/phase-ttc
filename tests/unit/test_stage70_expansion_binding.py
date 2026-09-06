"""Role-safe expansion source resolution and exact empty-window binding."""

import h5py
import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.data.stage70_expansion_binding import ExpansionTrainAccess, expansion_bindings


@pytest.mark.parametrize("outer_fold", [None, 0, 1, 2])
def test_expansion_binding_preserves_empty_window_and_reuses_complete_output(tmp_path, outer_fold):
    root = tmp_path / "train"
    sequence = root / "extra"
    sequence.mkdir(parents=True)
    with h5py.File(sequence / "events.h5", "w") as handle:
        for name, values in {
            "x": np.zeros(32),
            "y": np.zeros(32),
            "t": np.arange(32, dtype=np.int64),
            "p": np.arange(32) % 2,
        }.items():
            handle.create_dataset(f"events/{name}", data=values)
        handle.create_dataset("ms_to_idx", data=np.array([0], dtype=np.int64))
    metadata_path = tmp_path / "metadata.parquet"
    metadata = pd.DataFrame(
        {
            "sample_token": ["unique"],
            "sequence_id": ["extra"],
            "track_id": ["track"],
            "timestamp_us": [148],
            "frame_timestamps_us": [[116, 148]],
            "event_windows_us": [[[0, 16], [32, 48]]],
            "boxes_xyxy": [[[10, 10, 20, 20], [10, 10, 20, 20]]],
            "events_path": ["extra/events.h5"],
        }
    )
    if outer_fold is not None:
        metadata["outer_fold"] = outer_fold
    metadata.to_parquet(metadata_path, index=False)
    access = ExpansionTrainAccess(
        root,
        metadata_path,
        allowed_sequences=frozenset({"extra"}),
        journal_path=tmp_path / "reads.jsonl",
        resource_check=lambda: None,
    )
    output = tmp_path / "bindings"
    result = expansion_bindings(metadata_path, access, output)
    assert result["zero_event_windows_retained"] == 1
    bound = pd.read_csv(output / "RAW_BINDINGS.csv")
    assert set(bound.outer_fold) == {-1 if outer_fold is None else outer_fold}
    assert bound.start_event_index.tolist() == [0, 32]
    assert bound.stop_event_index.tolist() == [16, 32]
    assert bound.target_anchor_event_clock_us.tolist() == [48, 48]
    stamp = (output / "RAW_BINDINGS.csv").stat().st_mtime_ns
    assert expansion_bindings(metadata_path, access, output) == result
    assert (output / "RAW_BINDINGS.csv").stat().st_mtime_ns == stamp
    with pytest.raises(ValueError, match="not allowed"):
        access.event_path("confirmation", "confirmation/events.h5", "must not read")
    with pytest.raises(ValueError, match="exact TRAIN"):
        access.event_path("extra", "../extra/events.h5", "must not read")
