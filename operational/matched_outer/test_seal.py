from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from operational.matched_outer.seal import relative_artifact_path, verify_geometry_binding
from operational.phase_geometry.geometry import compute_table, learned_inputs


def test_geometry_binding_detects_cached_feature_change() -> None:
    data = pd.DataFrame(
        [
            {
                "sequence_id": "S",
                "sample_token": "q0",
                "track_id": "T",
                "timestamp_us": 1_000,
                "frame_timestamps_us": np.asarray([900, 1_000]),
                "boxes_xyxy": np.asarray([[0, 0, 10, 10], [0, 0, 11, 11]]),
            },
            {
                "sequence_id": "S",
                "sample_token": "q1",
                "track_id": "T",
                "timestamp_us": 1_100,
                "frame_timestamps_us": np.asarray([1_000, 1_100]),
                "boxes_xyxy": np.asarray([[0, 0, 11, 11], [0, 0, 12, 12]]),
            },
        ]
    )
    raw, _ = compute_table(
        data["sequence_id"],
        data["track_id"],
        data["timestamp_us"],
        data["frame_timestamps_us"].tolist(),
        data["boxes_xyxy"].tolist(),
    )
    arrays = {
        "sample_token": data["sample_token"].to_numpy(),
        "geometry_raw": raw,
        "geometry_g": learned_inputs(raw),
    }
    assert verify_geometry_binding(arrays, data)["status"] == "pass"
    arrays["geometry_raw"] = raw.copy()
    arrays["geometry_raw"][0, 0] += 1.0
    with pytest.raises(ValueError, match="geometry_raw"):
        verify_geometry_binding(arrays, data)


def test_relative_artifact_path_accepts_relative_input(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    assert relative_artifact_path(Path("run/file.npz"), root, cwd=root) == "run/file.npz"
