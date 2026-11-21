from __future__ import annotations

import numpy as np
import pandas as pd

from operational.matched_outer.evidence import build_query_contract, expand_level_lineage


def test_query_contract_uses_only_boxes_available_by_query() -> None:
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
    annotations = pd.DataFrame(
        [
            {"sample_token": "q0", "ttc": 2.0},
            {"sample_token": "q1", "ttc": 1.9},
        ]
    )
    result = build_query_contract(data, annotations, history_us=600)
    assert result.loc[0, "capture_start_us"] == 900
    assert result.loc[1, "capture_start_us"] == 900
    assert (result["raw_observation_end_us"] <= result["query_timestamp_us"]).all()
    assert (result["box_available_us"] <= result["query_timestamp_us"]).all()


def test_level_lineage_names_every_learned_boundary() -> None:
    weights = [
        {
            "component": "box_head",
            "level": "head",
            "learned": True,
            "outer_fold": 0,
            "seed": 7,
            "training_sequences": ["IN"],
            "outer_sequences": ["OUT"],
            "weight_sha256": "a" * 64,
        }
    ]
    rows = expand_level_lineage(weights)
    assert {row["level"] for row in rows} == {
        "source",
        "producer",
        "normalizer",
        "refiner",
        "head",
    }
    assert [row for row in rows if row["learned"]] == weights
