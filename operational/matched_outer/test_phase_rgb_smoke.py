from __future__ import annotations

import pytest

from operational.matched_outer.phase_rgb_smoke import select_fold_records


def test_phase_rgb_smoke_optimizer_never_sees_outer_sequence() -> None:
    records = [
        {"sequence_id": "TRAIN", "sample_token": "train"},
        {"sequence_id": "OUTER", "sample_token": "outer"},
    ]

    train, outer = select_fold_records(records, {"OUTER"})

    assert [row["sample_token"] for row in train] == ["train"]
    assert [row["sample_token"] for row in outer] == ["outer"]


def test_phase_rgb_smoke_requires_both_fold_roles() -> None:
    with pytest.raises(ValueError, match="both train and outer"):
        select_fold_records([{"sequence_id": "TRAIN", "sample_token": "train"}], {"OUTER"})
