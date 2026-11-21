from __future__ import annotations

from operational.matched_outer.smoke import split_probe_records


def test_smoke_training_records_exclude_outer_sequences() -> None:
    records = [
        {"sequence_id": "IN", "sample_token": "i"},
        {"sequence_id": "OUT", "sample_token": "o"},
    ]
    train, outer = split_probe_records(records, {"OUT"})
    assert [row["sample_token"] for row in train] == ["i"]
    assert [row["sample_token"] for row in outer] == ["o"]
