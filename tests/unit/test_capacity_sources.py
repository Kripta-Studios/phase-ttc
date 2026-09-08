"""Selected TRAIN clock/target identity and protected-path rejection with synthetic files."""

import h5py
import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.data.capacity_sources import materialize_capacity_counts, validate_capacity_rows
from e_jepa_ttc.data.event_context128_reader import CountReader


def signed(path, value):
    atomic_json(path, {**value, "artifact_sha256": object_digest(value)})
    return binding(path)


@pytest.mark.parametrize("failure", [None, "hash", "output", "mixed_mode"])
def test_count_reconstruction_cli_preserves_fixed_source(tmp_path, monkeypatch, failure):
    import sys

    from e_jepa_ttc.data import capacity_sources
    from scripts import prepare_stage70_capacity as compiler

    source_path = tmp_path / "ROW_USABILITY.json"
    source_path.write_text("{}", encoding="utf-8")
    source = binding(source_path)
    output = tmp_path / "artifacts/stage70_76_capacity/representation_30000"
    arguments = [
        "prepare_stage70_capacity.py",
        "--count-usability",
        str(source_path),
        "--count-usability-sha256",
        source["sha256"] if failure != "hash" else "0" * 64,
        "--output-root",
        str(output if failure != "output" else tmp_path / "historical"),
    ]
    if failure == "mixed_mode":
        arguments += ["--comparator-inputs", "unused.json"]
    calls = []

    def materialize(actual_source, actual_output, *, build_cache):
        calls.append((actual_source, actual_output, build_cache))
        return {"status": "COUNTS_COMPLETE", "optimizer_updates": 0}

    monkeypatch.setattr(compiler, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(capacity_sources, "materialize_capacity_counts", materialize)
    if failure:
        with pytest.raises(SystemExit if failure == "mixed_mode" else ValueError):
            compiler.main()
        assert calls == []
    else:
        compiler.main()
        assert calls == [(source, output, True)]


def inputs(tmp_path, *, bad_target=False, protected=False, bad_path=False):
    seqs = ["s0", "s1", "s2"]
    tokens = ["t0", "t1", "t2"]
    metadata = pd.DataFrame(
        {
            "sample_token": tokens,
            "sequence_id": seqs,
            "track_id": ["track"] * 3,
            "timestamp_us": [148] * 3,
            "frame_timestamps_us": [[116, 148]] * 3,
            "event_windows_us": [[[0, 16], [32, 48]]] * 3,
            "boxes_xyxy": [[[10, 10, 20, 20], [10, 10, 20, 20]]] * 3,
            "events_path": [f"{s}/events.h5" for s in seqs],
        }
    )
    metadata.to_parquet(tmp_path / "train_metadata.parquet", index=False)
    labels = pd.DataFrame(
        {
            "sample_token": tokens,
            "sequence_id": seqs,
            "timestamp_us": [148] * 3,
            "ttc": [2.0] * 3,
            "frame_ttc": [[3.0, 9.0 if bad_target else 2.0]] * 3,
        }
    )
    label_path = tmp_path / ("confirmation.parquet" if bad_path else "train_labels.parquet")
    labels.to_parquet(label_path, index=False)
    partition = signed(
        tmp_path / "ROLE_PARTITIONS.json",
        {
            "outputs": {
                "train_metadata": binding(tmp_path / "train_metadata.parquet"),
                "train_labels": binding(label_path),
            }
        },
    )
    roles = signed(
        tmp_path / "ROLES.json",
        {
            "roles": {
                "sequence_to_group": [[s, s] for s in seqs],
                "confirmation": ["s0"] if protected else [],
                "protected": [],
            }
        },
    )
    cohort = signed(
        tmp_path / "COHORT.json",
        {
            "kind": "CAPACITY_DEVELOPMENT_COHORT",
            "selected_sequences": seqs,
            "ordered_tokens": tokens,
            "sequence_to_outer_fold": dict(zip(seqs, range(3), strict=True)),
        },
    )
    return signed(
        tmp_path / "PREPARATION.json",
        {"cohort": cohort, "source_partitions": partition, "fixed_original_roles": roles},
    )


def test_actual_metadata_label_clock_join_is_complete_and_replayable(tmp_path):
    source = inputs(tmp_path)
    out = tmp_path / "validated"
    result = validate_capacity_rows(source, out, lambda: None)
    assert result["status"] == "ALL_SELECTED_ROWS_VALID"
    assert result["selected_rows"] == 3 and result["replacement_rows"] == 0
    assert result["maximum_offset_drift_us"] == 0
    assert pd.read_parquet(result["metadata"]["path"]).outer_fold.tolist() == [0, 1, 2]
    assert validate_capacity_rows(source, out, lambda: None) == result


def test_bad_time_target_anchor_blocks_selected_universe_without_replacements(tmp_path):
    result = validate_capacity_rows(
        inputs(tmp_path, bad_target=True), tmp_path / "validated", lambda: None
    )
    assert result["status"] == "DATA_PREREQUISITE_BLOCKED"
    assert len(result["rejected"]) == 3 and "metadata" not in result


@pytest.mark.parametrize("attack", ["protected", "bad_path"])
def test_reserved_roles_and_escaped_label_partitions_fail_before_read(tmp_path, attack):
    source = inputs(tmp_path, **{attack: True})
    out = tmp_path / "validated"
    with pytest.raises(ValueError):
        validate_capacity_rows(source, out, lambda: None)
    assert not (out / "SOURCE_READS.jsonl").exists()


def test_actual_count_builder_preserves_new_folds_and_zero_windows(tmp_path):
    import json

    preparation_source = inputs(tmp_path)
    prep = json.loads((tmp_path / "PREPARATION.json").read_text())
    cohort = json.loads((tmp_path / "COHORT.json").read_text())
    cohort.pop("artifact_sha256")
    cohort["sequence_to_group"] = {s: s for s in cohort["selected_sequences"]}
    prep["cohort"] = signed(tmp_path / "COHORT.json", cohort)
    raw = tmp_path / "raw_train"
    for sequence in cohort["selected_sequences"]:
        directory = raw / sequence
        directory.mkdir(parents=True)
        with h5py.File(directory / "events.h5", "w") as handle:
            for key, value in {
                "x": np.zeros(32),
                "y": np.zeros(32),
                "t": np.arange(32),
                "p": np.arange(32) % 2,
            }.items():
                handle.create_dataset(f"events/{key}", data=value)
            # Distinct physical bytes are required for these declared distinct groups.
            handle.attrs["sequence"] = sequence
            handle.create_dataset("ms_to_idx", data=np.array([0]))
    local = tmp_path / "LOCAL.json"
    atomic_json(local, {"raw_train_root": str(raw)})
    old_cache = signed(tmp_path / "OLD_CACHE.json", {"identity": {"sources": {}}})
    old_lock = signed(
        tmp_path / "OLD_LOCK.json",
        {
            "local_paths": binding(local),
            "count_caches": {"original": old_cache},
            "execution_reservations": {"job": {"disk_bytes": 100000}},
        },
    )
    prep.pop("artifact_sha256")
    prep["predecessor_lock"] = old_lock
    preparation_source = signed(tmp_path / "PREPARATION.json", prep)
    validate_capacity_rows(preparation_source, tmp_path / "validated", lambda: None)
    result = materialize_capacity_counts(
        binding(tmp_path / "validated/ROW_USABILITY.json"), tmp_path / "physical", build_cache=True
    )
    assert result["status"] == "COUNTS_COMPLETE" and result["optimizer_updates"] == 0
    reader = CountReader(tmp_path / "physical/counts")
    assert reader.metadata.outer_fold.tolist() == [0, 1, 2]
    assert reader.observation("t0").counts.shape == (2, 8, 2, 128, 128)
