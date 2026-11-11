"""Integrity and sequence-resampling checks for the standalone paper revision."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "paper/from_motion_to_time_20261102/scripts/build_transfer_revision.py"
)
spec = importlib.util.spec_from_file_location("corrected_transfer_statistics", SCRIPT)
revision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(revision)


def fixture_root(tmp_path):
    (tmp_path / "data").mkdir()
    audit = tmp_path / "audit_20261104"
    audit.mkdir()
    row = {
        "query_id": "test:150000",
        "group_id": "test",
        "cohort": "extended_new",
        "target_ttc": "8.0",
        "target_exclusion": "",
        **{model: "7.0" for model in revision.MODELS},
        "legacy_target_ttc": "8.0",
        "legacy_target_exclusion": "",
        "anchor_us": "150000",
        "target_timestamp_us": "100000",
        "target_delta_us": "50000",
        "neighbor_before_us": "100000",
        "neighbor_before_ttc": "8.0",
        "neighbor_after_us": "200000",
        "neighbor_after_ttc": "9.0",
        "target_source_sha256": "a" * 64,
    }
    paths = (audit / "corrected_transfer_candidates.csv", tmp_path / "data/transfer_queries.csv")
    for path in paths:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    metadata = {
        "corrected_projection_sha256": revision.digest(audit / "corrected_transfer_candidates.csv"),
        "historical_projection_sha256": revision.digest(tmp_path / "data/transfer_queries.csv"),
        "corrected_target_source_sha256": {"test": "a" * 64},
    }
    (audit / "transfer_reassociation.json").write_text(json.dumps(metadata))
    return tmp_path


def test_projection_tampering_fails_before_statistics(tmp_path):
    root = fixture_root(tmp_path)
    path = root / "audit_20261104/corrected_transfer_candidates.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="projection hash"):
        revision.verified_rows(root)


def test_nearest_target_accepts_50ms_and_earlier_tie(tmp_path):
    rows, _ = revision.verified_rows(fixture_root(tmp_path))
    assert rows[0]["target_ttc"] == "8.0"


def test_bound_neighbors_detect_incorrect_target_even_with_updated_hash(tmp_path):
    root = fixture_root(tmp_path)
    path = root / "audit_20261104/corrected_transfer_candidates.csv"
    path.write_text(path.read_text().replace("8.0", "9.0", 1))
    metadata_path = root / "audit_20261104/transfer_reassociation.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["corrected_projection_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="nearest target"):
        revision.verified_rows(root)


def test_resealed_projection_cannot_change_saved_predictions(tmp_path):
    root = fixture_root(tmp_path)
    path = root / "audit_20261104/corrected_transfer_candidates.csv"
    path.write_text(path.read_text().replace("7.0", "7.5", 1))
    metadata_path = root / "audit_20261104/transfer_reassociation.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["corrected_projection_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Saved predictions"):
        revision.verified_rows(root)


def test_sequence_bootstrap_keeps_large_group_whole_and_distinguishes_macro():
    candidate = np.array([2.0] * 20 + [10.0])
    reference = np.ones(21)
    groups = np.array(["large"] * 20 + ["small"])
    pooled = revision.paired_bootstrap(candidate, reference, groups, draws=1000, seed=7)
    macro = revision.paired_bootstrap(candidate, reference, groups, draws=1000, seed=7, macro=True)
    assert pooled["mean"]["point"] == pytest.approx(29 / 21)
    assert pooled["mean"]["CI95"] == pytest.approx([1, 9])
    assert macro["mean"]["point"] == pytest.approx(5)
    assert macro["mean"]["CI95"] == pytest.approx([1, 9])
