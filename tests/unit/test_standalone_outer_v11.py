"""No-fit checks for source-isolated outer inference and durable numeric publication."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.evaluation.standalone_outer_v11 import (
    _publish_arrays,
    infer_initial_endpoint,
    validate_outer_view,
    verify_inference_shards,
)
from e_jepa_ttc.training.stage72_jobs import FitJob
from e_jepa_ttc.training.stage73_factorial import PlannedFit


@pytest.fixture
def fixture():
    job = FitJob(
        PlannedFit("S72", "S72-DIRECT", 0, 7),
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
        ("train_group",),
        ("outer_group",),
        "f" * 64,
    )
    reader = SimpleNamespace(
        identity="cache",
        metadata=pd.DataFrame(
            {
                "sample_token": ["outer0", "outer1", "train0"],
                "sequence_id": ["outer_seq", "outer_seq", "train_seq"],
                "outer_fold": [0, 0, 1],
            }
        ),
    )
    lock = {
        "outer_count_cache_sha256": "cache",
        "outer_tokens_by_fold": {"0": ["outer0", "outer1"]},
        "sequence_to_group": {"outer_seq": "outer_group", "train_seq": "train_group"},
    }
    return job, reader, lock


def test_outer_requires_exact_fold_and_all_transitive_acquisition_exclusions(fixture):
    job, reader, lock = fixture
    validate_outer_view(job, reader, ("outer0", "outer1"), lock)
    for tokens in [("outer1", "outer0"), ("outer0",), ("train0",), ("outer0", "outer0")]:
        with pytest.raises(ValueError):
            validate_outer_view(job, reader, tokens, lock)
    with pytest.raises(ValueError, match="encoder and head"):
        validate_outer_view(
            replace(job, excluded_groups=("different",)), reader, ("outer0", "outer1"), lock
        )
    lock["sequence_to_group"]["outer_seq"] = "train_group"
    with pytest.raises(ValueError, match="encoder and head"):
        validate_outer_view(job, reader, ("outer0", "outer1"), lock)


def test_outer_cannot_substitute_cache_or_drop_unlisted_rows(fixture):
    job, reader, lock = fixture
    reader.identity = "other-cache"
    with pytest.raises(ValueError, match="cache"):
        validate_outer_view(job, reader, ("outer0", "outer1"), lock)
    reader.identity = "cache"
    lock["outer_tokens_by_fold"]["0"] = ["outer0"]
    with pytest.raises(ValueError, match="entire original fold"):
        validate_outer_view(job, reader, ("outer0",), lock)


def test_missing_stage_barrier_prevents_any_count_observation_or_model(fixture, tmp_path):
    job, reader, _ = fixture

    def forbidden(*args, **kwargs):
        pytest.fail("count observation accessed before all endpoints froze")

    reader.observation = forbidden
    with pytest.raises(FileNotFoundError):
        infer_initial_endpoint(
            job,
            barrier_path=tmp_path / "missing.json",
            lock_path=tmp_path / "missing_lock.json",
            reader=reader,
            tokens=("outer0", "outer1"),
            normalization=None,
            output=tmp_path,
            owner=SimpleNamespace(check=lambda: None),
        )


def test_numeric_shard_publication_and_resume_reject_partial_or_foreign_payloads(tmp_path):
    tokens = tuple(f"row{i}" for i in range(33))
    identity = "a" * 64
    shards = []
    for number in range(2):
        first = number * 32
        path = tmp_path / f"inference_{number:05d}.npz"
        arrays = {"logits": np.zeros((len(tokens[first : first + 32]), 108), dtype=np.float32)}
        record = {
            "identity_sha256": identity,
            "first_row": first,
            "tokens": list(tokens[first : first + 32]),
            "arrays": _publish_arrays(path, arrays),
        }
        shards.append({**record, "artifact_sha256": object_digest(record)})
        with np.load(path, allow_pickle=False) as loaded:
            np.testing.assert_array_equal(loaded["logits"], arrays["logits"])
        with pytest.raises(ValueError, match="already exists"):
            _publish_arrays(path, arrays)
    verify_inference_shards(shards, tokens, identity, tmp_path)
    with pytest.raises(ValueError, match="omitted"):
        verify_inference_shards(shards[:-1], tokens, identity, tmp_path)
    with pytest.raises(ValueError, match="order, source"):
        verify_inference_shards(shards[::-1], tokens, identity, tmp_path)
    with pytest.raises(ValueError, match="order, source"):
        verify_inference_shards(shards, tokens, "b" * 64, tmp_path)
    with pytest.raises(ValueError, match="pickle or object"):
        _publish_arrays(tmp_path / "object.npz", {"bad": np.array([{}], dtype=object)})
    with (tmp_path / "inference_00001.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="physical identity"):
        verify_inference_shards(shards, tokens, identity, tmp_path)
