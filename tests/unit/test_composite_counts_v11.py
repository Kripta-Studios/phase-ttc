"""Exercise actual small count files, source dispatch and transitive TRAIN normalization."""

import json

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.cache_components import component_record
from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.data.composite_counts_v11 import CompositeCountReader
from e_jepa_ttc.data.event_context128 import COUNT_SHAPE
from e_jepa_ttc.data.event_context128_reader import CountReader, TrainingView
from e_jepa_ttc.data.stage70_roles import TrainRoles


def cache(root, tokens, sequences, count):
    root.mkdir()
    np.save(root / "counts_00000.npy", np.full((2, *COUNT_SHAPE), count, dtype=np.uint16))
    np.savez_compressed(
        root / "observations_00000.npz",
        durations_s=np.full((2, 2, 8), 0.01),
        support_bits=np.full((2, 2048), 255, dtype=np.uint8),
        midpoint_twice_relative_us=np.tile(np.array([-7, -5, -3, -1]), (2, 1)),
    )
    (root / "diagnostics_00000.json").write_text("[]")
    pd.DataFrame(
        dict(sample_token=tokens, sequence_id=sequences, track_id=[1, 2], outer_fold=[0, 1])
    ).to_csv(root / "metadata.csv", index=False)
    identity = dict(
        targets_present=False,
        shape=[2, *COUNT_SHAPE],
        dtype="uint16",
        sensor_size=[1280, 720],
        channel_order="bin-major then negative,positive",
        bin_edges="integer_floor_8_half_open",
        normalization="not fitted in count cache",
        sensor_support="fixture physical coverage",
        producer_code=[dict(path="fixture_producer.py", sha256="a" * 64)],
    )
    manifest = dict(
        status="COMPLETE_COUNTS_ONLY",
        identity=identity,
        identity_sha256=object_digest(identity),
        metadata=component_record(root / "metadata.csv"),
        shards=[
            dict(
                first_row=0,
                rows=2,
                components={
                    name: component_record(root / name)
                    for name in (
                        "counts_00000.npy",
                        "observations_00000.npz",
                        "diagnostics_00000.json",
                    )
                },
            )
        ],
    )
    manifest["artifact_sha256"] = object_digest(manifest)
    (root / "CACHE_MANIFEST.json").write_text(json.dumps(manifest))
    return CountReader(root)


@pytest.fixture
def sources(tmp_path):
    old = cache(tmp_path / "old", ["a", "b"], ["train", "held"], 1)
    extra = cache(tmp_path / "extra", ["c", "d"], ["extra", "extra"], 3)
    roles = TrainRoles(
        (("train", "train"), ("held", "held"), ("extra", "extra"), ("seal", "seal")),
        ("train", "held"),
        ("train", "held"),
        (),
        ("seal",),
        ("extra",),
        "fixture",
    )
    return old, extra, roles


def test_union_reads_bound_rows_and_fits_one_train_normalizer(sources, monkeypatch):
    old, extra, roles = sources
    combined = CompositeCountReader((old, extra), roles)
    calls = []
    for source in (old, extra):
        original = source.observation

        def recorded(token, original=original):
            calls.append(token)
            return original(token)

        monkeypatch.setattr(source, "observation", recorded)
    view = TrainingView(
        combined, ("a", "c"), roles, excluded_sequences=("held",), include_expansion=True
    )
    norm = view.fit_normalization()
    assert calls == ["a", "c"]
    expected = (np.log1p(100) + np.log1p(300)) / 2
    np.testing.assert_allclose(norm.mean, [expected, expected], rtol=1e-6)
    np.testing.assert_allclose(view.inputs("a", norm).contexts, -1, atol=1e-6)
    np.testing.assert_allclose(view.inputs("c", norm).contexts, 1, atol=1e-6)
    with pytest.raises(ValueError, match="absent"):
        combined.observation("seal")
    for tokens, exclude, expand in (
        (("a", "b"), ("held",), True),
        (("a", "c"), ("extra",), True),
        (("a", "c"), (), False),
    ):
        with pytest.raises(ValueError):
            TrainingView(
                combined, tokens, roles, excluded_sequences=exclude, include_expansion=expand
            )


def test_duplicate_sources_and_tokens_rejected(sources, tmp_path):
    old, extra, roles = sources
    with pytest.raises(ValueError, match="distinct"):
        CompositeCountReader((old, old), roles)
    duplicate = cache(tmp_path / "duplicate", ["a", "z"], ["extra", "extra"], 5)
    with pytest.raises(ValueError, match="duplicate token"):
        CompositeCountReader((old, duplicate), roles)


def test_confirmation_source_rejected(sources, tmp_path):
    old, extra, roles = sources
    sealed = cache(tmp_path / "sealed", ["sealed1", "sealed2"], ["seal", "seal"], 4)
    with pytest.raises(ValueError, match="TRAIN role"):
        CompositeCountReader((old, sealed), roles)


def test_resigned_representation_drift_rejected(sources):
    old, extra, roles = sources
    path = extra.root / "CACHE_MANIFEST.json"
    manifest = json.loads(path.read_text())
    manifest["identity"]["bin_edges"] = "different_timing"
    manifest["identity_sha256"] = object_digest(manifest["identity"])
    del manifest["artifact_sha256"]
    manifest["artifact_sha256"] = object_digest(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest changed"):
        CompositeCountReader((old, extra), roles)
    with pytest.raises(ValueError, match="immutable representation"):
        CompositeCountReader((old, CountReader(extra.root)), roles)
