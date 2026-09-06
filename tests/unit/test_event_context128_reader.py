"""Production reader rejects substituted caches and transitive TRAIN contamination."""

import json

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.cache_components import component_record
from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.data.event_context128 import COUNT_SHAPE
from e_jepa_ttc.data.event_context128_reader import CountReader, TrainingView
from e_jepa_ttc.data.stage70_roles import TrainRoles


@pytest.fixture
def reader(tmp_path):
    counts = np.zeros((2, *COUNT_SHAPE), dtype=np.uint16)
    counts[0, :, :, 0] = 1
    counts[1] = 60000
    np.save(tmp_path / "counts_00000.npy", counts)
    np.savez_compressed(
        tmp_path / "observations_00000.npz",
        durations_s=np.full((2, 2, 8), 0.01),
        support_bits=np.full((2, 2048), 255, dtype=np.uint8),
        midpoint_twice_relative_us=np.tile(np.array([-7, -5, -3, -1]), (2, 1)),
    )
    (tmp_path / "diagnostics_00000.json").write_text("[]")
    pd.DataFrame(
        {
            "sample_token": ["a", "b"],
            "sequence_id": ["train", "held"],
            "track_id": ["1", "2"],
            "outer_fold": [0, 1],
        }
    ).to_csv(tmp_path / "metadata.csv", index=False)
    identity = {"targets_present": False, "shape": [2, *COUNT_SHAPE], "dtype": "uint16"}
    manifest = {
        "status": "COMPLETE_COUNTS_ONLY",
        "identity": identity,
        "identity_sha256": object_digest(identity),
        "metadata": component_record(tmp_path / "metadata.csv"),
        "shards": [
            {
                "first_row": 0,
                "rows": 2,
                "components": {
                    name: component_record(tmp_path / name)
                    for name in (
                        "counts_00000.npy",
                        "observations_00000.npz",
                        "diagnostics_00000.json",
                    )
                },
            }
        ],
    }
    manifest["artifact_sha256"] = object_digest(manifest)
    (tmp_path / "CACHE_MANIFEST.json").write_text(json.dumps(manifest))
    return CountReader(tmp_path)


def roles():
    return TrainRoles(
        (("train", "g0"), ("alias", "g0"), ("held", "g1")),
        ("g0", "g1"),
        ("g0", "g1"),
        (),
        (),
        (),
        "fixture",
    )


def test_normalization_reads_only_bound_train_and_closes_maps(reader, monkeypatch):
    calls = []
    original = reader.observation

    def recorded(token):
        calls.append(token)
        return original(token)

    monkeypatch.setattr(reader, "observation", recorded)
    view = TrainingView(
        reader, ("a",), roles(), excluded_sequences=("held",), include_expansion=False
    )
    normalization = view.fit_normalization()
    assert calls == ["a"]
    np.testing.assert_allclose(normalization.mean, [np.log1p(100), 0], rtol=1e-6)
    inputs = view.inputs("a", normalization)
    assert inputs.contexts.shape == (4, 8, 128, 128)
    assert not inputs.contexts.any()
    # Windows cannot rename an open memory mapped payload: this also tests closure.
    path = reader.root / "counts_00000.npy"
    renamed = path.with_suffix(".closed")
    path.rename(renamed)
    renamed.rename(path)
    with pytest.raises(ValueError, match="another TRAIN view"):
        view.inputs("b", normalization)


@pytest.mark.parametrize("tokens,excluded", [(("b",), ("held",)), (("a",), ("alias",))])
def test_outer_and_acquisition_alias_contamination_rejected(reader, tokens, excluded):
    with pytest.raises(ValueError, match="held-out acquisition"):
        TrainingView(reader, tokens, roles(), excluded_sequences=excluded, include_expansion=False)


def test_reader_rejects_post_verification_component_change(reader):
    path = reader.root / "counts_00000.npy"
    with path.open("ab") as stream:
        stream.write(b"substitution")
    with pytest.raises(ValueError, match="changed during use"):
        reader.observation("a")


def test_reader_rejects_traversal_even_with_resigned_manifest(reader):
    path = reader.root / "CACHE_MANIFEST.json"
    manifest = json.loads(path.read_text())
    records = manifest["shards"][0]["components"]
    records["../counts_00000.npy"] = records.pop("counts_00000.npy")
    manifest.pop("artifact_sha256")
    manifest["artifact_sha256"] = object_digest(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inventory"):
        CountReader(reader.root)
