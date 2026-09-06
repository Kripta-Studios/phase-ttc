"""Fold-bound source rehydration rejects silently changed supervision and estimands."""

from types import SimpleNamespace

import pandas as pd
import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.data.event_context128_reader import TrainingView
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.training.resolved_architecture_sources_v11 import load_architecture_sources


def fixture(root, *, damage=None):
    reader = SimpleNamespace(
        identity="cache",
        metadata=pd.DataFrame(
            {
                "sample_token": ["r0", "r1", "outer0"],
                "sequence_id": ["s0", "s0", "s1"],
                "track_id": ["t0", "t1", "outer_track"],
                "outer_fold": [1, 1, 0],
            }
        ),
    )
    roles = TrainRoles((("s0", "g0"), ("s1", "g1")), ("g0", "g1"), (), (), (), (), "fixture")
    view = TrainingView(
        reader, ("r0", "r1"), roles, excluded_sequences=("s1",), include_expansion=False
    )
    norm = {
        "mean": [0.0, 1.0],
        "std": [1.0, 2.0],
        "training_view_sha256": view.identity,
        "valid_values_per_polarity": 100,
    }
    supervision = {
        "training_view_sha256": view.identity,
        "tokens": ["r0", "r1"],
        "estimand": "strict_original_TRAIN",
        "target_phase": [0.02, -0.01],
        "mass": [0.5, 0.5],
        "prior_bin_mass": [1.0] + [0.0] * 107,
        "outside_support_mass": 0.0,
        "identity_sha256": "d" * 64,
    }
    if damage == "mass":
        supervision["mass"] = [0.5, 0.6]
    elif damage == "order":
        supervision["tokens"] = ["r1", "r0"]
    elif damage == "estimand":
        supervision["estimand"] = "present_bucket_normalized_TRAIN"
    elif damage == "normalization":
        norm["training_view_sha256"] = "b" * 64
    atomic_json(root / "NORMALIZATION.json", norm)
    atomic_json(root / "SUPERVISION.json", supervision)
    source = {
        "kind": "RESOLVED_ARCHITECTURE_TRAIN_SOURCE_VIEW",
        "outer_fold": 0,
        "include_expansion": False,
        "source_role_sha256": "a" * 64,
        "ordered_tokens": ["r0", "r1"],
        "excluded_sequences": ["s1"],
        "train_sequences": ["s0"],
        "allowed_train_groups": ["g0"],
        "excluded_groups": ["g1"],
        "training_view_sha256": view.identity,
        "normalization": binding(root / "NORMALIZATION.json"),
        "supervision": binding(root / "SUPERVISION.json"),
    }
    source["artifact_sha256"] = object_digest(source)
    atomic_json(root / "SOURCE.json", source)
    return binding(root / "SOURCE.json"), reader, roles


def test_frozen_train_tensors_are_rehydrated_without_reading_count_observations(tmp_path):
    source, reader, roles = fixture(tmp_path)
    result = load_architecture_sources(
        source,
        reader,
        roles,
        expected_fold=0,
        expected_expansion=False,
        source_role_sha256="a" * 64,
    )
    assert result.view.tokens == result.supervision.tokens == ("r0", "r1")
    assert result.supervision.mass.sum() == 1
    assert result.supervision.prior_bin_mass.shape == (108,)
    assert result.normalization.training_view_sha256 == result.view.identity
    with pytest.raises(ValueError, match="treatment fold"):
        load_architecture_sources(
            source,
            reader,
            roles,
            expected_fold=1,
            expected_expansion=False,
            source_role_sha256="a" * 64,
        )


@pytest.mark.parametrize("damage", ["mass", "order", "estimand", "normalization"])
def test_resigned_manifest_does_not_bypass_numeric_and_fold_contract(tmp_path, damage):
    source, reader, roles = fixture(tmp_path, damage=damage)
    with pytest.raises(ValueError):
        load_architecture_sources(
            source,
            reader,
            roles,
            expected_fold=0,
            expected_expansion=False,
            source_role_sha256="a" * 64,
        )
