"""Actual normalization and supervision consume only the common optimizer acquisition pool."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.data.event_context128_reader import CountObservation
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.training.capacity_source_views import compile_capacity_outer_view
from e_jepa_ttc.training.resolved_architecture_sources_v11 import load_architecture_sources


def fixture():
    observed = []
    metadata = pd.DataFrame(
        {
            "sample_token": ["fit", "diagnostic", "outer"],
            "sequence_id": ["s0", "s1", "s2"],
            "track_id": ["t"] * 3,
            "outer_fold": [1, 2, 0],
        }
    )

    def observation(token):
        assert token == "fit", "excluded acquisition was physically read"
        observed.append(token)
        return CountObservation(
            np.ones((2, 8, 2, 128, 128), dtype=np.uint16),
            np.full((2, 8), 0.0125),
            np.ones((128, 128), dtype=bool),
            np.array([-350000, -250000, -150000, -50000]),
        )

    reader = SimpleNamespace(metadata=metadata, identity="cache", observation=observation)
    roles = TrainRoles(
        tuple((f"s{i}", f"g{i}") for i in range(3)), ("g0", "g1", "g2"), (), (), (), (), "test"
    )
    recipe = {
        "kind": "CAPACITY_NESTED_PRODUCER_ROLES",
        "folds": [
            {
                "outer_fold": 0,
                "fit_sequences": ["s0"],
                "diagnostic_sequences": ["s1"],
                "outer_held_sequences": ["s2"],
                "all_fit_ancestor_exclusions": ["s1", "s2"],
                "normalization_and_pretext_exclusions": ["s1", "s2"],
            }
        ],
    }
    recipe["artifact_sha256"] = object_digest(recipe)
    labels = pd.DataFrame(
        {"sample_token": ["fit"], "sequence_id": ["s0"], "target_ttc": [1.0], "bucket": ["crucial"]}
    )
    return reader, roles, recipe, labels, observed


def test_real_normalization_and_missing_bucket_prior_exclude_diagnostic_and_outer(tmp_path):
    reader, roles, recipe, labels, observed = fixture()
    kwargs = dict(
        outer_fold=0,
        source_role_sha256="a" * 64,
        source_bindings={},
        output=tmp_path / "view",
        check=lambda: None,
    )
    source = compile_capacity_outer_view(reader, labels, roles, recipe, **kwargs)
    assert observed == ["fit"]
    loaded = load_architecture_sources(
        source,
        reader,
        roles,
        expected_fold=0,
        expected_expansion=False,
        source_role_sha256="a" * 64,
        producer_recipe=recipe,
    )
    assert loaded.supervision.mass.tolist() == [1.0]
    assert loaded.view.tokens == ("fit",)
    assert compile_capacity_outer_view(reader, labels, roles, recipe, **kwargs) == source
    assert observed == ["fit"]


def test_diagnostic_in_fit_manifest_is_rejected_before_count_read(tmp_path):
    reader, roles, recipe, labels, observed = fixture()
    recipe["folds"][0]["fit_sequences"].append("s1")
    recipe.pop("artifact_sha256")
    recipe["artifact_sha256"] = object_digest(recipe)
    with pytest.raises(ValueError, match="leaked"):
        compile_capacity_outer_view(
            reader,
            labels,
            roles,
            recipe,
            outer_fold=0,
            source_role_sha256="a" * 64,
            source_bindings={},
            output=tmp_path / "view",
            check=lambda: None,
        )
    assert not observed
