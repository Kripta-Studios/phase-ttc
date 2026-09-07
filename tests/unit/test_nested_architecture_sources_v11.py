"""Real statistics and supervision integration on tiny synthetic nested TRAIN observations."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.data.canonical_token_identity import hash_sorted_token_strings
from e_jepa_ttc.data.event_context128_reader import CountObservation
from e_jepa_ttc.data.nested_architecture_recipes_v11 import (
    CountMetadataCatalog,
    nested_source_recipe,
)
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.training import nested_architecture_sources_v11 as module
from e_jepa_ttc.training.stage73_factorial import PlannedFit


@pytest.fixture(params=["historical", "capacity", "capacity_subset"])
def fixture(tmp_path, monkeypatch, request):
    metadata = pd.DataFrame(
        {
            "sample_token": [f"r{i}" for i in range(10)],
            "sequence_id": [f"s{i}" for i in range(10)],
            "track_id": [f"t{i}" for i in range(10)],
            "outer_fold": [i // 3 if i < 9 else -1 for i in range(10)],
        }
    )
    original = metadata.iloc[:9].copy()
    roles = TrainRoles(
        tuple((f"s{i}", f"g{i}") for i in range(10)),
        tuple(f"g{i}" for i in range(9)),
        (),
        (),
        (),
        ("g9",),
        "fixture",
    )
    catalog = CountMetadataCatalog(metadata, "a" * 64)
    parent = {
        "outer_fold": 0,
        "include_expansion": True,
        "ordered_tokens": [f"r{i}" for i in range(3, 10)],
    }
    parent["artifact_sha256"] = object_digest(parent)
    split = {
        "train_sequence_ids": [f"s{i}" for i in range(5, 9)],
        "dev_sequence_ids": ["s3", "s4"],
        "train_rows": 4,
        "dev_rows": 2,
        "train_sample_tokens_sha256": hash_sorted_token_strings([f"r{i}" for i in range(5, 9)]),
        "dev_sample_tokens_sha256": hash_sorted_token_strings(["r3", "r4"]),
    }
    producer = None
    if request.param != "historical":
        producer = {
            "kind": "CAPACITY_NESTED_PRODUCER_ROLES",
            "folds": [
                {
                    "outer_fold": 0,
                    "diagnostic_sequences": ["s5"],
                    "outer_held_sequences": ["s0", "s1", "s2"],
                    "fit_sequences": ["s3", "s4", "s6", "s7", "s8"],
                    "all_fit_ancestor_exclusions": ["s0", "s1", "s2", "s5"],
                    "normalization_and_pretext_exclusions": ["s0", "s1", "s2", "s5"],
                    "inner_roles": [
                        {
                            "inner_fold": 0,
                            "fit_sequences": ["s6", "s7", "s8"],
                            "prediction_sequences": ["s3", "s4"],
                            "all_fit_ancestor_exclusions": [f"s{i}" for i in range(6)],
                        }
                    ],
                }
            ],
        }
        producer["artifact_sha256"] = object_digest(producer)
        parent.pop("artifact_sha256")
        parent["ordered_tokens"].remove("r5")
        if request.param == "capacity_subset":
            parent["ordered_tokens"].remove("r9")
            parent["include_expansion"] = False
        parent["producer_roles_sha256"] = producer["artifact_sha256"]
        parent["artifact_sha256"] = object_digest(parent)
        split.update(
            train_sequence_ids=["s6", "s7", "s8"],
            train_rows=3,
            train_sample_tokens_sha256=hash_sorted_token_strings(["r6", "r7", "r8"]),
        )
    recipe = nested_source_recipe(
        catalog,
        original,
        parent,
        {"outer_fold": 0, "inner_fold": 0, "folds": [split]},
        roles,
        outer=0,
        inner=0,
        expanded=request.param != "capacity_subset",
        producer_recipe=producer,
    )
    recipe["source_role_sha256"] = "e" * 64
    recipe["artifact_sha256"] = object_digest(recipe)
    path = tmp_path / "recipe.json"
    atomic_json(path, recipe)
    labels = original[["sample_token", "sequence_id"]].assign(target_ttc=2.0, bucket="crucial")
    labels.to_csv(tmp_path / "old.csv", index=False)
    pd.DataFrame({"sample_token": ["r9"], "sequence_id": ["s9"], "ttc": [4.0]}).to_parquet(
        tmp_path / "extra.parquet"
    )
    plan = PlannedFit(
        "S75", "S73-PPS-A" if request.param == "capacity_subset" else "S73-PPS-AD", 0, 7, 0
    )
    lock = {
        "artifact_sha256": "b" * 64,
        "source_role_sha256": "e" * 64,
        "nested_recipe_by_job": {plan.job_id: binding(path)},
        "nested_source_root": str(tmp_path / "sources"),
        "campaign_root": str(tmp_path),
        "nested_label_sources": {
            "original": binding(tmp_path / "old.csv"),
            "expanded": binding(tmp_path / "extra.parquet"),
        },
    }
    reads = []

    def observation(token):
        assert token in recipe["ordered_tokens"], "an excluded inner/outer observation was read"
        reads.append(token)
        return CountObservation(
            np.full((2, 8, 2, 128, 128), int(token[1:]), np.uint16),
            np.full((2, 8), 0.01),
            np.ones((128, 128), bool),
            np.zeros((2, 8), np.int64),
        )

    reader = SimpleNamespace(identity=catalog.identity, metadata=metadata, observation=observation)
    monkeypatch.setattr(
        module, "verify_stage_decision", lambda *args: ({"artifact_sha256": "f" * 64}, (plan,))
    )
    kwargs = {
        "lock": lock,
        "decision_source": {"path": "synthetic gate"},
        "reader": reader,
        "roles": roles,
        "owner": SimpleNamespace(check=lambda: None),
    }
    return plan, kwargs, reads, recipe


def test_fresh_inner_statistics_and_prior_use_only_prescribed_rows_and_resume_without_refitting(
    fixture,
):
    plan, kwargs, reads, recipe = fixture
    loaded = module.materialize_nested_sources(plan, **kwargs)
    assert reads == recipe["ordered_tokens"]
    assert loaded.view.tokens == tuple(recipe["ordered_tokens"])
    assert loaded.normalization.training_view_sha256 == loaded.view.identity
    assert loaded.supervision.mass.sum().item() == pytest.approx(1.0)
    assert (
        loaded.source["normalization_fit_rows"]
        == loaded.source["prior_fit_rows"]
        == len(recipe["ordered_tokens"])
    )
    again = module.materialize_nested_sources(plan, **kwargs)
    assert again.source_sha256 == loaded.source_sha256
    assert reads == recipe["ordered_tokens"]


def test_missing_headroom_gate_prevents_any_normalization_count_read(fixture, monkeypatch):
    plan, kwargs, reads, _ = fixture
    monkeypatch.setattr(module, "verify_stage_decision", lambda *args: ({}, ()))
    with pytest.raises(ValueError, match="not triggered"):
        module.materialize_nested_sources(plan, **kwargs)
    assert reads == []


def test_changed_nested_statistics_are_not_silently_refitted(fixture):
    plan, kwargs, reads, recipe = fixture
    loaded = module.materialize_nested_sources(plan, **kwargs)
    from pathlib import Path

    Path(loaded.source["normalization"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="physical identity changed"):
        module.materialize_nested_sources(plan, **kwargs)
    assert reads == recipe["ordered_tokens"]
