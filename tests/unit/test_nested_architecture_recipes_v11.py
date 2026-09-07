"""Nested metadata recipes keep old split identity and exclude every held-out ancestor."""

import pandas as pd
import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.data.canonical_token_identity import hash_sorted_token_strings
from e_jepa_ttc.data.nested_architecture_recipes_v11 import (
    CountMetadataCatalog,
    nested_source_recipe,
)
from e_jepa_ttc.data.stage70_roles import TrainRoles


def fixture():
    original = pd.DataFrame(
        {
            "sample_token": [f"r{i}" for i in range(9)],
            "sequence_id": [f"s{i}" for i in range(9)],
            "track_id": [f"t{i}" for i in range(9)],
            "outer_fold": [i // 3 for i in range(9)],
        }
    )
    extra = pd.DataFrame(
        {
            "sample_token": ["d0", "d1"],
            "sequence_id": ["s9", "s10"],
            "track_id": ["t9", "t10"],
            "outer_fold": [-1, -1],
        }
    )
    roles = TrainRoles(
        tuple((f"s{i}", f"g{i}") for i in range(13)),
        tuple(f"g{i}" for i in range(9)),
        (),
        ("g12",),
        ("g11",),
        ("g9", "g10"),
        "fixture",
    )
    catalog = CountMetadataCatalog(pd.concat([original, extra], ignore_index=True), "a" * 64)
    parent = {
        "outer_fold": 0,
        "include_expansion": True,
        "ordered_tokens": ["d1", *[f"r{i}" for i in range(3, 9)], "d0"],
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
    protocol = {"outer_fold": 0, "inner_fold": 0, "folds": [split]}
    return catalog, original, parent, protocol, roles


def test_nested_recipe_preserves_fixed_expansion_and_refuses_event_reads():
    catalog, original, parent, protocol, roles = fixture()
    recipe = nested_source_recipe(
        catalog, original, parent, protocol, roles, outer=0, inner=0, expanded=True
    )
    assert recipe["ordered_tokens"] == ["d1", "r5", "r6", "r7", "r8", "d0"]
    assert recipe["excluded_groups"] == ["g0", "g1", "g2", "g3", "g4"]
    assert recipe["inner_prediction_tokens"] == ["r3", "r4"]
    assert not recipe["normalization_fitted"] and not recipe["prior_fitted"]
    assert recipe["materialization_requires_stage75_headroom_authorization"]
    with pytest.raises(PermissionError, match="cannot read"):
        catalog.observation("r5")


@pytest.mark.parametrize("damage", ["outer", "tokens", "overlap", "signature"])
def test_nested_recipe_rejects_changed_fold_membership_and_parent_identity(damage):
    catalog, original, parent, protocol, roles = fixture()
    if damage == "outer":
        protocol["outer_fold"] = 1
    elif damage == "tokens":
        protocol["folds"][0]["train_sample_tokens_sha256"] = "b" * 64
    elif damage == "overlap":
        protocol["folds"][0]["train_sequence_ids"][0] = "s3"
    else:
        parent["ordered_tokens"].append("r0")
    with pytest.raises(ValueError):
        nested_source_recipe(
            catalog, original, parent, protocol, roles, outer=0, inner=0, expanded=True
        )


def test_catalog_does_not_smuggle_targets_into_the_recipe_interface():
    catalog, *_ = fixture()
    with pytest.raises(ValueError, match="target-free"):
        CountMetadataCatalog(catalog.metadata.assign(target_ttc_s=2), catalog.identity)


@pytest.mark.parametrize("damage", [None, "diagnostic", "inner_exclusion", "parent"])
def test_capacity_nested_recipe_excludes_diagnostic_and_all_held_ancestors(damage):
    catalog, original, parent, protocol, roles = fixture()
    split = protocol["folds"][0]
    split.update(
        train_sequence_ids=["s6", "s7", "s8"],
        dev_sequence_ids=["s4", "s5"],
        train_rows=3,
        dev_rows=2,
        train_sample_tokens_sha256=hash_sorted_token_strings(["r6", "r7", "r8"]),
        dev_sample_tokens_sha256=hash_sorted_token_strings(["r4", "r5"]),
    )
    producer = {
        "kind": "CAPACITY_NESTED_PRODUCER_ROLES",
        "folds": [
            {
                "outer_fold": 0,
                "diagnostic_sequences": ["s3"],
                "outer_held_sequences": ["s0", "s1", "s2"],
                "fit_sequences": ["s4", "s5", "s6", "s7", "s8"],
                "all_fit_ancestor_exclusions": ["s0", "s1", "s2", "s3"],
                "normalization_and_pretext_exclusions": ["s0", "s1", "s2", "s3"],
                "inner_roles": [
                    {
                        "inner_fold": 0,
                        "fit_sequences": ["s6", "s7", "s8"],
                        "prediction_sequences": ["s4", "s5"],
                        "all_fit_ancestor_exclusions": [f"s{i}" for i in range(6)],
                    }
                ],
            }
        ],
    }
    if damage == "diagnostic":
        producer["folds"][0]["diagnostic_sequences"] = ["s6"]
    elif damage == "inner_exclusion":
        producer["folds"][0]["inner_roles"][0]["all_fit_ancestor_exclusions"].remove("s3")
    producer["artifact_sha256"] = object_digest(producer)
    parent.pop("artifact_sha256")
    parent["ordered_tokens"].remove("r3")
    if damage == "parent":
        parent["ordered_tokens"].append("r3")
    parent["producer_roles_sha256"] = producer["artifact_sha256"]
    parent["artifact_sha256"] = object_digest(parent)
    if damage:
        with pytest.raises(ValueError, match="capacity"):
            nested_source_recipe(
                catalog,
                original,
                parent,
                protocol,
                roles,
                outer=0,
                inner=0,
                expanded=True,
                producer_recipe=producer,
            )
        return
    recipe = nested_source_recipe(
        catalog,
        original,
        parent,
        protocol,
        roles,
        outer=0,
        inner=0,
        expanded=True,
        producer_recipe=producer,
    )
    assert recipe["ordered_tokens"] == ["d1", "r6", "r7", "r8", "d0"]
    assert recipe["excluded_groups"] == [f"g{i}" for i in range(6)]
    assert recipe["inner_prediction_tokens"] == ["r4", "r5"]
    assert recipe["capacity_producer_roles"] == producer
    assert recipe["optimizer_updates"] == 0
