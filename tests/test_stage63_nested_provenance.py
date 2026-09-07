from __future__ import annotations

import copy

import pandas as pd
import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.data.canonical_token_identity import hash_sorted_token_strings
from e_jepa_ttc.data.nested_provenance import (
    validate_exact_oof_union,
    validate_frozen_teacher_dependency,
    validate_nested_split,
    validate_uninitialized_producer_contract,
)
from e_jepa_ttc.training.capacity_campaign import producer_roles


@pytest.mark.parametrize("completed", [False, True])
def test_capacity_executor_stops_before_prediction_reads_until_barrier(
    tmp_path, monkeypatch, completed
):
    from dataclasses import asdict
    from types import SimpleNamespace

    import torch
    import yaml

    from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding
    from e_jepa_ttc.training.stage70_update_ledger import FitSpec
    from scripts.train_scientific_recovery_v8_router_expert import execute_capacity_comparator

    cache = tmp_path / "cache.json"
    protocol = tmp_path / "protocol.json"
    config = tmp_path / "config.yaml"
    atomic_json(cache, {"fixture": True})
    atomic_json(protocol, {"outer_fold": 0})
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "cache_manifest": str(cache),
                    "cache_manifest_sha256": binding(cache)["sha256"],
                    "development_protocol": {
                        "path": str(protocol),
                        "file_sha256": binding(protocol)["sha256"],
                    },
                },
                "training": {},
            }
        ),
        encoding="utf-8",
    )
    spec = FitSpec(
        "A5", "COMPARATOR", "A5", 0, 7, 18, binding(config)["sha256"], binding(protocol)["sha256"]
    )
    job = {
        "spec": asdict(spec),
        "effective_config": binding(config),
        "protocol": binding(protocol),
        "cache_manifest": binding(cache),
        "output_root": str(tmp_path / "fit"),
    }
    lock = {
        "comparator_jobs": {"A5": job},
        "campaign_root": str(tmp_path),
        "stage_barrier_root": str(tmp_path / "barriers"),
    }
    monkeypatch.setattr("e_jepa_ttc.training.stage72_jobs.require_scientific_lock", lambda *_: lock)
    owner = SimpleNamespace(root=tmp_path, check=lambda: None)
    ledger = SimpleNamespace(
        owner_root=tmp_path, state={"prior_campaign": {"fixture": True}, "jobs": {}}
    )
    if completed:
        ledger.state["jobs"]["A5"] = {"status": "COMPLETE"}
    fits = []

    def fit(*args, **kwargs):
        fits.append(kwargs["accounting"])
        ledger.state["jobs"]["A5"] = {"status": "COMPLETE"}

    def barrier(*args, **kwargs):
        raise ValueError("fixture architecture endpoints not frozen")

    def forbidden(*args, **kwargs):
        pytest.fail("prediction dataset opened before architecture barrier")

    monkeypatch.setattr("scripts.train_causal_scale_eap_screen.run", fit)
    monkeypatch.setattr(
        "e_jepa_ttc.artifacts.stage_endpoint_barrier_v11.verify_initial_barrier", barrier
    )
    monkeypatch.setattr(
        "scripts.train_scientific_recovery_v8_router_expert.GarlTTCObjectEventV4Dataset",
        forbidden,
    )
    with pytest.raises(ValueError, match="architecture endpoints not frozen"):
        execute_capacity_comparator(
            "A5",
            lock_path=tmp_path / "lock",
            lock_sha256="a" * 64,
            ledger=ledger,
            owner=owner,
            device=torch.device("cpu"),
        )
    assert len(fits) == (0 if completed else 1)
    if fits:
        assert fits[0] == (ledger, spec)


@pytest.mark.parametrize("inner", [None, 0, 1, 2])
@pytest.mark.parametrize("diagnostic_contract", [False, True])
def test_capacity_comparator_split_binds_roles_and_rejects_acquisition_leaks(
    inner, diagnostic_contract
):
    canonical = pd.DataFrame(
        {
            "sample_token": [f"t{i}" for i in range(54)],
            "sequence_id": [f"s{i // 2}" for i in range(54)],
            "track_id": "track",
            "outer_fold": [i // 2 % 3 for i in range(54)],
            "acquisition_group": [f"a{i // 2}" for i in range(54)],
        }
    )
    cohort = {
        "recipe": {"prefix": "fixed"},
        "sequence_to_group": dict(
            zip(canonical.sequence_id, canonical.acquisition_group, strict=True)
        ),
        "folds": [
            {
                "outer_fold": f,
                "train_sequences": sorted(
                    set(canonical.loc[canonical.outer_fold != f, "sequence_id"])
                ),
                "held_sequences": sorted(
                    set(canonical.loc[canonical.outer_fold == f, "sequence_id"])
                ),
            }
            for f in range(3)
        ],
    }
    cohort["artifact_sha256"] = object_digest(cohort)
    recipe = producer_roles(cohort)
    role = recipe["folds"][0]
    train = role["fit_sequences"] if inner is None else role["inner_roles"][inner]["fit_sequences"]
    dev = (
        role["outer_held_sequences"]
        if inner is None
        else role["inner_roles"][inner]["prediction_sequences"]
    )
    split = {}
    for name, sequences in (("train", train), ("dev", dev)):
        selected = canonical.loc[canonical.sequence_id.isin(sequences)]
        split.update(
            {
                f"{name}_sequence_ids": sequences,
                f"{name}_rows": len(selected),
                f"{name}_sample_tokens_sha256": hash_sorted_token_strings(selected.sample_token),
            }
        )
    protocol = {
        "outer_fold": 0,
        "inner_fold": inner,
        "folds": [split],
        "producer_roles_sha256": recipe["artifact_sha256"],
    }
    if diagnostic_contract:
        protocol.update(
            artifact_type="scientific_recovery_v11_capacity_comparator_diagnostic_v1",
            dev_role="TRAIN_diagnostic_only",
            checkpoint_selection_mode="last_epoch",
            fixed_epochs=18,
            prediction_sequence_ids=split["dev_sequence_ids"],
            prediction_rows=split["dev_rows"],
            prediction_sample_tokens_sha256=split["dev_sample_tokens_sha256"],
        )
        diagnostic_rows = canonical.loc[canonical.sequence_id.isin(role["diagnostic_sequences"])]
        split.update(
            dev_sequence_ids=role["diagnostic_sequences"],
            dev_rows=len(diagnostic_rows),
            dev_sample_tokens_sha256=hash_sorted_token_strings(diagnostic_rows.sample_token),
        )
    artifact = {
        "outer_fold": 0,
        "inner_fold": inner,
        "role": "outer_dev" if inner is None else "inner_oof",
        "producer_roles_sha256": recipe["artifact_sha256"],
    }
    output = canonical.loc[canonical.sequence_id.isin(dev)].assign(outer_fold=0, inner_fold=inner)
    result = validate_nested_split(
        protocol,
        artifact,
        canonical,
        output,
        outer_fold=0,
        inner_fold=inner,
        producer_recipe=recipe,
    )
    assert result["excluded_diagnostic_sequence_ids"] == role["diagnostic_sequences"]
    assert result["ancestry_validated"] is False
    with pytest.raises(ValueError, match="bound protocol"):
        validate_nested_split(
            protocol,
            {**artifact, "producer_roles_sha256": "0" * 64},
            canonical,
            output,
            outer_fold=0,
            inner_fold=inner,
            producer_recipe=recipe,
        )
    contaminated = canonical.copy()
    contaminated.loc[contaminated.sequence_id == train[0], "acquisition_group"] = canonical.loc[
        canonical.sequence_id == dev[0], "acquisition_group"
    ].iloc[0]
    with pytest.raises(ValueError, match="acquisition crosses"):
        validate_nested_split(
            protocol,
            artifact,
            contaminated,
            output,
            outer_fold=0,
            inner_fold=inner,
            producer_recipe=recipe,
        )
    wrong = copy.deepcopy(protocol)
    wrong["folds"][0]["train_sequence_ids"] = [*train, *role["diagnostic_sequences"]]
    with pytest.raises(ValueError, match="exact registered role"):
        validate_nested_split(
            wrong,
            artifact,
            canonical,
            output,
            outer_fold=0,
            inner_fold=inner,
            producer_recipe=recipe,
        )


@pytest.mark.parametrize("count", [8192, 30000])
def test_teacher_dependency_binds_exact_variable_token_universe(count):
    tokens = [f"token-{i}" for i in range(count)]
    initialization = {"representation_teacher_artifact_sha256": "a" * 64}
    audit = {
        "status": "passed",
        "preexisting_frozen_teacher": True,
        "fold_fitted_teacher": False,
        "teacher_artifact_sha256": "a" * 64,
        "verified_tokens": count,
        "materializer_ast_no_fit_calls": True,
        "verified_tokens_sha256": hash_sorted_token_strings(tokens),
    }
    validate_frozen_teacher_dependency(initialization, audit, required_tokens=tokens)
    with pytest.raises(ValueError, match="exact required token"):
        validate_frozen_teacher_dependency(
            initialization, audit, required_tokens=["other", *tokens[1:]]
        )
    with pytest.raises(ValueError, match="unique"):
        validate_frozen_teacher_dependency(
            initialization, audit, required_tokens=[*tokens, tokens[0]]
        )
    if count == 8192:
        validate_frozen_teacher_dependency(initialization, audit)
    else:
        with pytest.raises(ValueError, match="verified frozen"):
            validate_frozen_teacher_dependency(initialization, audit)
    audit["fold_fitted_teacher"] = True
    with pytest.raises(ValueError, match="verified frozen"):
        validate_frozen_teacher_dependency(initialization, audit, required_tokens=tokens)


def test_aggregate_must_equal_verified_outputs_not_just_share_tokens() -> None:
    first = pd.DataFrame({"token_id": ["a"], "prediction": [1.0], "outer_fold": [0]})
    second = pd.DataFrame({"token_id": ["b"], "prediction": [2.0], "outer_fold": [0]})
    aggregate = pd.concat([second, first], ignore_index=True)
    validate_exact_oof_union(aggregate, [first, second])
    aggregate.loc[0, "prediction"] = 3.0
    with pytest.raises(ValueError, match="verified producer union"):
        validate_exact_oof_union(aggregate, [first, second])


def test_fitted_ancestor_or_different_effective_train_set_is_rejected() -> None:
    checkpoint = {
        "training_config": {"initialization_mode": "none", "seed": 7},
        "initialization": {"mode": "none", "validated_source": None},
    }
    effective = {
        "data": {"train_sequence_ids": ["train"], "dev_sequence_ids": ["dev"]},
        "training": {"seed": 7},
    }
    protocol = {"folds": [effective["data"]]}
    assert (
        validate_uninitialized_producer_contract(checkpoint, effective, protocol)[
            "initialization_ancestors"
        ]
        == []
    )
    ancestor = copy.deepcopy(checkpoint)
    ancestor["training_config"]["initialization_mode"] = "warm_start"
    with pytest.raises(ValueError, match="ancestor"):
        validate_uninitialized_producer_contract(ancestor, effective, protocol)
    wrong = copy.deepcopy(effective)
    wrong["data"]["train_sequence_ids"].append("dev")
    with pytest.raises(ValueError, match="train sets"):
        validate_uninitialized_producer_contract(checkpoint, wrong, protocol)


def test_producer_that_saw_dev_is_rejected_even_with_matching_hashes() -> None:
    canonical = pd.DataFrame(
        [
            {
                "sample_token": f"t{i}",
                "sequence_id": f"s{i}",
                "track_id": f"k{i}",
                "outer_fold": i // 3,
            }
            for i in range(9)
        ]
    )
    train, dev = ["s5", "s6", "s7", "s8"], ["s3", "s4"]
    split = {
        "train_sequence_ids": train,
        "dev_sequence_ids": dev,
        "train_rows": 4,
        "dev_rows": 2,
        "train_sample_tokens_sha256": hash_sorted_token_strings(["t5", "t6", "t7", "t8"]),
        "dev_sample_tokens_sha256": hash_sorted_token_strings(["t3", "t4"]),
    }
    protocol = {"outer_fold": 0, "inner_fold": 1, "folds": [split]}
    artifact = {"outer_fold": 0, "inner_fold": 1, "role": "inner_oof"}
    output = canonical.loc[canonical.sequence_id.isin(dev)].copy()
    output["outer_fold"] = 0
    output["inner_fold"] = 1
    result = validate_nested_split(
        protocol, artifact, canonical, output, outer_fold=0, inner_fold=1
    )
    assert result["split_relationships_validated"]
    assert not result["ancestry_validated"]
    contaminated = copy.deepcopy(protocol)
    contaminated["folds"][0]["train_sequence_ids"].append("s3")
    contaminated["folds"][0]["train_rows"] = 5
    contaminated["folds"][0]["train_sample_tokens_sha256"] = hash_sorted_token_strings(
        ["t3", "t5", "t6", "t7", "t8"]
    )
    with pytest.raises(ValueError, match="trained on inner-dev"):
        validate_nested_split(contaminated, artifact, canonical, output, outer_fold=0, inner_fold=1)
