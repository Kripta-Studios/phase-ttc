from __future__ import annotations

import copy

import pandas as pd
import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.data.canonical_token_identity import hash_sorted_token_strings
from e_jepa_ttc.data.nested_provenance import (
    validate_accounted_comparator_ancestry,
    validate_exact_oof_union,
    validate_frozen_teacher_dependency,
    validate_nested_split,
    validate_uninitialized_producer_contract,
)
from e_jepa_ttc.training.capacity_campaign import producer_roles


def test_capacity_joint_tables_ridge_publication_and_transitive_parent_attack(
    tmp_path, monkeypatch
):
    """Real ridge/physical publication with synthetic neural-contract fixtures, zero NN fits."""
    import json
    from dataclasses import asdict
    from types import SimpleNamespace

    import numpy as np
    import torch
    import yaml

    from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding
    from e_jepa_ttc.evaluation.initial_campaign_evidence_v11 import comparison_source_identity
    from e_jepa_ttc.evaluation.initial_treatment_export_v11 import publish_export_payloads
    from e_jepa_ttc.evaluation.nested_treatment_export_v11 import historical_table
    from e_jepa_ttc.models.three_expert_router import BASE8_FEATURES
    from e_jepa_ttc.training.capacity_campaign import (
        comparator_development_contract,
        comparator_epoch_updates,
    )
    from e_jepa_ttc.training.stage61_pair_head import PairHeadTrainingConfig
    from e_jepa_ttc.training.stage70_update_ledger import FitSpec
    from scripts.run_scientific_recovery_v9_stage65 import publish_capacity_risk17

    canonical = pd.DataFrame(
        {
            "sample_token": [f"t{i}" for i in range(54)],
            "sequence_id": [f"s{i // 2}" for i in range(54)],
            "track_id": [f"track{i % 2}" for i in range(54)],
            "target_ttc": np.resize([2.0, 4.0, 8.0, -2.0], 54),
        }
    )
    sequences = canonical.sequence_id.unique().tolist()
    mapping = {s: int(s[1:]) % 3 for s in sequences}
    cohort = {
        "kind": "CAPACITY_DEVELOPMENT_COHORT",
        "recipe": {"prefix": "fixture"},
        "selected_sequences": sequences,
        "ordered_tokens": canonical.sample_token.tolist(),
        "sequence_to_group": {s: f"acq_{s}" for s in sequences},
        "sequence_to_outer_fold": mapping,
        "confirmation_sequences": [],
        "protected_sequences": [],
        "folds": [
            {
                "outer_fold": fold,
                "train_sequences": [s for s in sequences if mapping[s] != fold],
                "held_sequences": [s for s in sequences if mapping[s] == fold],
                "held_tokens": canonical.loc[
                    canonical.sequence_id.map(mapping) == fold, "sample_token"
                ].tolist(),
            }
            for fold in range(3)
        ],
    }
    cohort["artifact_sha256"] = object_digest(cohort)
    roles = producer_roles(cohort)
    atomic_json(tmp_path / "cohort.json", cohort)
    canonical.to_csv(tmp_path / "targets.csv", index=False)
    jobs, states, protocols = {}, {}, {}
    for fold in range(3):
        for inner in (None, 0, 1, 2):
            protocol = comparator_development_contract(
                cohort, canonical, roles, outer_fold=fold, inner_fold=inner
            )
            protocol["status"] = "frozen_before_capacity_comparator_training"
            protocol_path = tmp_path / f"protocol_{fold}_{inner}.json"
            atomic_json(protocol_path, protocol)
            split = protocol["folds"][0]
            for arm in ("A5", "C2F", "PAIR"):
                name = f"{arm}_{fold}_{inner}"
                output = tmp_path / name
                output.mkdir()
                config_path = output / "config.json"
                training = {
                    "seed": 7,
                    "epochs": 18,
                    "batch_size": 32,
                    "gradient_accumulation_steps": 1,
                    "checkpoint_selection_mode": "last_epoch",
                    "initialization_mode": "none",
                }
                effective = {
                    "training": training,
                    "data": {
                        f"{r}_sequence_ids": split[f"{r}_sequence_ids"] for r in ("train", "dev")
                    },
                }
                if arm == "PAIR":
                    atomic_json(config_path, asdict(PairHeadTrainingConfig(seed=7)))
                    endpoint_updates = 6840
                else:
                    config_path.write_text(yaml.safe_dump(effective), encoding="utf-8")
                    endpoint_updates = comparator_epoch_updates(
                        split["train_rows"], epochs=18, batch_size=32, accumulation=1
                    )
                spec = FitSpec(
                    name,
                    "COMPARATOR",
                    arm,
                    fold,
                    7,
                    endpoint_updates,
                    binding(config_path)["sha256"],
                    binding(protocol_path)["sha256"],
                )
                job = {
                    "spec": asdict(spec),
                    "protocol": binding(protocol_path),
                    "output_root": str(output),
                }
                job["pair_config" if arm == "PAIR" else "effective_config"] = binding(config_path)
                if arm == "PAIR":
                    job["a5_job_id"] = f"A5_{fold}_{inner}"
                else:
                    checkpoint = {
                        "artifact_type": "causal_scale_eap_resume_state_v1",
                        "fit_spec": asdict(spec),
                        "epoch": 18,
                        "training_config": training,
                        "train_samples": split["train_rows"],
                        "validation_samples": split["dev_rows"],
                    }
                    torch.save(checkpoint, output / "endpoint.pt")
                    states[name] = {
                        "spec": asdict(spec),
                        "status": "COMPLETE",
                        "endpoint": binding(output / "endpoint.pt"),
                    }
                jobs[name], protocols[name] = job, protocol
    lock = {
        "campaign_root": str(tmp_path),
        "capacity_cohort": binding(tmp_path / "cohort.json"),
        "capacity_target_rows": binding(tmp_path / "targets.csv"),
        "comparator_jobs": jobs,
        "capacity_comparison_output": str(tmp_path / "comparison"),
        "stage_barrier_root": str(tmp_path / "barriers"),
    }
    lock_sha = object_digest(lock)
    for name, job in jobs.items():
        spec, protocol = job["spec"], protocols[name]
        arm, fold, inner = spec["arm"], spec["fold"], protocol["inner_fold"]
        output = tmp_path / name
        identity = {"training_lock_sha256": lock_sha, "job": job}
        if arm == "PAIR":
            parent = jobs[job["a5_job_id"]]
            identity.update(
                a5_job=parent,
                a5_endpoint=states[job["a5_job_id"]]["endpoint"],
                protocol=job["protocol"],
            )
            train = canonical.loc[
                canonical.sequence_id.isin(protocol["folds"][0]["train_sequence_ids"])
            ].rename(columns={"target_ttc": "target_ttc_s"})
            inputs = output / "training_inputs"
            inputs.mkdir()
            train_rows, train_features = publish_export_payloads(
                inputs, train, {"pair_features": np.zeros((len(train), 133), np.float32)}
            )
            torch.save(
                {
                    "completed_updates": 6840,
                    "identity": {
                        **identity,
                        "fit_spec": spec,
                        "training_rows": train_rows,
                        "training_features": train_features,
                        "config": asdict(PairHeadTrainingConfig(seed=7)),
                    },
                },
                output / "endpoint.pt",
            )
            states[name] = {
                "spec": spec,
                "status": "COMPLETE",
                "endpoint": binding(output / "endpoint.pt"),
            }
        frame = (
            canonical.loc[canonical.sequence_id.isin(protocol["prediction_sequence_ids"])]
            .rename(columns={"sample_token": "token_id"})
            .copy()
        )
        frame["outer_fold"], frame["inner_fold"] = fold, inner
        frame["prediction_ttc"] = frame.target_ttc + {"A5": 0.0, "C2F": 0.4, "PAIR": 0.8}[arm]
        if arm != "PAIR":
            for field in BASE8_FEATURES:
                frame[field] = 0.0
        directory = output / "prediction"
        directory.mkdir()
        rows, arrays = publish_export_payloads(directory, frame, {})
        receipt = {
            "kind": "CAPACITY_PAIR_COMPARATOR_EXPORT"
            if arm == "PAIR"
            else "CAPACITY_NATIVE_COMPARATOR_EXPORT",
            "identity": identity,
            "endpoint": states[name]["endpoint"],
            "rows": rows,
            "arrays": arrays,
            "outer_fold": fold,
            "inner_fold": inner,
            "role": "outer_dev" if inner is None else "inner_oof",
            "producer_roles_sha256": roles["artifact_sha256"],
            "ancestry_validated": True,
        }
        if arm == "PAIR":
            receipt["a5_export"] = binding(
                tmp_path / job["a5_job_id"] / "prediction/COMPARATOR_EXPORT.json"
            )
        receipt["artifact_sha256"] = object_digest(receipt)
        atomic_json(
            directory / ("PAIR_EXPORT.json" if arm == "PAIR" else "COMPARATOR_EXPORT.json"), receipt
        )
    monkeypatch.setattr("e_jepa_ttc.training.stage72_jobs.require_scientific_lock", lambda *_: lock)
    monkeypatch.setattr(
        "e_jepa_ttc.artifacts.stage_endpoint_barrier_v11.verify_initial_barrier",
        lambda *a, **k: None,
    )
    owner = SimpleNamespace(root=tmp_path / "owner", check=lambda: None)
    owner.root.mkdir()
    attempts = {}

    def record_attempt(attempt, completed):
        attempts[attempt["sha256"]] = completed

    ledger = SimpleNamespace(
        owner_root=owner.root,
        state={"prior_campaign": {}, "jobs": states},
        record_ridge_attempt=record_attempt,
    )
    result = publish_capacity_risk17(tmp_path / "lock", lock_sha, ledger, owner)
    assert len(result["tables"]) == 6 and len(attempts) == 3 and all(attempts.values())
    assert len(result["producer_exports"]) == 36
    assert comparison_source_identity(lock)["capacity_comparison_rows"] == result["comparison_rows"]
    table, _ = historical_table(lock, 0, "inner_oof")
    assert table.metadata.role.eq("inner_oof").all()
    assert table.record["mass_rule"] == "present_bucket_macro"
    assert publish_capacity_risk17(tmp_path / "lock", lock_sha, ledger, owner) == result
    assert len(attempts) == 3, "resume must not solve an already published ridge again"
    path = tmp_path / "PAIR_0_0/prediction/PAIR_EXPORT.json"
    damaged = json.loads(path.read_text(encoding="utf-8"))
    damaged["identity"]["a5_endpoint"] = states["A5_1_0"]["endpoint"]
    damaged["artifact_sha256"] = object_digest(
        {k: v for k, v in damaged.items() if k != "artifact_sha256"}
    )
    atomic_json(path, damaged)
    with pytest.raises(ValueError, match="different nested producer"):
        publish_capacity_risk17(tmp_path / "lock", lock_sha, ledger, owner)


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
    if diagnostic_contract:
        training = {
            "epochs": 18,
            "checkpoint_selection_mode": "last_epoch",
            "seed": 7,
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "initialization_mode": "none",
        }
        spec = {
            "technical": False,
            "arm": "A5",
            "fold": 0,
            "seed": 7,
            "endpoint": 18 * ((split["train_rows"] + 31) // 32),
        }
        checkpoint = {
            "artifact_type": "causal_scale_eap_resume_state_v1",
            "fit_spec": spec,
            "epoch": 18,
            "training_config": training,
            "train_samples": split["train_rows"],
            "validation_samples": split["dev_rows"],
        }
        effective = {
            "training": dict(training),
            "data": {f"{r}_sequence_ids": split[f"{r}_sequence_ids"] for r in ("train", "dev")},
        }
        artifact["identity"] = {"job": {"spec": spec}}

        def audit():
            return validate_accounted_comparator_ancestry(
                checkpoint,
                effective,
                protocol,
                artifact,
                canonical,
                output,
                recipe,
                teacher_audit=None,
            )

        assert audit()["ancestry_validated"] is True
        checkpoint["epoch"] = 17
        with pytest.raises(ValueError, match="complete prescribed"):
            audit()
        checkpoint["epoch"] = 18
        training["initialization_checkpoint"] = "excluded-ancestor.pt"
        with pytest.raises(ValueError, match="fitted initialization"):
            audit()
        training.pop("initialization_checkpoint")
        training["representation_teacher_cache_artifact_sha256"] = "a" * 64
        with pytest.raises(ValueError, match="separately verified"):
            audit()
        training.pop("representation_teacher_cache_artifact_sha256")
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
