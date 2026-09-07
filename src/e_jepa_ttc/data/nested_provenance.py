"""Validate nested split relationships against canonical rows, not declared flags."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.data.canonical_token_identity import hash_sorted_token_strings


def validate_exact_oof_union(aggregate: pd.DataFrame, parts: list[pd.DataFrame]) -> None:
    """Prove aggregate rows and all their fields equal the verified producer union."""
    if not parts:
        raise ValueError("nested OOF union has no verified producer parts")
    key = "token_id" if "token_id" in aggregate else "sample_token"
    union = pd.concat(parts, ignore_index=True)
    if (
        key not in union
        or aggregate[key].duplicated().any()
        or union[key].duplicated().any()
        or set(aggregate.columns) != set(union.columns)
    ):
        raise ValueError("nested OOF aggregate inventory/uniqueness mismatch")
    columns = sorted(aggregate.columns)
    left = aggregate.sort_values(key)[columns].reset_index(drop=True)
    right = union.sort_values(key)[columns].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise ValueError("nested OOF aggregate differs from verified producer union") from error


def validate_uninitialized_producer_contract(
    checkpoint: dict[str, Any], effective: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    """Bind effective training sets/config to a producer with no fitted initializer.

    Producers with an initialization ancestor require a separate recursive audit;
    this validator deliberately does not infer safety from an ancestor hash.
    """
    training = checkpoint["training_config"]
    initialization = checkpoint["initialization"]
    if (
        training.get("initialization_mode") != "none"
        or training.get("initialization_checkpoint") is not None
        or training.get("initialization_checkpoint_sha256") is not None
        or initialization.get("mode") != "none"
        or initialization.get("validated_source") is not None
        or training.get("soft_geometry_teacher_checkpoint") is not None
    ):
        raise ValueError("producer has an initialization ancestor requiring recursive validation")
    split = protocol["folds"][0]
    for role in ("train", "dev"):
        if set(effective["data"][f"{role}_sequence_ids"]) != set(split[f"{role}_sequence_ids"]):
            raise ValueError(f"effective producer {role} sets disagree with nested contract")
    for key, value in effective["training"].items():
        if key in training and training[key] != value:
            raise ValueError(f"checkpoint/effective training configuration mismatch: {key}")
    return {
        "initialization_ancestors": [],
        "checkpoint_initialization_validated": True,
        "effective_training_sets_validated": True,
        "representation_teacher_requires_separate_validation": bool(
            training.get("representation_teacher_cache_artifact_sha256")
        ),
        "representation_teacher_artifact_sha256": training.get(
            "representation_teacher_cache_artifact_sha256"
        ),
    }


def validate_frozen_teacher_dependency(
    initialization: dict[str, Any],
    teacher_audit: dict[str, Any],
    *,
    required_tokens: Sequence[str] | None = None,
) -> None:
    """Require the exact independently audited, preexisting teacher dependency."""
    expected = initialization.get("representation_teacher_artifact_sha256")
    if expected is None:
        return
    expected_count = 8192
    if required_tokens is not None:
        tokens = list(required_tokens)
        if (
            not tokens
            or any(not isinstance(t, str) or not t for t in tokens)
            or len(set(tokens)) != len(tokens)
        ):
            raise ValueError("teacher token universe must be nonempty and unique")
        expected_count = len(tokens)
        if teacher_audit.get("verified_tokens_sha256") != hash_sorted_token_strings(tokens):
            raise ValueError("teacher audit does not cover the exact required token universe")
    if (
        teacher_audit.get("status") != "passed"
        or teacher_audit.get("preexisting_frozen_teacher") is not True
        or teacher_audit.get("fold_fitted_teacher") is not False
        or teacher_audit.get("teacher_artifact_sha256") != expected
        or teacher_audit.get("verified_tokens") != expected_count
        or teacher_audit.get("materializer_ast_no_fit_calls") is not True
    ):
        raise ValueError("producer representation teacher dependency is not verified frozen")


def validate_nested_split(
    protocol: dict[str, Any],
    artifact: dict[str, Any],
    canonical: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    outer_fold: int,
    inner_fold: int | None,
    producer_recipe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require exact disjoint producer training/dev sets and row-level fold identity."""
    if canonical["sample_token"].duplicated().any():
        raise ValueError("canonical nested universe contains duplicate tokens")
    if protocol.get("outer_fold") != outer_fold or artifact.get("outer_fold") != outer_fold:
        raise ValueError("nested outer_fold disagrees across contracts")
    if protocol.get("inner_fold") != inner_fold or artifact.get("inner_fold") != inner_fold:
        raise ValueError("nested inner_fold disagrees across contracts")
    expected_role = "outer_dev" if inner_fold is None else "inner_oof"
    if artifact.get("role") != expected_role:
        raise ValueError("nested producer role disagrees")
    folds = protocol.get("folds", [])
    if len(folds) != 1:
        raise ValueError("producer must bind one exact nested split")
    split = folds[0]
    train_sequences = set(split["train_sequence_ids"])
    dev_sequences = set(split["dev_sequence_ids"])
    outer_dev = set(canonical.loc[canonical.outer_fold == outer_fold, "sequence_id"])
    outer_train = set(canonical.sequence_id) - outer_dev
    if train_sequences & dev_sequences or train_sequences & outer_dev:
        raise ValueError("nested producer trained on inner-dev or outer-dev")
    diagnostic: set[str] = set()
    if producer_recipe is not None:
        identity = producer_recipe.get("artifact_sha256")
        if (
            object_digest({k: v for k, v in producer_recipe.items() if k != "artifact_sha256"})
            != identity
            or producer_recipe.get("kind") != "CAPACITY_NESTED_PRODUCER_ROLES"
            or protocol.get("producer_roles_sha256") != identity
            or artifact.get("producer_roles_sha256") != identity
        ):
            raise ValueError("expanded producer roles differ from bound protocol and artifact")
        matches = [r for r in producer_recipe["folds"] if r["outer_fold"] == outer_fold]
        if len(matches) != 1:
            raise ValueError("expanded producer requires one exact outer role")
        role = matches[0]
        diagnostic = set(role["diagnostic_sequences"])
        fit = set(role["fit_sequences"])
        excluded = outer_dev | diagnostic
        if (
            not diagnostic
            or not fit
            or diagnostic & (fit | outer_dev)
            or fit | diagnostic != outer_train
            or set(role["outer_held_sequences"]) != outer_dev
            or set(role["all_fit_ancestor_exclusions"]) != excluded
            or set(role["normalization_and_pretext_exclusions"]) != excluded
        ):
            raise ValueError("expanded outer diagnostic or ancestor exclusions changed")
        expected_train, expected_dev = fit, outer_dev
        if inner_fold is not None:
            inners = [r for r in role["inner_roles"] if r["inner_fold"] == inner_fold]
            if len(inners) != 1:
                raise ValueError("expanded producer requires one exact inner role")
            expected_train = set(inners[0]["fit_sequences"])
            expected_dev = set(inners[0]["prediction_sequences"])
            if (
                not expected_train
                or not expected_dev
                or expected_train & expected_dev
                or expected_train | expected_dev != fit
                or set(inners[0]["all_fit_ancestor_exclusions"]) != excluded | expected_dev
            ):
                raise ValueError("expanded inner partition or ancestor exclusions changed")
        if train_sequences != expected_train or dev_sequences != expected_dev:
            raise ValueError("expanded producer differs from its exact registered role")
        if "acquisition_group" not in canonical or canonical.acquisition_group.isna().any():
            raise ValueError("expanded producer requires acquisition identities")
        assigned = canonical.assign(
            source_role=canonical.sequence_id.map(
                {
                    s: role_name
                    for role_name, members in (
                        ("train", train_sequences),
                        ("dev", dev_sequences),
                        ("diagnostic", diagnostic),
                        (
                            "excluded",
                            set(canonical.sequence_id)
                            - train_sequences
                            - dev_sequences
                            - diagnostic,
                        ),
                    )
                    for s in members
                }
            )
        )
        if assigned.groupby("acquisition_group").source_role.nunique().max() != 1:
            raise ValueError("expanded acquisition crosses producer roles")
    elif inner_fold is None:
        if train_sequences != outer_train or dev_sequences != outer_dev:
            raise ValueError("outer-final split differs from canonical split")
    elif (
        train_sequences | dev_sequences != outer_train
        or dev_sequences & outer_dev
        or len(train_sequences) != 4
        or len(dev_sequences) != 2
    ):
        raise ValueError("inner split is not the prescribed four-train/two-dev partition")
    for role, sequences in (("train", train_sequences), ("dev", dev_sequences)):
        rows = canonical.loc[canonical.sequence_id.isin(sequences)]
        if len(rows) != split[f"{role}_rows"]:
            raise ValueError(f"nested {role} row count disagrees")
        if hash_sorted_token_strings(rows.sample_token) != split[f"{role}_sample_tokens_sha256"]:
            raise ValueError(f"nested {role} token hash disagrees")
    expected = canonical.loc[canonical.sequence_id.isin(dev_sequences)].set_index("sample_token")
    actual = predictions.rename(columns={"token_id": "sample_token"}).set_index("sample_token")
    if actual.index.duplicated().any() or set(actual.index) != set(expected.index):
        raise ValueError("nested output is not the exact producer dev universe")
    actual = actual.loc[expected.index]
    for field in ("sequence_id", "track_id"):
        if not actual[field].astype(str).equals(expected[field].astype(str)):
            raise ValueError(f"nested output {field} differs from canonical identity")
    # The expert CSV labels the producing outer experiment, while canonical
    # metadata labels the fold holding out this token. These differ for inner OOF.
    if not actual["outer_fold"].eq(outer_fold).all():
        raise ValueError("nested output producer outer_fold differs from its contract")
    if inner_fold is not None and not actual["inner_fold"].eq(inner_fold).all():
        raise ValueError("nested output inner_fold differs from producer")
    result = {
        "split_relationships_validated": True,
        "train_sequence_ids": sorted(train_sequences),
        "dev_sequence_ids": sorted(dev_sequences),
        "excluded_outer_dev_sequence_ids": sorted(outer_dev),
        "ancestry_validated": False,
        "output_outer_fold_semantics": "producer_outer_experiment",
    }
    if producer_recipe is not None:
        result.update(
            producer_roles_sha256=producer_recipe["artifact_sha256"],
            excluded_diagnostic_sequence_ids=sorted(diagnostic),
        )
    return result
