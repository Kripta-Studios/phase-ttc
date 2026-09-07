from __future__ import annotations

import copy

import pandas as pd
import pytest

from e_jepa_ttc.data.canonical_token_identity import hash_sorted_token_strings
from e_jepa_ttc.data.nested_provenance import (
    validate_exact_oof_union,
    validate_frozen_teacher_dependency,
    validate_nested_split,
    validate_uninitialized_producer_contract,
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
