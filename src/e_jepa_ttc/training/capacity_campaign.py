"""Closed capacity graph and shared optimizer-budget admission after its predecessor."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.data.canonical_token_identity import hash_sorted_token_strings
from e_jepa_ttc.data.capacity_cohort import grouped_folds
from e_jepa_ttc.data.stage70_roles import stable_rank
from e_jepa_ttc.training.stage73_factorial import (
    Availability,
    PlannedFit,
    canonical_candidates,
    initial_jobs,
    stage_update_total,
)


def comparator_epoch_updates(
    training_rows: int, *, epochs: int, batch_size: int, accumulation: int
) -> int:
    """Count the existing causal-scale loader's final partial batch and accumulation.

    The loader does not drop rows and train_one_real_epoch steps on the final
    batch even when the accumulation group is incomplete. This is a reservation
    for a fixed last-epoch endpoint, never a count inferred from early stopping.
    """
    if any(type(n) is not int or n < 1 for n in (training_rows, epochs, batch_size, accumulation)):
        raise ValueError("comparator row and execution counts must be positive integers")
    batches = (training_rows + batch_size - 1) // batch_size
    return epochs * ((batches + accumulation - 1) // accumulation)


def capacity_graph(availability: Availability) -> dict[str, Any]:
    """Register the prospective size scope with the fixed4M primary."""
    if availability.capacities not in (("4M",), ("2M", "4M", "8M")) or (
        availability.primary_capacity != "4M"
    ):
        raise ValueError("expanded graph requires the registered size scope and fixed4M primary")
    initial = initial_jobs(availability)
    canonical = canonical_candidates(availability)
    replication = tuple(
        PlannedFit("S74", name, fold, seed, capacity="4M")
        for name in canonical.values()
        for seed in (13, 23)
        for fold in range(3)
    )
    nested = tuple(
        PlannedFit("S75", name, fold, 7, inner, capacity="4M")
        for name in canonical.values()
        for fold in range(3)
        for inner in range(3)
    )
    stages = {
        "S72": tuple(job for job in initial if job.stage == "S72"),
        "S73": tuple(job for job in initial if job.stage == "S73"),
        "S74": replication,
        "S75": nested,
    }
    return {
        "kind": "PROSPECTIVE_CAPACITY_JOB_GRAPH",
        "primary_hypothesis": {"treatment": "S72-PPS-TRAJ", "capacity": "4M"},
        "canonical_candidates": {
            family: {"treatment": name, "capacity": "4M"} for family, name in canonical.items()
        },
        "availability": asdict(availability),
        "jobs": {stage: [asdict(job) for job in jobs] for stage, jobs in stages.items()},
        "optimizer_updates": {stage: stage_update_total(jobs) for stage, jobs in stages.items()},
        "architecture_maximum_updates": stage_update_total(initial + replication + nested),
        "conditional_stages": ["S74", "S75", "S76"],
        "S73_requires_positive_S72": False,
        "comparator_and_S76_updates_included": False,
        "scientific_execution_admitted": False,
    }


def admit_capacity_budget(
    graph: dict[str, Any],
    *,
    predecessor_terminal: bool,
    charged_updates: int,
    outstanding_reserved_updates: int,
    comparator_updates: int | None,
    confirmation_updates: int | None,
    technical_updates_reserved: int,
    maximum_updates: int = 900_000,
) -> dict[str, Any]:
    """Never spend the original campaign's future budget or assume unknown fits cost zero."""
    counts = (charged_updates, outstanding_reserved_updates, technical_updates_reserved)
    if type(maximum_updates) is not int or maximum_updates < 1:
        raise ValueError("shared maximum must be a positive integer")
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError("shared ledger counts must be nonnegative integers")
    for count in (comparator_updates, confirmation_updates):
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("invalid external fit budget")
    reasons = []
    if not predecessor_terminal:
        reasons.append("ORIGINAL_CAMPAIGN_NOT_TERMINAL")
    if outstanding_reserved_updates:
        reasons.append("PREDECESSOR_HAS_OUTSTANDING_RESERVATIONS")
    if comparator_updates is None:
        reasons.append("COMPARATOR_BUDGET_UNRESOLVED")
    if confirmation_updates is None:
        reasons.append("CONFIRMATION_RECIPE_OR_BRANCH_DISPOSITION_UNRESOLVED")
    required = None
    if comparator_updates is not None and confirmation_updates is not None:
        required = (
            charged_updates
            + outstanding_reserved_updates
            + graph["architecture_maximum_updates"]
            + comparator_updates
            + confirmation_updates
            + technical_updates_reserved
        )
        if required > maximum_updates:
            reasons.append("SHARED_UPDATE_BUDGET_EXCEEDED")
    return {
        "status": "ADMITTED" if not reasons else "PREREQUISITE_BLOCKED",
        "reasons": reasons,
        "maximum_updates": maximum_updates,
        "required_total_updates": required,
        "charged_updates": charged_updates,
        "outstanding_reserved_updates": outstanding_reserved_updates,
        "architecture_updates": graph["architecture_maximum_updates"],
        "comparator_updates": comparator_updates,
        "confirmation_updates": confirmation_updates,
        "technical_updates_reserved": technical_updates_reserved,
        "partial_endpoints_allowed": False,
    }


def producer_roles(cohort: dict[str, Any]) -> dict[str, Any]:
    """Reserve one shared TRAIN diagnostic acquisition and nest only optimizer groups."""
    unsigned = {k: v for k, v in cohort.items() if k != "artifact_sha256"}
    if object_digest(unsigned) != cohort.get("artifact_sha256"):
        raise ValueError("cohort changed before producer roles were resolved")
    mapping = cohort["sequence_to_group"]
    prefix = cohort["recipe"]["prefix"]
    folds = []
    for source in cohort["folds"]:
        outer = source["outer_fold"]
        groups = sorted({mapping[s] for s in source["train_sequences"]})
        if len(groups) < 2:
            raise ValueError("TRAIN pool cannot separate optimization and diagnostic acquisitions")
        diagnostic_group = min(groups, key=lambda g: (stable_rank(f"{prefix}/diag/{outer}", g), g))
        diagnostic = sorted(s for s in source["train_sequences"] if mapping[s] == diagnostic_group)
        fit_sequences = sorted(set(source["train_sequences"]) - set(diagnostic))
        excluded = sorted(set(source["held_sequences"]) | set(diagnostic))
        inner = (
            grouped_folds(fit_sequences, mapping, prefix=f"{prefix}/optimizer_inner/{outer}")
            if len({mapping[s] for s in fit_sequences}) >= 3
            else None
        )
        folds.append(
            {
                "outer_fold": outer,
                "outer_held_sequences": source["held_sequences"],
                "diagnostic_sequences": diagnostic,
                "fit_sequences": fit_sequences,
                "all_fit_ancestor_exclusions": excluded,
                "normalization_and_pretext_exclusions": excluded,
                "inner_sequence_to_fold": inner,
                "inner_roles": []
                if inner is None
                else [
                    {
                        "inner_fold": i,
                        "fit_sequences": sorted(s for s in fit_sequences if inner[s] != i),
                        "prediction_sequences": sorted(s for s in fit_sequences if inner[s] == i),
                        "all_fit_ancestor_exclusions": sorted(
                            set(excluded) | {s for s in fit_sequences if inner[s] == i}
                        ),
                    }
                    for i in range(3)
                ],
            }
        )
    result = {
        "kind": "CAPACITY_NESTED_PRODUCER_ROLES",
        "cohort_sha256": cohort["artifact_sha256"],
        "folds": folds,
        "shared_across_sizes_arms_seeds_and_comparators": True,
        "diagnostic_checkpoint_selection_allowed": False,
        "optimizer_updates": 0,
    }
    return {**result, "artifact_sha256": object_digest(result)}


def comparator_development_contract(
    cohort: dict[str, Any],
    metadata: pd.DataFrame,
    roles: dict[str, Any],
    *,
    outer_fold: int,
    inner_fold: int | None,
) -> dict[str, Any]:
    """Compile optimizer/diagnostic pools while keeping prediction groups unread by fits.

    The returned recipe is not a training authorization. The caller must bind
    the physical inputs and freeze it with the complete campaign before use.
    """
    if outer_fold not in (0, 1, 2) or inner_fold not in (None, 0, 1, 2):
        raise ValueError("unregistered comparator producer fold")
    forbidden = set(cohort["confirmation_sequences"]) | set(cohort["protected_sequences"])
    selected = set(cohort["selected_sequences"])
    mapping = cohort["sequence_to_group"]
    if selected & forbidden or {mapping[s] for s in selected} & {mapping[s] for s in forbidden}:
        raise ValueError("comparator cohort contains a protected acquisition")
    if roles != producer_roles(cohort):
        raise ValueError("comparator roles differ from prospective acquisition assignment")
    if (
        metadata.sample_token.duplicated().any()
        or list(metadata.sample_token) != cohort["ordered_tokens"]
        or set(metadata.sequence_id) != set(cohort["selected_sequences"])
    ):
        raise ValueError("comparator metadata differs from complete cohort")
    for fold in cohort["folds"]:
        actual = metadata.loc[
            metadata.sequence_id.isin(fold["held_sequences"]), "sample_token"
        ].tolist()
        if actual != fold["held_tokens"]:
            raise ValueError("comparator token sequence identity changed")
    outer = next(r for r in roles["folds"] if r["outer_fold"] == outer_fold)
    diagnostic = outer["diagnostic_sequences"]
    if inner_fold is None:
        train = outer["fit_sequences"]
        prediction = outer["outer_held_sequences"]
        excluded = outer["all_fit_ancestor_exclusions"]
    else:
        inner = next(r for r in outer["inner_roles"] if r["inner_fold"] == inner_fold)
        train = inner["fit_sequences"]
        prediction = inner["prediction_sequences"]
        excluded = inner["all_fit_ancestor_exclusions"]
    assignment = {
        s: name
        for name, sequences in (
            ("train", train),
            ("diagnostic", diagnostic),
            ("prediction", prediction),
            (
                "excluded",
                set(cohort["selected_sequences"]) - set(train) - set(diagnostic) - set(prediction),
            ),
        )
        for s in sequences
    }
    group_roles: dict[str, set[str]] = {}
    for sequence, role in assignment.items():
        group_roles.setdefault(cohort["sequence_to_group"][sequence], set()).add(role)
    if any(len(values) != 1 for values in group_roles.values()):
        raise ValueError("comparator acquisition crosses source roles")
    split: dict[str, Any] = {"fold": outer_fold}
    for name, sequences in (("train", train), ("dev", diagnostic)):
        tokens = metadata.loc[metadata.sequence_id.isin(sequences), "sample_token"]
        split[f"{name}_sequence_ids"] = sorted(sequences)
        split[f"{name}_rows"] = len(tokens)
        split[f"{name}_sample_tokens_sha256"] = hash_sorted_token_strings(tokens)
    prediction_tokens = metadata.loc[metadata.sequence_id.isin(prediction), "sample_token"]
    return {
        "artifact_type": "scientific_recovery_v11_capacity_comparator_diagnostic_v1",
        "status": "prepared_not_authorized",
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "producer_roles_sha256": roles["artifact_sha256"],
        "cohort_sha256": cohort["artifact_sha256"],
        "sequence_ids": sorted(set(train) | set(diagnostic)),
        "sample_count": split["train_rows"] + split["dev_rows"],
        "cache_universe_rows": len(metadata),
        "folds": [split],
        "prediction_sequence_ids": sorted(prediction),
        "prediction_rows": len(prediction_tokens),
        "prediction_sample_tokens_sha256": hash_sorted_token_strings(prediction_tokens),
        "all_fit_ancestor_exclusions": sorted(excluded),
        "normalization_and_pretext_exclusions": sorted(excluded),
        "dev_role": "TRAIN_diagnostic_only",
        "checkpoint_selection_mode": "last_epoch",
        "fixed_epochs": 18,
        "checks": {
            "train_only_grouped_dev": True,
            "sequence_disjoint_folds": True,
            "sample_token_unique": True,
            "same_cache_universe": True,
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
        },
    }
