"""Closed capacity graph and shared optimizer-budget admission after its predecessor."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.data.capacity_cohort import grouped_folds
from e_jepa_ttc.data.stage70_roles import stable_rank
from e_jepa_ttc.training.stage73_factorial import (
    Availability,
    PlannedFit,
    canonical_candidates,
    initial_jobs,
    stage_update_total,
)


def capacity_graph(availability: Availability) -> dict[str, Any]:
    """Register all36 S72 endpoints and restrict investigations to prospective4M."""
    if availability.capacities != ("2M", "4M", "8M") or availability.primary_capacity != "4M":
        raise ValueError("expanded graph requires all three sizes and the fixed4M primary")
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
