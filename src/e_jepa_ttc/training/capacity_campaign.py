"""Closed capacity graph and shared optimizer-budget admission after its predecessor."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

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
) -> dict[str, Any]:
    """Never spend the original campaign's future budget or assume unknown fits cost zero."""
    counts = (charged_updates, outstanding_reserved_updates, technical_updates_reserved)
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
        if required > 900_000:
            reasons.append("SHARED_900000_UPDATE_BUDGET_EXCEEDED")
    return {
        "status": "ADMITTED" if not reasons else "PREREQUISITE_BLOCKED",
        "reasons": reasons,
        "maximum_updates": 900_000,
        "required_total_updates": required,
        "charged_updates": charged_updates,
        "outstanding_reserved_updates": outstanding_reserved_updates,
        "architecture_updates": graph["architecture_maximum_updates"],
        "comparator_updates": comparator_updates,
        "confirmation_updates": confirmation_updates,
        "technical_updates_reserved": technical_updates_reserved,
        "partial_endpoints_allowed": False,
    }
