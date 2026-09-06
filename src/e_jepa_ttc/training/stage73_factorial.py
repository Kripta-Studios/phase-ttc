"""Prospective architecture recipes and bounded conditional campaign job expansion."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from e_jepa_ttc.evaluation.stage70_76 import Evidence, decide
from e_jepa_ttc.models.projective_phase_network import Arm, Capacity

Family = Literal["C_DIRECT", "C_PPS"]


@dataclass(frozen=True)
class Availability:
    """Technical/data prerequisites resolved before scientific outcomes are available."""

    pretext: bool
    diversity: bool
    pretext_evidence_sha256: str
    diversity_evidence_sha256: str
    capacities: tuple[Capacity, ...] = ("2M",)
    primary_capacity: Capacity = "2M"

    def __post_init__(self) -> None:
        object.__setattr__(self, "capacities", tuple(self.capacities))
        if (self.capacities, self.primary_capacity) not in (
            (("2M",), "2M"),
            (("2M", "4M", "8M"), "4M"),
        ):
            raise ValueError("capacity matrix and primary size must be prospectively registered")
        if type(self.pretext) is not bool or type(self.diversity) is not bool:
            raise ValueError("A and D availability must both be resolved before plan freeze")
        for value in (self.pretext_evidence_sha256, self.diversity_evidence_sha256):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("availability requires a hashed prerequisite receipt")


@dataclass(frozen=True)
class Treatment:
    """Fixed supervised architecture and any independently repeated encoder pretext phase."""

    name: str
    architecture: Arm
    expanded: bool = False
    pretext_updates: int = 0
    supervised_updates: int = 8000
    supervised_warmup: int = 400

    @property
    def optimizer_updates(self) -> int:
        """Composite job reservation includes the entire encoder plus supervised work."""
        return self.pretext_updates + self.supervised_updates


TREATMENTS = (
    Treatment("S72-DIRECT", "DIRECT"),
    Treatment("S72-PPS-PAIR", "PPS-PAIR"),
    Treatment("S72-PPS-TRAJ", "PPS-TRAJ"),
    Treatment("S72-PPS-TIMEPERM", "PPS-TIMEPERM"),
    Treatment("S73-PPS-A", "PPS-TRAJ", pretext_updates=3000),
    Treatment("S73-PPS-D", "PPS-TRAJ", expanded=True),
    Treatment("S73-PPS-AD", "PPS-TRAJ", expanded=True, pretext_updates=3000),
    Treatment("S73-PPS-EXTRA", "PPS-TRAJ", supervised_updates=11000, supervised_warmup=550),
    Treatment("S73-DIRECT-D", "DIRECT", expanded=True),
)


def treatment(name: str) -> Treatment:
    """Reject any renamed control, unregistered recipe or extra search arm."""
    for value in TREATMENTS:
        if value.name == name:
            return value
    raise ValueError("unregistered architecture treatment")


def canonical_candidates(availability: Availability) -> dict[Family, str]:
    """Resolve identities from prerequisites only; failed fits do not change this mapping."""
    direct = "S73-DIRECT-D" if availability.diversity else "S72-DIRECT"
    if availability.pretext and availability.diversity:
        pps = "S73-PPS-AD"
    elif availability.pretext:
        pps = "S73-PPS-A"
    elif availability.diversity:
        pps = "S73-PPS-D"
    else:
        pps = "S72-PPS-TRAJ"
    return {"C_DIRECT": direct, "C_PPS": pps}


@dataclass(frozen=True)
class PlannedFit:
    """Prospective recipe; a concrete executor must additionally bind all source ancestors."""

    stage: Literal["S72", "S73", "S74", "S75"]
    treatment_name: str
    outer_fold: int
    seed: int
    inner_fold: int | None = None
    capacity: Capacity = "2M"

    def __post_init__(self) -> None:
        if self.capacity not in ("2M", "4M", "8M"):
            raise ValueError("unregistered model capacity")
        recipe = treatment(self.treatment_name)
        if self.outer_fold not in (0, 1, 2):
            raise ValueError("all architecture fits use the original three outer folds")
        if self.stage in ("S72", "S73") and (
            self.seed != 7 or not recipe.name.startswith(self.stage)
        ):
            raise ValueError("initial matrix stage or seed changed")
        if self.stage == "S74" and self.seed not in (13, 23):
            raise ValueError("replication requires both registered new seeds")
        if self.stage == "S75":
            if self.seed != 7 or self.inner_fold not in (0, 1, 2):
                raise ValueError("new nested producers require seed7 and three inner folds")
        elif self.inner_fold is not None:
            raise ValueError("inner fold supplied outside nested routing")

    @property
    def job_id(self) -> str:
        """Return a stable identity without a result-dependent suffix."""
        suffix = f"_inner{self.inner_fold}" if self.inner_fold is not None else ""
        capacity = "" if self.capacity == "2M" else f"_capacity{self.capacity}"
        return (
            f"{self.stage}_{self.treatment_name}_outer{self.outer_fold}{suffix}"
            f"_seed{self.seed}{capacity}"
        )


def initial_jobs(availability: Availability) -> tuple[PlannedFit, ...]:
    """S72 is mandatory; available S73 arms and EXTRA do not inspect an S72 outcome."""
    recipes = [
        recipe
        for recipe in TREATMENTS
        if (
            recipe.name.startswith("S72")
            or (
                (not recipe.expanded or availability.diversity)
                and (not recipe.pretext_updates or availability.pretext)
            )
        )
    ]
    return tuple(
        PlannedFit(
            "S72" if recipe.name.startswith("S72") else "S73",
            recipe.name,
            fold,
            7,
            capacity=capacity,
        )
        for recipe in recipes
        for capacity in (
            availability.capacities
            if recipe.name.startswith("S72")
            else (availability.primary_capacity,)
        )
        for fold in (0, 1, 2)
    )


def replication_jobs(
    availability: Availability, evidence: Mapping[Family, Evidence | None]
) -> tuple[PlannedFit, ...]:
    """Apply the exact practical/complementarity gate; no CI-significance prerequisite."""
    if set(evidence) != {"C_DIRECT", "C_PPS"}:
        raise ValueError("both canonical candidates require evidence or an explicit fit block")
    candidates = canonical_candidates(availability)
    jobs = []
    for family, name in candidates.items():
        result = evidence[family]
        if result is not None and decide(result)["replicate_new_architecture"]:
            jobs.extend(
                PlannedFit("S74", name, fold, seed, capacity=availability.primary_capacity)
                for seed in (13, 23)
                for fold in (0, 1, 2)
            )
    return tuple(jobs)


def nested_jobs(
    availability: Availability,
    *,
    joint_oracle_gain: float,
    technically_valid: Mapping[Family, bool],
) -> tuple[PlannedFit, ...]:
    """Include every technically valid canonical expert once joint oracle headroom reaches8."""
    if not math.isfinite(joint_oracle_gain) or set(technically_valid) != {"C_DIRECT", "C_PPS"}:
        raise ValueError("complete joint oracle and canonical endpoint evidence required")
    if joint_oracle_gain < 8:
        return ()
    return tuple(
        PlannedFit("S75", name, fold, 7, inner, capacity=availability.primary_capacity)
        for family, name in canonical_candidates(availability).items()
        if technically_valid[family]
        for fold in (0, 1, 2)
        for inner in (0, 1, 2)
    )


def stage_update_total(jobs: tuple[PlannedFit, ...]) -> int:
    """Count pretraining honestly, separately from architecture endpoint count."""
    return sum(treatment(job.treatment_name).optimizer_updates for job in jobs)
