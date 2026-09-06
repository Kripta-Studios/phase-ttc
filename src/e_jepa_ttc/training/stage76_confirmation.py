"""Prospective standalone confirmation eligibility and development-only family selection."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.training.stage73_factorial import Family, PlannedFit


def publish_blocked_confirmation(
    lock: dict[str, Any], replication: dict[str, Any], output: Path
) -> dict[str, Any]:
    """Close the audited unavailable S76 branch without touching confirmation sources.

    The frozen prerequisite receipt deliberately authorizes no final fit. Resolving
    its missing data requires a visible amendment, never a silent weaker comparator.
    """
    root = Path(lock["campaign_root"])
    if output.resolve() != root / "S76_DISPOSITION.json":
        raise ValueError("confirmation disposition escaped the campaign")
    prerequisite_source = lock["confirmation_prerequisites"]
    prerequisite = json.loads(verify(prerequisite_source).read_text(encoding="utf-8"))
    if prerequisite["status"] != "PREREQUISITE_BLOCKED" or not prerequisite["blockers"]:
        raise ValueError("this frozen campaign disposition requires a demonstrated S76 data block")
    for source in prerequisite["evidence"]:
        verify(source)
    verify(lock["confirmation_future_recipe"])
    evidence: dict[Family, ReplicationSummary | None] = {"C_DIRECT": None, "C_PPS": None}
    costs = {}
    replication_source = None
    if replication["status"] != "NOT_TRIGGERED":
        path = Path(lock["replication_evidence_root"]) / "REPLICATION_EVIDENCE.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        signature = record.pop("artifact_sha256")
        if (
            object_digest(record) != signature
            or record["identity"]["training_lock_sha256"] != lock["artifact_sha256"]
        ):
            raise ValueError("confirmation selection has altered replication evidence")
        replication_source = binding(path)
        for family in evidence:
            summary = record["families"][family]
            if "seed_scores" not in summary:
                continue
            name = record["identity"]["canonical_candidates"][family]
            values, sources = [], []
            for seed in (7, 13, 23):
                for fold in range(3):
                    stage = ("S72" if name.startswith("S72") else "S73") if seed == 7 else "S74"
                    plan = PlannedFit(stage, name, fold, seed)
                    cost_path = (
                        Path(lock["endpoint_root"])
                        / plan.job_id
                        / "deployment_cost/DEPLOYMENT_COST.json"
                    )
                    cost = json.loads(cost_path.read_text(encoding="utf-8"))
                    digest = cost.pop("artifact_sha256")
                    if object_digest(cost) != digest:
                        raise ValueError("deployment timing evidence changed")
                    if cost["identity"]["training_lock_sha256"] != lock["artifact_sha256"]:
                        raise ValueError("deployment timing belongs to another campaign")
                    if cost["identity"]["endpoint"]["job"]["plan"] != asdict(plan):
                        raise ValueError("deployment timing belongs to another seed, fold or arm")
                    verify(cost["identity"]["endpoint"]["inference_weights"])
                    verify(cost["raw_source_reads"])
                    values.append(float(cost["measurement"]["summary"]["end_to_end_ms"]["mean"]))
                    sources.append(binding(cost_path))
            latency = math.fsum(values) / len(values)
            costs[family] = {"sources": sources, "mean_of_all_nine_endpoint_means_ms": latency}
            evidence[family] = ReplicationSummary(
                tuple(summary["seed_scores"]),
                tuple(summary["seed_finite_fractions"]),
                summary["mean_sign_error_increase"],
                summary["mean_crucial_delta"],
                summary["risk17_score"],
                latency,
                signature,
            )
    family = select_confirmation_family(evidence)
    result = {
        "kind": "SOURCE_BOUND_S76_PREREQUISITE_DISPOSITION",
        "status": "PREREQUISITE_BLOCKED",
        "training_lock_sha256": lock["artifact_sha256"],
        "prerequisites": prerequisite_source,
        "blockers": prerequisite["blockers"],
        "replication_evidence": replication_source,
        "selected_stable_family": family,
        "deployment_costs": costs,
        "future_recipe": lock["confirmation_future_recipe"],
        "confirmation_labels_read": False,
        "final_fits": 0,
        "optimizer_updates": 0,
        "scientific_negative": False,
    }
    result["artifact_sha256"] = object_digest(result)
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise ValueError("completed S76 disposition changed")
    else:
        atomic_json(output, result)
    return result


@dataclass(frozen=True)
class ReplicationSummary:
    """Per-seed aggregate development losses, never an average of signed TTC predictions."""

    seed_scores: tuple[float, float, float]  # Fixed order7,13,23.
    seed_finite_fractions: tuple[float, float, float]
    mean_sign_error_increase: float
    mean_crucial_delta: float
    risk17_score: float
    deployment_latency_ms: float
    evidence_sha256: str

    def __post_init__(self) -> None:
        if len(self.seed_scores) != 3 or len(self.seed_finite_fractions) != 3:
            raise ValueError("replication requires all seeds7/13/23 in fixed order")
        values = (
            *self.seed_scores,
            *self.seed_finite_fractions,
            self.mean_sign_error_increase,
            self.mean_crucial_delta,
            self.risk17_score,
            self.deployment_latency_ms,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("incomplete replication evidence cannot select a confirmation family")
        if min(self.seed_scores) < 0 or self.risk17_score < 0 or self.deployment_latency_ms <= 0:
            raise ValueError("invalid loss or measured end-to-end latency")
        if any(not 0 <= value <= 1 for value in self.seed_finite_fractions):
            raise ValueError("invalid replication coverage")
        if len(self.evidence_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.evidence_sha256
        ):
            raise ValueError("replication selection requires hashed development evidence")

    @property
    def mean_loss(self) -> float:
        """Mean seed loss with every optimization replicate retained."""
        return math.fsum(self.seed_scores) / 3

    @property
    def stable(self) -> bool:
        """Exact S76 replication gate, with no additional CI-significance requirement."""
        return (
            all(value == 1 for value in self.seed_finite_fractions)
            and all(score < self.risk17_score for score in self.seed_scores[1:])
            and self.mean_loss <= self.risk17_score - 5
            and self.mean_sign_error_increase <= 0.005
            and self.mean_crucial_delta <= 3
        )


def select_confirmation_family(
    evidence: Mapping[Family, ReplicationSummary | None],
) -> Family | None:
    """Select only stable canonical families on old development, before confirmation access."""
    if set(evidence) != {"C_DIRECT", "C_PPS"}:
        raise ValueError(
            "both fixed canonical families require evidence or explicit unavailability"
        )
    available = [
        (family, result)
        for family, result in evidence.items()
        if result is not None and result.stable
    ]
    if not available:
        return None
    if len(available) == 1:
        return cast(Family, available[0][0])
    direct, pps = evidence["C_DIRECT"], evidence["C_PPS"]
    assert direct is not None and pps is not None
    if abs(direct.mean_loss - pps.mean_loss) > 0.1:
        return "C_DIRECT" if direct.mean_loss < pps.mean_loss else "C_PPS"
    if direct.deployment_latency_ms != pps.deployment_latency_ms:
        return "C_DIRECT" if direct.deployment_latency_ms < pps.deployment_latency_ms else "C_PPS"
    return "C_DIRECT"


def confirmation_prerequisites(
    evidence: Mapping[Family, ReplicationSummary | None],
    *,
    freshness_verified: bool,
    credible_garl_available: bool,
) -> dict[str, str | None]:
    """Return a precise branch prerequisite block without consuming confirmation targets."""
    if type(freshness_verified) is not bool or type(credible_garl_available) is not bool:
        raise ValueError("confirmation prerequisites must be explicitly resolved")
    family = select_confirmation_family(evidence)
    if family is None:
        return {"status": "BLOCKED", "reason": "NO_STABLE_ARCHITECTURE_REPLICATION", "family": None}
    if not freshness_verified:
        return {
            "status": "BLOCKED",
            "reason": "CONFIRMATION_ACQUISITION_OR_ANCESTRY_UNVERIFIED",
            "family": family,
        }
    if not credible_garl_available:
        return {
            "status": "BLOCKED",
            "reason": "CREDIBLE_TRAIN_ONLY_GARL_UNAVAILABLE",
            "family": family,
        }
    return {"status": "ELIGIBLE_FOR_FINAL_FITS", "reason": None, "family": family}
