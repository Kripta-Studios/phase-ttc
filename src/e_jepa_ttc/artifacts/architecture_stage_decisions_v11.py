"""Publish source-bound conditional stage decisions without changing prospective recipes."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import _owned
from e_jepa_ttc.evaluation.initial_campaign_evidence_v11 import verify_initial_evidence
from e_jepa_ttc.evaluation.stage70_76 import Evidence
from e_jepa_ttc.training.stage73_factorial import (
    Availability,
    Family,
    PlannedFit,
    canonical_candidates,
    nested_jobs,
    replication_jobs,
    stage_update_total,
)


def decision_payload(stage: str, evidence: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    """Derive exact jobs from verified canonical evidence; this pure helper grants no access."""
    availability = Availability(**lock["availability"])
    if stage == "S74":
        values = cast(
            dict[Family, Evidence | None],
            {
                family: None if value is None else Evidence(**value)
                for family, value in evidence["canonical_evidence"].items()
            },
        )
        plans = replication_jobs(availability, values)
        kind = "S74_ARCHITECTURE_REPLICATION_DECISION"
    elif stage == "S75":
        plans = nested_jobs(
            availability,
            joint_oracle_gain=evidence["canonical_joint_oracle_gain"],
            technically_valid=evidence["canonical_technically_valid"],
        )
        kind = "S75_NEW_EXPERT_NESTED_DECISION"
    else:
        raise ValueError("only S74 and S75 derive from the initial canonical comparison")
    return {
        "kind": kind,
        "stage": stage,
        "training_lock_sha256": lock["artifact_sha256"],
        "canonical_candidates": canonical_candidates(availability),
        "evidence": evidence["canonical_evidence"],
        "canonical_technically_valid": evidence["canonical_technically_valid"],
        "canonical_joint_oracle_gain": evidence["canonical_joint_oracle_gain"],
        "plans": [asdict(plan) for plan in plans],
        "authorized_job_ids": [plan.job_id for plan in plans],
        "prescribed_optimizer_updates": stage_update_total(plans),
        "status": "AUTHORIZED" if plans else "NOT_TRIGGERED",
        "independent_confirmation": False,
    }


def publish_stage_decision(stage: str, lock: dict[str, Any]) -> dict[str, Any]:
    """Verify all initial endpoints and rows, then atomically freeze an exact conditional plan."""
    evidence_path = Path(lock["initial_evidence_root"]) / "INITIAL_EVIDENCE.json"
    evidence = verify_initial_evidence(evidence_path, lock)
    record = {**decision_payload(stage, evidence, lock), "initial_evidence": binding(evidence_path)}
    record["artifact_sha256"] = object_digest(record)
    path = Path(lock["stage_decision_root"]) / f"{stage}_DECISION.json"
    if path.resolve() != path.absolute():
        raise ValueError("conditional decision path was redirected")
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError("published conditional decision cannot be replaced")
    else:
        atomic_json(path, record)
    return record


def verify_stage_decision(
    source: dict[str, Any], lock: dict[str, Any], stage: str
) -> tuple[dict[str, Any], tuple[PlannedFit, ...]]:
    """Require physically bound initial evidence before reserving any conditional fit."""
    path = _owned(source, Path(lock["stage_decision_root"]), f"{stage}_DECISION.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    evidence_path = _owned(
        record["initial_evidence"], Path(lock["initial_evidence_root"]), "INITIAL_EVIDENCE.json"
    )
    evidence = verify_initial_evidence(evidence_path, lock)
    expected = {
        **decision_payload(stage, evidence, lock),
        "initial_evidence": binding(evidence_path),
    }
    expected["artifact_sha256"] = object_digest(expected)
    if record != expected:
        raise ValueError("conditional decision differs from its complete canonical source evidence")
    return record, tuple(PlannedFit(**values) for values in record["plans"])
