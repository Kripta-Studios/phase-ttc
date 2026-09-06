"""Freeze all prospectively triggered new-seed endpoints before replication outer inference."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import _verify_job_source, verify_fit_endpoint
from e_jepa_ttc.evaluation.stage70_76 import Evidence
from e_jepa_ttc.training.stage72_jobs import FitJob, require_scientific_lock
from e_jepa_ttc.training.stage73_factorial import (
    Availability,
    Family,
    PlannedFit,
    canonical_candidates,
    replication_jobs,
)


def replication_decision_plans(
    record: dict[str, Any], lock: dict[str, Any]
) -> tuple[PlannedFit, ...]:
    """Recompute exact practical/complementarity decisions; never choose a successful seed."""
    unsigned = {key: value for key, value in record.items() if key != "artifact_sha256"}
    availability = Availability(**lock["availability"])
    if (
        object_digest(unsigned) != record.get("artifact_sha256")
        or record.get("kind") != "S74_ARCHITECTURE_REPLICATION_DECISION"
        or record.get("stage") != "S74"
        or record.get("training_lock_sha256") != lock["artifact_sha256"]
        or record.get("canonical_candidates") != canonical_candidates(availability)
        or set(record.get("evidence", {})) != {"C_DIRECT", "C_PPS"}
    ):
        raise ValueError("replication decision changed its lock, evidence or canonical identities")
    evidence = cast(
        dict[Family, Evidence | None],
        {
            name: None if value is None else Evidence(**value)
            for name, value in record["evidence"].items()
        },
    )
    plans = replication_jobs(availability, evidence)
    if record.get("plans") != [asdict(plan) for plan in plans] or record.get(
        "authorized_job_ids"
    ) != [plan.job_id for plan in plans]:
        raise ValueError("replication decision omitted a prescribed seed/fold or added a control")
    return plans


def _replication_source(job: FitJob, lock: dict[str, Any]) -> None:
    """New seeds reuse precisely their canonical seed7 TRAIN pool, normalizer and prior."""
    stage = "S72" if job.plan.treatment_name.startswith("S72") else "S73"
    initial = PlannedFit(stage, job.plan.treatment_name, job.plan.outer_fold, 7)
    _verify_job_source(replace(job, plan=initial, stage_authorization_sha256=None), lock)


def freeze_replication_stage(
    jobs: tuple[FitJob, ...],
    *,
    decision_path: Path,
    lock_path: Path,
    lock_sha256: str,
    endpoint_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Require every new encoder/head seed endpoint authorized by the frozen S74 decision."""
    lock = require_scientific_lock(lock_path, lock_sha256)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    plans = replication_decision_plans(decision, {**lock, "artifact_sha256": lock_sha256})
    if not plans or tuple(job.plan for job in jobs) != plans:
        raise ValueError("replication barrier requires every prospectively authorized endpoint")
    if lock["endpoint_root"] != str(endpoint_root.resolve()):
        raise ValueError("replication output root changed")
    for job in jobs:
        if (
            job.training_lock_sha256 != lock_sha256
            or job.stage_authorization_sha256 != decision["artifact_sha256"]
        ):
            raise ValueError("replication job belongs to another lock or gate decision")
        _replication_source(job, lock)
    if output.exists():
        existing = verify_replication_barrier(output, expected_lock=lock_sha256)
        if existing["stage_authorization"] != binding(decision_path) or {
            row["job_sha256"] for row in existing["endpoints"]
        } != {job.identity for job in jobs}:
            raise ValueError("completed replication barrier belongs to a different gate decision")
        return existing
    endpoints = [verify_fit_endpoint(job, endpoint_root, lock["code_commit"]) for job in jobs]
    record = {
        "kind": "ALL_REPLICATION_ENDPOINTS_FROZEN",
        "stage": "S74",
        "training_lock_sha256": lock_sha256,
        "training_lock": binding(lock_path),
        "stage_authorization": binding(decision_path),
        "endpoint_root": str(endpoint_root.resolve()),
        "endpoints": endpoints,
        "required_endpoint_count": len(jobs),
        "outer_inference_authorized": True,
        "best_seed_selection": False,
    }
    record["artifact_sha256"] = object_digest(record)
    atomic_json(output, record)
    return record


def verify_replication_barrier(path: Path, *, expected_lock: str) -> dict[str, Any]:
    """Recheck complete all-seed endpoint coverage and unchanged canonical source ancestry."""
    record = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in record.items() if key != "artifact_sha256"}
    if (
        object_digest(unsigned) != record.get("artifact_sha256")
        or record.get("kind") != "ALL_REPLICATION_ENDPOINTS_FROZEN"
        or record.get("stage") != "S74"
        or record.get("training_lock_sha256") != expected_lock
        or record.get("outer_inference_authorized") is not True
        or record.get("best_seed_selection") is not False
    ):
        raise ValueError("invalid full replication endpoint barrier")
    lock = require_scientific_lock(verify(record["training_lock"]), expected_lock)
    decision = json.loads(verify(record["stage_authorization"]).read_text(encoding="utf-8"))
    plans = replication_decision_plans(decision, {**lock, "artifact_sha256": expected_lock})
    if (
        not plans
        or record["required_endpoint_count"] != len(plans)
        or [row["job_id"] for row in record["endpoints"]] != [plan.job_id for plan in plans]
        or record["endpoint_root"] != lock["endpoint_root"]
    ):
        raise ValueError(
            "replication barrier lost a prescribed seed/fold or changed its output root"
        )
    root = Path(record["endpoint_root"])
    for row, plan in zip(record["endpoints"], plans, strict=True):
        endpoint_path = root / plan.job_id / "FIT_ENDPOINT.json"
        if Path(row["endpoint"]["path"]).resolve() != endpoint_path.absolute():
            raise ValueError("replication endpoint escaped its owned job path")
        endpoint = json.loads(verify(row["endpoint"]).read_text(encoding="utf-8"))
        fields = dict(endpoint["job"])
        if fields.pop("plan") != asdict(plan):
            raise ValueError("replication barrier substituted another seed or architecture")
        job = FitJob(plan=plan, **fields)
        if (
            job.stage_authorization_sha256 != decision["artifact_sha256"]
            or job.training_lock_sha256 != expected_lock
        ):
            raise ValueError("replication endpoint gate ancestry differs")
        _replication_source(job, lock)
        if verify_fit_endpoint(job, root, lock["code_commit"]) != row:
            raise ValueError("replication full endpoint bindings changed")
    return record
