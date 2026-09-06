"""Freeze genuinely new inner producers with transitive source exclusions before routing."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.architecture_stage_decisions_v11 import verify_stage_decision
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import _owned, verify_fit_endpoint
from e_jepa_ttc.training.nested_architecture_sources_v11 import verify_nested_materialization
from e_jepa_ttc.training.stage72_jobs import FitJob, require_scientific_lock


def verify_nested_job_source(job: FitJob, lock: dict[str, Any], decision_sha: str) -> None:
    """Bind encoder, prior and normalization to the same exact permitted inner source rows."""
    source = json.loads(
        verify(verify_nested_materialization(job.plan, lock, decision_sha)).read_text(
            encoding="utf-8"
        )
    )
    if (
        source["artifact_sha256"] != job.source_manifest_sha256
        or job.training_lock_sha256 != lock["artifact_sha256"]
        or job.stage_authorization_sha256 != decision_sha
        or source["source_role_sha256"] != job.source_role_sha256
        or source["training_view_sha256"] != job.training_view_sha256
        or tuple(source["allowed_train_groups"]) != job.allowed_train_groups
        or tuple(source["excluded_groups"]) != job.excluded_groups
    ):
        raise ValueError("nested endpoint source differs from its locked inner producer")
    norm = json.loads(verify(source["normalization"]).read_text(encoding="utf-8"))
    supervision = json.loads(verify(source["supervision"]).read_text(encoding="utf-8"))
    if (
        object_digest(norm) != job.normalization_sha256
        or supervision["identity_sha256"] != job.supervision_sha256
        or norm["training_view_sha256"] != job.training_view_sha256
        or supervision["training_view_sha256"] != job.training_view_sha256
        or supervision["tokens"] != source["ordered_tokens"]
    ):
        raise ValueError("nested endpoint reused foreign normalization or prior")


def freeze_nested_stage(
    jobs: tuple[FitJob, ...],
    *,
    decision_source: dict[str, Any],
    lock_path: Path,
    lock_sha256: str,
    output: Path,
) -> dict[str, Any]:
    """Freeze every new canonical inner endpoint; pooled outer-OOF cannot stand in for these."""
    lock = {**require_scientific_lock(lock_path, lock_sha256), "artifact_sha256": lock_sha256}
    decision, plans = verify_stage_decision(decision_source, lock, "S75")
    if not plans or tuple(job.plan for job in jobs) != plans:
        raise ValueError("nested barrier requires all prescribed new inner producers")
    expected = Path(lock["stage_barrier_root"]) / "S75_ALL_ENDPOINTS_FROZEN.json"
    if output.absolute() != expected or output.resolve() != expected:
        raise ValueError("nested barrier output differs from the frozen campaign")
    for job in jobs:
        verify_nested_job_source(job, lock, decision["artifact_sha256"])
    if output.exists():
        record = verify_nested_barrier(output, expected_lock=lock_sha256)
        if [row["job_sha256"] for row in record["endpoints"]] != [job.identity for job in jobs]:
            raise ValueError("nested barrier cannot replace earlier fixed producers")
        return record
    root = Path(lock["endpoint_root"])
    record = {
        "kind": "ALL_NEW_NESTED_ARCHITECTURE_ENDPOINTS_FROZEN",
        "stage": "S75",
        "training_lock": binding(lock_path),
        "training_lock_sha256": lock_sha256,
        "stage_authorization": decision_source,
        "endpoint_root": str(root),
        "endpoints": [verify_fit_endpoint(job, root, lock["code_commit"]) for job in jobs],
        "required_endpoint_count": len(jobs),
        "inner_inference_authorized": True,
        "pooled_outer_oof_used_for_router_training": False,
    }
    record["artifact_sha256"] = object_digest(record)
    atomic_json(output, record)
    return record


def verify_nested_barrier(path: Path, *, expected_lock: str) -> dict[str, Any]:
    """Revalidate full new inner endpoints and their prospective transitive source identities."""
    record = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in record.items() if key != "artifact_sha256"}
    if (
        object_digest(unsigned) != record.get("artifact_sha256")
        or record.get("kind") != "ALL_NEW_NESTED_ARCHITECTURE_ENDPOINTS_FROZEN"
        or record.get("stage") != "S75"
        or record.get("training_lock_sha256") != expected_lock
        or record.get("inner_inference_authorized") is not True
        or record.get("pooled_outer_oof_used_for_router_training") is not False
    ):
        raise ValueError("invalid new nested endpoint barrier")
    lock = {
        **require_scientific_lock(verify(record["training_lock"]), expected_lock),
        "artifact_sha256": expected_lock,
    }
    decision, plans = verify_stage_decision(record["stage_authorization"], lock, "S75")
    if (
        not plans
        or record["required_endpoint_count"] != len(plans)
        or [row["job_id"] for row in record["endpoints"]] != [plan.job_id for plan in plans]
        or record["endpoint_root"] != lock["endpoint_root"]
    ):
        raise ValueError("nested barrier omitted or substituted a required producer")
    root = Path(lock["endpoint_root"])
    for row, plan in zip(record["endpoints"], plans, strict=True):
        endpoint = json.loads(
            _owned(row["endpoint"], root / plan.job_id, "FIT_ENDPOINT.json").read_text(
                encoding="utf-8"
            )
        )
        fields = dict(endpoint["job"])
        if fields.pop("plan") != asdict(plan):
            raise ValueError("nested endpoint changed its inner/outer fold")
        job = FitJob(plan=plan, **fields)
        verify_nested_job_source(job, lock, decision["artifact_sha256"])
        if verify_fit_endpoint(job, root, lock["code_commit"]) != row:
            raise ValueError("nested endpoint physical bindings changed")
    return record
