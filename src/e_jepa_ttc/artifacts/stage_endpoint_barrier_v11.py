"""Close complete prescribed architecture stages before their first outer inference."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.training.stage70_update_ledger import FitSpec
from e_jepa_ttc.training.stage72_jobs import FitJob, require_scientific_lock
from e_jepa_ttc.training.stage73_factorial import Availability, PlannedFit, initial_jobs, treatment


def validate_stage_membership(
    stage: str, jobs: tuple[FitJob, ...], availability: Availability
) -> None:
    """Require all 12 S72 endpoints or every independently available S73 endpoint."""
    if stage not in ("S72", "S73"):
        raise ValueError("initial endpoint barrier only accepts S72 or S73")
    expected = {
        plan.job_id: asdict(plan) for plan in initial_jobs(availability) if plan.stage == stage
    }
    supplied = {job.plan.job_id: asdict(job.plan) for job in jobs}
    if len(supplied) != len(jobs) or supplied != expected:
        raise ValueError("all prescribed stage endpoints are required before outer inference")


def _owned(record: dict[str, Any], directory: Path, expected_name: str) -> Path:
    path = Path(record["path"])
    expected = directory.absolute() / expected_name
    if path.resolve(strict=True) != expected or path.is_symlink():
        raise ValueError("endpoint payload escaped its prescribed owned path")
    return verify(record)


def _verify_job_source(job: FitJob, lock: dict[str, Any]) -> None:
    """Recheck the prospective fold-to-source assignment without opening outer data."""
    if lock.get("initial_source_manifest_by_job", {}).get(job.plan.job_id) != (
        job.source_manifest_sha256
    ):
        raise ValueError("job source differs from the prospective fold assignment")
    sources = lock.get("authorized_source_manifests", {})
    if job.source_manifest_sha256 not in sources:
        raise ValueError("job source is absent from the scientific lock")
    source = json.loads(verify(sources[job.source_manifest_sha256]).read_text(encoding="utf-8"))
    signature = source.pop("artifact_sha256", None)
    if signature != job.source_manifest_sha256 or object_digest(source) != signature:
        raise ValueError("job source manifest signature differs")
    for field, expected in {
        "outer_fold": job.plan.outer_fold,
        "training_view_sha256": job.training_view_sha256,
        "allowed_train_groups": list(job.allowed_train_groups),
        "excluded_groups": list(job.excluded_groups),
        "source_role_sha256": job.source_role_sha256,
        "include_expansion": treatment(job.plan.treatment_name).expanded,
    }.items():
        if source.get(field) != expected:
            raise ValueError(f"locked job source field differs: {field}")
    if job.source_role_sha256 != lock.get("source_role_sha256"):
        raise ValueError("job roles differ from the scientific lock")
    normalization = json.loads(verify(source["normalization"]).read_text(encoding="utf-8"))
    supervision = json.loads(verify(source["supervision"]).read_text(encoding="utf-8"))
    if (
        object_digest(normalization) != job.normalization_sha256
        or normalization.get("training_view_sha256") != job.training_view_sha256
        or supervision.get("identity_sha256") != job.supervision_sha256
        or supervision.get("training_view_sha256") != job.training_view_sha256
        or supervision.get("tokens") != source.get("ordered_tokens")
    ):
        raise ValueError("job normalization or supervision differs from its locked TRAIN view")


def verify_fit_endpoint(job: FitJob, root: Path, code_commit: str) -> dict[str, Any]:
    """Verify full checkpoint metadata and exact inference weights against the fixed job."""
    directory = root.resolve() / job.plan.job_id
    if directory.parent != root.resolve() or directory.resolve() != directory:
        raise ValueError("architecture output escaped its owned stage root")
    path = directory / "FIT_ENDPOINT.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    signature = record.pop("artifact_sha256")
    recipe = treatment(job.plan.treatment_name)
    if (
        object_digest(record) != signature
        or record.get("kind") != "STANDALONE_ARCHITECTURE_ENDPOINT"
        or record.get("job_sha256") != job.identity
        or object_digest(record.get("job")) != job.identity
        or record.get("training_code_commit") != code_commit
        or record.get("scientific_endpoint") is not True
        or record.get("outer_evaluated") is not False
        or record.get("completed_supervised_updates") != recipe.supervised_updates
        or record.get("completed_pretext_updates") != recipe.pretext_updates
    ):
        raise ValueError("architecture endpoint is incomplete or differs from the locked recipe")
    checkpoint = _owned(
        record["checkpoint"],
        directory / "supervised",
        f"checkpoint_{recipe.supervised_updates:06d}.pt",
    )
    weights_path = _owned(record["inference_weights"], directory, "inference_weights.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = FitSpec(
        job.plan.job_id + "_supervised",
        job.plan.stage,
        recipe.architecture,
        job.plan.outer_fold,
        job.plan.seed,
        recipe.supervised_updates,
        job.identity,
        job.training_view_sha256,
    )
    if payload.get("spec") != asdict(expected) or payload.get("update") != expected.endpoint:
        raise ValueError("supervised full-state checkpoint does not prove its fixed endpoint")
    if not {"optimizer", "rng", "runtime", "history", "identity"} <= payload.keys():
        raise ValueError("full-state checkpoint lacks exact-resume evidence")
    history = payload["history"]
    if len(history) != expected.endpoint or any(
        row.get("update") != index for index, row in enumerate(history, 1)
    ):
        raise ValueError("checkpoint optimizer history does not cover every prescribed update")
    runtime = payload["runtime"]
    if any(
        runtime.get(key) != value
        for key, value in {
            "warmup": recipe.supervised_warmup,
            "effective_batch": 16,
            "precision": "float32_TF32_off",
            "deterministic": True,
            "checkpoint_interval": 250,
            "cublas_workspace_config": ":4096:8",
        }.items()
    ) or runtime.get("microbatch") not in (1, 2):
        raise ValueError("checkpoint runtime differs from the registered scientific execution")
    if payload["identity"] != object_digest({"spec": asdict(expected), "runtime": runtime}):
        raise ValueError("checkpoint identity differs from its full fixed-fit contract")
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or set(weights) != set(payload["model"]):
        raise ValueError("inference weight schema differs from the verified endpoint")
    for key, value in weights.items():
        state = payload["model"][key]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != state.dtype
            or not torch.equal(value, state)
            or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
        ):
            raise ValueError("inference weights differ from the verified full checkpoint")
    del payload, weights
    if recipe.pretext_updates:
        if record.get("pretext_checkpoint") is None:
            raise ValueError("pretext-enabled endpoint omitted its parent encoder checkpoint")
        pretext_path = _owned(
            record["pretext_checkpoint"], directory / "pretext", "checkpoint_003000.pt"
        )
        pretext = torch.load(pretext_path, map_location="cpu", weights_only=False)
        spec = pretext.get("spec", {})
        if (
            pretext.get("update") != 3000
            or spec.get("endpoint") != 3000
            or spec.get("job_id") != job.plan.job_id + "_pretext"
            or spec.get("recipe_sha256") != job.identity
            or spec.get("seed") != job.plan.seed
            or spec.get("arm") != "CORRESPONDENCE"
            or spec.get("technical") is not False
        ):
            raise ValueError("pretext parent is partial or belongs to a different source job")
    elif record.get("pretext_checkpoint") is not None:
        raise ValueError("scratch endpoint cannot inherit an undeclared pretext parent")
    return {
        "job_id": job.plan.job_id,
        "job_sha256": job.identity,
        "endpoint": binding(path),
        "checkpoint": record["checkpoint"],
        "inference_weights": record["inference_weights"],
        "pretext_checkpoint": record.get("pretext_checkpoint"),
    }


def freeze_initial_stage(
    stage: str,
    jobs: tuple[FitJob, ...],
    availability: Availability,
    *,
    lock_path: Path,
    lock_sha256: str,
    endpoint_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Publish a one-way barrier only after every prescribed endpoint verifies."""
    validate_stage_membership(stage, jobs, availability)
    lock = require_scientific_lock(lock_path, lock_sha256)
    if lock.get("availability") != asdict(availability) or any(
        job.training_lock_sha256 != lock_sha256 for job in jobs
    ):
        raise ValueError("stage availability or job lock identity changed")
    if lock.get("endpoint_root") != str(endpoint_root.resolve()):
        raise ValueError("stage output root differs from the scientific lock")
    for job in jobs:
        _verify_job_source(job, lock)
    if output.exists():
        record = verify_initial_barrier(output, expected_stage=stage, expected_lock=lock_sha256)
        if {row["job_sha256"] for row in record["endpoints"]} != {job.identity for job in jobs}:
            raise ValueError("completed stage barrier belongs to different resolved source jobs")
        return record
    endpoints = [verify_fit_endpoint(job, endpoint_root, lock["code_commit"]) for job in jobs]
    record = {
        "kind": "ALL_INITIAL_STAGE_ENDPOINTS_FROZEN",
        "stage": stage,
        "training_lock_sha256": lock_sha256,
        "training_lock": binding(lock_path),
        "endpoint_root": str(endpoint_root.resolve()),
        "availability": asdict(availability),
        "endpoints": endpoints,
        "required_endpoint_count": len(jobs),
        "outer_inference_authorized": True,
        "selection_using_outer_results": False,
    }
    result = {**record, "artifact_sha256": object_digest(record)}
    atomic_json(output, result)
    return result


def verify_initial_barrier(
    path: Path, *, expected_stage: str, expected_lock: str
) -> dict[str, Any]:
    """Revalidate every immutable endpoint before accessing outer inference inputs."""
    record = json.loads(path.read_text(encoding="utf-8"))
    signature = record.pop("artifact_sha256")
    if (
        object_digest(record) != signature
        or record.get("stage") != expected_stage
        or record.get("training_lock_sha256") != expected_lock
        or record.get("kind") != "ALL_INITIAL_STAGE_ENDPOINTS_FROZEN"
        or record.get("outer_inference_authorized") is not True
        or record.get("selection_using_outer_results") is not False
    ):
        raise ValueError("invalid initial-stage outer-inference barrier")
    availability = Availability(**record["availability"])
    expected = {job.job_id for job in initial_jobs(availability) if job.stage == expected_stage}
    rows = record["endpoints"]
    if (
        expected_stage not in ("S72", "S73")
        or not expected
        or len(rows) != len(expected)
        or record["required_endpoint_count"] != len(expected)
        or {row["job_id"] for row in rows} != expected
    ):
        raise ValueError("stage barrier omits or duplicates a prescribed endpoint")
    lock = require_scientific_lock(verify(record["training_lock"]), expected_lock)
    if record["availability"] != lock.get("availability"):
        raise ValueError("stage availability differs from the scientific lock")
    if record.get("endpoint_root") != lock.get("endpoint_root"):
        raise ValueError("stage output root differs from its scientific lock")
    root = Path(record["endpoint_root"])
    for row in rows:
        endpoint_path = _owned(row["endpoint"], root / row["job_id"], "FIT_ENDPOINT.json")
        endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
        fields = dict(endpoint["job"])
        plan = PlannedFit(**fields.pop("plan"))
        job = FitJob(plan=plan, **fields)
        if job.plan.job_id != row["job_id"] or job.training_lock_sha256 != expected_lock:
            raise ValueError("barrier substituted another fold or source job")
        _verify_job_source(job, lock)
        checked = verify_fit_endpoint(job, root, lock["code_commit"])
        if checked != row:
            raise ValueError("barrier endpoint bindings were substituted")
    return {**record, "artifact_sha256": signature}
