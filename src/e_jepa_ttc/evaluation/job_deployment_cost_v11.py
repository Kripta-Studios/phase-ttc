"""Immutable source-bound deployment measurements for completed scientific endpoints."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import _owned, verify_fit_endpoint
from e_jepa_ttc.data.raw_deployment_v11 import RawDeploymentInputs
from e_jepa_ttc.data.raw_event_binding import ReadOnlyTrainAccess
from e_jepa_ttc.evaluation.deployment_cost_v11 import benchmark_raw_deployment
from e_jepa_ttc.models.projective_phase_network import ProjectivePhaseNetwork
from e_jepa_ttc.training.eap_highres_jepa import _capture_rng_state, _restore_rng_state
from e_jepa_ttc.training.resolved_architecture_sources_v11 import ResolvedArchitectureSources
from e_jepa_ttc.training.stage72_jobs import FitJob
from e_jepa_ttc.training.stage73_factorial import treatment


def observe_job_deployment(
    job: FitJob,
    loaded: ResolvedArchitectureSources,
    lock: dict[str, Any],
    owner: ResourceOwner,
) -> dict[str, Any]:
    """Measure first complete attempt; preserve interruptions without picking a fast rerun."""
    if (
        loaded.view.identity != job.training_view_sha256
        or object_digest(asdict(loaded.normalization)) != job.normalization_sha256
        or loaded.source_sha256 != job.source_manifest_sha256
    ):
        raise ValueError("deployment observation changed its fitted TRAIN source")
    endpoint_root = Path(lock["endpoint_root"])
    endpoint = verify_fit_endpoint(job, endpoint_root, lock["code_commit"])
    recipe = json.loads(verify(lock["deployment_cost_recipe"]).read_text(encoding="utf-8"))
    if any(
        recipe.get(key) != value
        for key, value in {
            "rows": 16,
            "warmup_rows": 4,
            "complete_measured_passes": 2,
            "batch": 1,
            "roi_pixels": [128, 128],
            "optimizer_updates": 0,
            "labels_used": False,
        }.items()
    ):
        raise ValueError("deployment timing recipe differs from its registered implementation")
    raw_sources = lock["raw_deployment"]
    root = endpoint_root / job.plan.job_id / "deployment_cost"
    if root.resolve() != root.absolute():
        raise ValueError("deployment output was redirected")
    identity = {
        "job_sha256": job.identity,
        "endpoint": endpoint,
        "raw_sources": raw_sources,
        "recipe": lock["deployment_cost_recipe"],
        "training_lock_sha256": job.training_lock_sha256,
    }
    root.mkdir(parents=True, exist_ok=True)
    intent = root / "DEPLOYMENT_INTENT.json"
    if intent.exists():
        if json.loads(intent.read_text(encoding="utf-8")) != identity:
            raise ValueError("deployment recovery changed its fixed endpoint or source recipe")
    elif any(root.iterdir()):
        raise ValueError("deployment output lacks its prospective identity")
    else:
        atomic_json(intent, identity)
    final = root / "DEPLOYMENT_COST.json"
    complete = [final] if final.exists() else list(root.glob("attempt_*/DEPLOYMENT_COST.json"))
    if len(complete) > 1:
        raise ValueError("multiple completed deployment attempts cannot be selected")
    if complete:
        if complete[0].resolve() != complete[0].absolute():
            raise ValueError("deployment result was redirected")
        record = json.loads(complete[0].read_text(encoding="utf-8"))
        unsigned = {key: value for key, value in record.items() if key != "artifact_sha256"}
        if (
            object_digest(unsigned) != record.get("artifact_sha256")
            or record.get("identity") != identity
        ):
            raise ValueError("completed deployment observation changed")
        trace_root = Path(record["raw_source_reads"]["path"]).parent
        if trace_root.parent != root or not trace_root.name.startswith("attempt_"):
            raise ValueError("deployment source trace escaped its owned attempt")
        _owned(record["raw_source_reads"], trace_root, "RAW_SOURCE_READS.jsonl")
        if not final.exists():
            atomic_json(final, record)
        return record
    attempt = root / f"attempt_{len(list(root.glob('attempt_*'))):04d}"
    attempt.mkdir()
    atomic_json(attempt / "STARTED.json", identity)
    access = ReadOnlyTrainAccess(
        Path(raw_sources["raw_train_root"]),
        Path(raw_sources["train_metadata"]),
        journal_path=attempt / "RAW_SOURCE_READS.jsonl",
        resource_check=owner.check,
    )
    raw = RawDeploymentInputs(
        raw_sources["bindings"], raw_sources["clock_charter"], access, loaded.view
    )
    generator = torch.Generator().manual_seed(720076)
    state = _capture_rng_state(generator)
    model = None
    try:
        model = ProjectivePhaseNetwork(checkpoint_sweep=False, capacity=job.plan.capacity).cuda()
        model.load_state_dict(
            torch.load(verify(endpoint["inference_weights"]), map_location="cpu", weights_only=True)
        )
        result = benchmark_raw_deployment(
            model,
            treatment(job.plan.treatment_name).architecture,
            job.plan.seed,
            raw,
            loaded.normalization,
            check=owner.check,
        )
        raw.close()
        record = {
            "kind": "SOURCE_BOUND_FIXED_ENDPOINT_DEPLOYMENT_COST",
            "identity": identity,
            "measurement": result,
            "raw_source_reads": binding(attempt / "RAW_SOURCE_READS.jsonl"),
            "incomplete_prior_attempts": len(list(root.glob("attempt_*"))) - 1,
            "partial_attempt_timing_note": (
                "interrupted source reads retained; incomplete timing totals are not fabricated"
            ),
            "optimizer_updates": 0,
        }
        record["artifact_sha256"] = object_digest(record)
        atomic_json(attempt / "DEPLOYMENT_COST.json", record)
        atomic_json(final, record)
        return record
    finally:
        raw.close()
        _restore_rng_state(state, generator)
        del model
        torch.cuda.empty_cache()
