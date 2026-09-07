"""Run the fixed new inner producers before assembling generalized full-cost routing tables."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.architecture_stage_decisions_v11 import verify_stage_decision
from e_jepa_ttc.artifacts.nested_architecture_barrier_v11 import freeze_nested_stage
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.evaluation.job_deployment_cost_v11 import observe_job_deployment
from e_jepa_ttc.evaluation.nested_treatment_export_v11 import export_nested_treatment
from e_jepa_ttc.evaluation.pretext_diagnostics_v11 import diagnose_pretext_checkpoint
from e_jepa_ttc.evaluation.standalone_outer_v11 import infer_initial_endpoint
from e_jepa_ttc.evaluation.supervised_checkpoint_diagnostics_v11 import diagnose_supervised_series
from e_jepa_ttc.training.correspondence_pretext_v11 import PretextTextures
from e_jepa_ttc.training.nested_architecture_sources_v11 import (
    materialize_nested_sources,
    prospective_nested_recipe,
)
from e_jepa_ttc.training.resolved_architecture_sources_v11 import ResolvedArchitectureSources
from e_jepa_ttc.training.stage72_73_runtime_v11 import InitialArchitectureRuntime
from e_jepa_ttc.training.stage72_jobs import FitJob, execute_architecture_job
from e_jepa_ttc.training.stage73_factorial import PlannedFit, treatment


def run_new_nested_producers(
    runtime: InitialArchitectureRuntime, decision_source: dict[str, Any]
) -> dict[str, Any]:
    """Finish every gated inner encoder/head/pretext endpoint and its held-inner predictions."""
    lock = {**runtime.lock, "artifact_sha256": runtime.lock_sha256}
    decision, plans = verify_stage_decision(decision_source, lock, "S75")
    queue = runtime.root / "NESTED_JOB_QUEUE.json"
    completed: list[str] = []

    def progress(operation: str, active: str | None) -> None:
        atomic_json(
            queue,
            {
                "training_lock_sha256": runtime.lock_sha256,
                "stage_authorization": decision_source,
                "operation": operation,
                "active_job": active,
                "prescribed_jobs": [plan.job_id for plan in plans],
                "completed_in_operation": list(completed),
                "optimizer_totals": runtime.ledger.totals(),
                "checkpoint_and_rng_authority": str(runtime.ledger.path),
                "scientific_campaign_complete": False,
            },
        )

    if not plans:
        progress("NOT_TRIGGERED", None)
        return {"status": "NOT_TRIGGERED", "stage_authorization": decision_source, "inference": {}}
    for field in ("nested_source_root", "inner_inference_root"):
        path = Path(lock[field])
        if path.resolve() != path or not path.is_relative_to(runtime.root):
            raise ValueError("nested output escaped the frozen campaign root")

    def resolve(plan: PlannedFit) -> tuple[FitJob, ResolvedArchitectureSources]:
        reader = runtime.expanded if treatment(plan.treatment_name).expanded else runtime.original
        if reader is None:
            raise ValueError("nested D producer lacks its authorized count source")
        loaded = materialize_nested_sources(
            plan,
            lock=lock,
            decision_source=decision_source,
            reader=reader,
            roles=runtime.roles,
            owner=runtime.owner,
        )
        job = FitJob(
            plan,
            runtime.lock_sha256,
            loaded.view.identity,
            object_digest(asdict(loaded.normalization)),
            loaded.supervision.identity_sha256,
            lock["source_role_sha256"],
            tuple(loaded.source["allowed_train_groups"]),
            tuple(loaded.source["excluded_groups"]),
            loaded.source_sha256,
            decision["artifact_sha256"],
        )
        return job, loaded

    jobs = []
    for plan in plans:
        runtime._reserve_resources(plan.job_id)
        progress("MATERIALIZE_AND_FIT_NEW_INNER", plan.job_id)
        job, loaded = resolve(plan)
        execute_architecture_job(
            job,
            lock_path=runtime.lock_path,
            view=loaded.view,
            normalization=loaded.normalization,
            supervision=loaded.supervision,
            ledger=runtime.ledger,
            owner=runtime.owner,
            output_root=runtime.paths["endpoint_root"],
            stage_authorization=decision_source,
        )
        if treatment(plan.treatment_name).pretext_updates:
            root = runtime.paths["endpoint_root"] / plan.job_id
            pool = PretextTextures(
                loaded.view, loaded.normalization, root / "UNUSED_DIAGNOSTIC_POOL.jsonl"
            )
            for update in (0, 3000):
                diagnose_pretext_checkpoint(
                    binding(root / "pretext" / f"checkpoint_{update:06d}.pt"),
                    pool,
                    seed=7,
                    capacity=plan.capacity,
                    output=root / f"pretext_diagnostics_{update:06d}",
                    check=runtime.owner.check,
                )
        diagnose_supervised_series(
            runtime.paths["endpoint_root"] / plan.job_id,
            loaded.view,
            loaded.normalization,
            loaded.supervision,
            recipe_sha256=job.identity,
            endpoint=treatment(plan.treatment_name).supervised_updates,
            capacity=plan.capacity,
            check=runtime.owner.check,
        )
        observe_job_deployment(job, loaded, runtime.lock, runtime.owner)
        jobs.append(job)
        completed.append(plan.job_id)
        progress("MATERIALIZE_AND_FIT_NEW_INNER", None)
    barrier = runtime.paths["stage_barrier_root"] / "S75_ALL_ENDPOINTS_FROZEN.json"
    freeze_nested_stage(
        tuple(jobs),
        decision_source=decision_source,
        lock_path=runtime.lock_path,
        lock_sha256=runtime.lock_sha256,
        output=barrier,
    )
    completed.clear()
    inference = {}
    for plan in plans:
        runtime._reserve_resources("inner_inference")
        progress("INNER_INFERENCE", plan.job_id)
        job, loaded = resolve(plan)
        _, recipe = prospective_nested_recipe(plan, lock)
        output = Path(lock["inner_inference_root"]) / plan.job_id
        infer_initial_endpoint(
            job,
            barrier_path=barrier,
            lock_path=runtime.lock_path,
            reader=runtime.original,
            tokens=tuple(recipe["inner_prediction_tokens"]),
            normalization=loaded.normalization,
            output=output,
            owner=runtime.owner,
        )
        inference[plan.job_id] = binding(output / "INFERENCE_COMPLETE.json")
        completed.append(plan.job_id)
        progress("INNER_INFERENCE", None)
    exports = {}
    for name, outer, key in dict.fromkeys(
        (plan.treatment_name, plan.outer_fold, plan.candidate_key) for plan in plans
    ):
        runtime._reserve_resources("nested_export")
        export_nested_treatment(
            name,
            outer,
            barrier_path=barrier,
            lock_path=runtime.lock_path,
            lock_sha256=runtime.lock_sha256,
            owner=runtime.owner,
        )
        path = Path(lock["nested_export_root"]) / key / f"outer{outer}" / "NESTED_EXPORT.json"
        exports[f"{key}_outer{outer}"] = binding(path)
    progress("NEW_INNER_ENDPOINTS_AND_EXPORTS_COMPLETE_ROUTER_PENDING", None)
    return {
        "status": "NEW_INNER_PRODUCERS_COMPLETE",
        "stage_authorization": decision_source,
        "barrier": binding(barrier),
        "inference": inference,
        "exports": exports,
        "router_fit_complete": False,
    }
