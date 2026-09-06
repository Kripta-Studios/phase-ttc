"""Execute every triggered canonical replication with new encoder and head seeds."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.architecture_stage_decisions_v11 import verify_stage_decision
from e_jepa_ttc.artifacts.replication_barrier_v11 import freeze_replication_stage
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding
from e_jepa_ttc.evaluation.initial_treatment_export_v11 import export_initial_treatment
from e_jepa_ttc.evaluation.job_deployment_cost_v11 import observe_job_deployment
from e_jepa_ttc.evaluation.pretext_diagnostics_v11 import diagnose_pretext_checkpoint
from e_jepa_ttc.evaluation.replication_evidence_v11 import build_replication_evidence
from e_jepa_ttc.evaluation.standalone_outer_v11 import infer_initial_endpoint
from e_jepa_ttc.evaluation.supervised_checkpoint_diagnostics_v11 import diagnose_supervised_series
from e_jepa_ttc.training.correspondence_pretext_v11 import PretextTextures
from e_jepa_ttc.training.resolved_architecture_sources_v11 import ResolvedArchitectureSources
from e_jepa_ttc.training.stage72_73_runtime_v11 import InitialArchitectureRuntime
from e_jepa_ttc.training.stage72_jobs import FitJob, execute_architecture_job
from e_jepa_ttc.training.stage73_factorial import PlannedFit, treatment


def run_replication(
    runtime: InitialArchitectureRuntime, decision_source: dict[str, Any]
) -> dict[str, Any]:
    """Finish all prescribed seeds before any replication outer read, retaining every seed."""
    lock = {**runtime.lock, "artifact_sha256": runtime.lock_sha256}
    decision, plans = verify_stage_decision(decision_source, lock, "S74")
    queue = runtime.root / "REPLICATION_JOB_QUEUE.json"
    completed: list[str] = []

    def progress(operation: str, active: str | None) -> None:
        atomic_json(
            queue,
            {
                "stage_authorization": decision_source,
                "training_lock_sha256": runtime.lock_sha256,
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
        return {"status": "NOT_TRIGGERED", "stage_authorization": decision_source, "exports": {}}
    export_root = Path(lock["replication_export_root"])
    if export_root.resolve() != export_root or not export_root.is_relative_to(runtime.root):
        raise ValueError("replication exports escaped the frozen campaign root")

    def resolve(plan: PlannedFit) -> tuple[FitJob, ResolvedArchitectureSources]:
        initial_stage = "S72" if plan.treatment_name.startswith("S72") else "S73"
        original = PlannedFit(initial_stage, plan.treatment_name, plan.outer_fold, 7)
        job, loaded = runtime.resolve(original)
        return replace(
            job, plan=plan, stage_authorization_sha256=decision["artifact_sha256"]
        ), loaded

    jobs = []
    for plan in plans:
        runtime._reserve_resources(plan.job_id)
        progress("FITTING", plan.job_id)
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
                    seed=plan.seed,
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
            check=runtime.owner.check,
        )
        observe_job_deployment(job, loaded, runtime.lock, runtime.owner)
        jobs.append(job)
        completed.append(plan.job_id)
        progress("FITTING", None)
    barrier_path = runtime.paths["stage_barrier_root"] / "S74_ALL_ENDPOINTS_FROZEN.json"
    freeze_replication_stage(
        tuple(jobs),
        decision_path=Path(decision_source["path"]),
        lock_path=runtime.lock_path,
        lock_sha256=runtime.lock_sha256,
        endpoint_root=runtime.paths["endpoint_root"],
        output=barrier_path,
    )
    completed.clear()
    for plan in plans:
        runtime._reserve_resources("outer_inference")
        progress("OUTER_INFERENCE", plan.job_id)
        job, loaded = resolve(plan)
        infer_initial_endpoint(
            job,
            barrier_path=barrier_path,
            lock_path=runtime.lock_path,
            reader=runtime.original,
            tokens=tuple(lock["outer_tokens_by_fold"][str(plan.outer_fold)]),
            normalization=loaded.normalization,
            output=runtime.paths["outer_inference_root"] / plan.job_id,
            owner=runtime.owner,
        )
        completed.append(plan.job_id)
        progress("OUTER_INFERENCE", None)
    exports = {}
    for name, seed in dict.fromkeys((plan.treatment_name, plan.seed) for plan in plans):
        runtime._reserve_resources("treatment_export")
        output = export_root / name / f"seed{seed}"
        export_initial_treatment(
            name,
            stage="S74",
            seed=seed,
            barrier_path=barrier_path,
            lock_path=runtime.lock_path,
            lock_sha256=runtime.lock_sha256,
            output=output,
            owner=runtime.owner,
        )
        exports[f"{name}_seed{seed}"] = binding(output / "TREATMENT_EXPORT.json")
    runtime._reserve_resources("replication_evidence")
    analysis = build_replication_evidence(
        exports,
        decision_source,
        lock_path=runtime.lock_path,
        lock_sha256=runtime.lock_sha256,
        owner=runtime.owner,
    )
    progress("REPLICATION_ANALYSIS_COMPLETE", None)
    return {
        "status": "ENDPOINTS_AND_EXPORTS_COMPLETE",
        "stage_authorization": decision_source,
        "barrier": binding(barrier_path),
        "exports": exports,
        "analysis": analysis,
    }
