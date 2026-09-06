"""Source-bound sequential execution of every available initial architecture treatment."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.architecture_stage_decisions_v11 import publish_stage_decision
from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner, host_ram_reservation
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import freeze_initial_stage
from e_jepa_ttc.data.composite_counts_v11 import CompositeCountReader
from e_jepa_ttc.data.event_context128_reader import CountReader
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.evaluation.initial_campaign_evidence_v11 import build_initial_evidence
from e_jepa_ttc.evaluation.initial_treatment_export_v11 import export_initial_treatment
from e_jepa_ttc.evaluation.pretext_diagnostics_v11 import diagnose_pretext_checkpoint
from e_jepa_ttc.evaluation.standalone_outer_v11 import infer_initial_endpoint
from e_jepa_ttc.training.correspondence_pretext_v11 import PretextTextures
from e_jepa_ttc.training.resolved_architecture_sources_v11 import (
    ResolvedArchitectureSources,
    load_architecture_sources,
)
from e_jepa_ttc.training.stage70_update_ledger import UpdateLedger
from e_jepa_ttc.training.stage72_jobs import (
    FitJob,
    execute_architecture_job,
    require_scientific_lock,
)
from e_jepa_ttc.training.stage73_factorial import Availability, PlannedFit, initial_jobs, treatment


class InitialArchitectureRuntime:
    """Run S72 AND S73 before any outcome can affect their available job matrix.

    This is the initial-stage backend of the complete campaign, not a replacement
    for S74–76. A complete scientific lock and the single GPU owner are mandatory.
    """

    def __init__(
        self, lock_path: Path, lock_sha256: str, owner: ResourceOwner, ledger: UpdateLedger
    ) -> None:
        self.lock_path, self.lock_sha256, self.owner, self.ledger = (
            lock_path,
            lock_sha256,
            owner,
            ledger,
        )
        self.lock = require_scientific_lock(lock_path, lock_sha256)
        self.availability = Availability(**self.lock["availability"])
        self.plans = initial_jobs(self.availability)
        self.root = Path(self.lock["campaign_root"])
        if (
            self.root.resolve() != self.root
            or owner.root.resolve() != self.root / "training_writer"
        ):
            raise ValueError("initial-stage runtime requires its frozen campaign GPU owner")
        self.paths = {}
        for field in (
            "endpoint_root",
            "outer_inference_root",
            "treatment_export_root",
            "stage_barrier_root",
            "initial_evidence_root",
            "stage_decision_root",
        ):
            path = Path(self.lock[field])
            if path.resolve() != path or not path.is_relative_to(self.root):
                raise ValueError("scientific outputs escaped their frozen campaign root")
            self.paths[field] = path
        role_record = json.loads(verify(self.lock["data_roles"]).read_text(encoding="utf-8"))
        signature = role_record.pop("artifact_sha256")
        if object_digest(role_record) != signature or signature != self.lock["source_role_sha256"]:
            raise ValueError("runtime source roles differ from the scientific lock")
        values = role_record["roles"]
        self.roles = TrainRoles(
            tuple(tuple(pair) for pair in values["sequence_to_group"]),
            tuple(values["original"]),
            tuple(values["historical_exposed"]),
            tuple(values["protected"]),
            tuple(values["confirmation"]),
            tuple(values["expansion"]),
            values["role_prefix"],
        )
        self.original = CountReader(verify(self.lock["count_caches"]["original"]).parent)
        self.expanded = None
        if self.availability.diversity:
            additional = CountReader(verify(self.lock["count_caches"]["expanded"]).parent)
            self.expanded = CompositeCountReader((self.original, additional), self.roles)
        if self.original.identity != self.lock["outer_count_cache_sha256"]:
            raise ValueError("runtime original cache differs from locked evaluation rows")
        self.queue_path = self.root / "INITIAL_JOB_QUEUE.json"

    def resolve(self, plan: PlannedFit) -> tuple[FitJob, ResolvedArchitectureSources]:
        """Load real source tensors through the same audited production adapter as QA."""
        if plan not in self.plans:
            raise ValueError("job is absent from the available initial recipe matrix")
        sha = self.lock["initial_source_manifest_by_job"][plan.job_id]
        expanded = treatment(plan.treatment_name).expanded
        reader = self.expanded if expanded else self.original
        if reader is None:
            raise ValueError("D treatment has no authorized expanded count source")
        loaded = load_architecture_sources(
            self.lock["authorized_source_manifests"][sha],
            reader,
            self.roles,
            expected_fold=plan.outer_fold,
            expected_expansion=expanded,
            source_role_sha256=self.lock["source_role_sha256"],
        )
        if loaded.source_sha256 != sha:
            raise ValueError("resolved job source differs from its frozen assignment")
        job = FitJob(
            plan,
            self.lock_sha256,
            loaded.view.identity,
            object_digest(asdict(loaded.normalization)),
            loaded.supervision.identity_sha256,
            self.lock["source_role_sha256"],
            tuple(loaded.source["allowed_train_groups"]),
            tuple(loaded.source["excluded_groups"]),
            sha,
        )
        return job, loaded

    def _reserve_resources(self, key: str) -> None:
        reservation = self.lock["execution_reservations"][key]
        disk, ram = reservation["disk_bytes"], reservation["ram_bytes"]
        if not isinstance(disk, int) or not isinstance(ram, int) or min(disk, ram) <= 0:
            raise ValueError("each operation requires its entire positive resource reservation")
        if not host_ram_reservation(ram).passes:
            raise RuntimeError(
                "RESOURCE_BLOCKED: full operation RAM reservation leaves less than4GiB"
            )
        self.owner.estimated_remaining_bytes = disk
        self.owner.check()

    def _progress(self, operation: str, active: str | None, completed: list[str]) -> None:
        atomic_json(
            self.queue_path,
            {
                "training_lock_sha256": self.lock_sha256,
                "operation": operation,
                "active_job": active,
                "completed_in_operation": completed,
                "prescribed_jobs": [plan.job_id for plan in self.plans],
                "optimizer_totals": self.ledger.totals(),
                "checkpoint_and_rng_authority": str(self.ledger.path),
                "scientific_campaign_complete": False,
            },
        )

    def fit_initial_stages(self) -> dict[str, Any]:
        """Complete and close both initial stages independently of S72 accuracy or CIs."""
        barriers = {}
        completed: list[str] = []
        for stage in ("S72", "S73"):
            jobs = []
            for plan in self.plans:
                if plan.stage != stage:
                    continue
                self._reserve_resources(plan.job_id)
                self._progress("FITTING", plan.job_id, completed)
                job, loaded = self.resolve(plan)
                execute_architecture_job(
                    job,
                    lock_path=self.lock_path,
                    view=loaded.view,
                    normalization=loaded.normalization,
                    supervision=loaded.supervision,
                    ledger=self.ledger,
                    owner=self.owner,
                    output_root=self.paths["endpoint_root"],
                )
                if treatment(plan.treatment_name).pretext_updates:
                    pool = PretextTextures(
                        loaded.view,
                        loaded.normalization,
                        self.paths["endpoint_root"]
                        / plan.job_id
                        / "pretext/UNUSED_DIAGNOSTIC_POOL.jsonl",
                    )
                    for update in (0, 3000):
                        checkpoint = (
                            self.paths["endpoint_root"]
                            / plan.job_id
                            / "pretext"
                            / f"checkpoint_{update:06d}.pt"
                        )
                        diagnose_pretext_checkpoint(
                            binding(checkpoint),
                            pool,
                            seed=plan.seed,
                            output=self.paths["endpoint_root"]
                            / plan.job_id
                            / f"pretext_diagnostics_{update:06d}",
                            check=self.owner.check,
                        )
                jobs.append(job)
                completed.append(plan.job_id)
                self._progress("FITTING", None, completed)
            path = self.paths["stage_barrier_root"] / f"{stage}_ALL_ENDPOINTS_FROZEN.json"
            freeze_initial_stage(
                stage,
                tuple(jobs),
                self.availability,
                lock_path=self.lock_path,
                lock_sha256=self.lock_sha256,
                endpoint_root=self.paths["endpoint_root"],
                output=path,
            )
            barriers[stage] = binding(path)
        self._progress("BOTH_INITIAL_STAGES_FROZEN", None, completed)
        return barriers

    def infer_and_export(self, barriers: dict[str, Any]) -> dict[str, Any]:
        """Publish each complete outer prediction and report, retaining every fixed treatment."""
        if set(barriers) != {"S72", "S73"}:
            raise ValueError("both independent initial stages must close before joint analysis")
        completed: list[str] = []
        for plan in self.plans:
            self._reserve_resources("outer_inference")
            self._progress("OUTER_INFERENCE", plan.job_id, completed)
            job, loaded = self.resolve(plan)
            infer_initial_endpoint(
                job,
                barrier_path=verify(barriers[plan.stage]),
                lock_path=self.lock_path,
                reader=self.original,
                tokens=tuple(self.lock["outer_tokens_by_fold"][str(plan.outer_fold)]),
                normalization=loaded.normalization,
                output=self.paths["outer_inference_root"] / plan.job_id,
                owner=self.owner,
            )
            completed.append(plan.job_id)
            self._progress("OUTER_INFERENCE", None, completed)
        exports = {}
        for name in dict.fromkeys(plan.treatment_name for plan in self.plans):
            self._reserve_resources("treatment_export")
            stage = "S72" if name.startswith("S72") else "S73"
            export_initial_treatment(
                name,
                stage=stage,
                barrier_path=verify(barriers[stage]),
                lock_path=self.lock_path,
                lock_sha256=self.lock_sha256,
                output=self.paths["treatment_export_root"] / name,
                owner=self.owner,
            )
            exports[name] = binding(
                self.paths["treatment_export_root"] / name / "TREATMENT_EXPORT.json"
            )
        self._progress("INITIAL_EXPORTS_COMPLETE_CONDITIONAL_STAGES_PENDING", None, completed)
        return exports

    def analyze_initial_stages(
        self, exports: dict[str, Any], barriers: dict[str, Any]
    ) -> dict[str, Any]:
        """Bind fixed endpoint statistics to both exact conditional stage decisions."""
        self._reserve_resources("initial_evidence")
        self._progress("INITIAL_PAIRED_ANALYSIS", None, [])
        build_initial_evidence(
            exports,
            barriers,
            lock_path=self.lock_path,
            lock_sha256=self.lock_sha256,
            output=self.paths["initial_evidence_root"],
            owner=self.owner,
        )
        decisions = {}
        for stage in ("S74", "S75"):
            publish_stage_decision(stage, {**self.lock, "artifact_sha256": self.lock_sha256})
            decisions[stage] = binding(self.paths["stage_decision_root"] / f"{stage}_DECISION.json")
        self._progress("INITIAL_EVIDENCE_COMPLETE_CONDITIONAL_DECISIONS_FROZEN", None, [])
        return decisions
