"""Composite architecture fits using the proven full-state trainer and update ledger."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.data.event_context128 import RateNormalization
from e_jepa_ttc.data.event_context128_reader import TrainingView
from e_jepa_ttc.models.projective_phase_network import ContextEncoder
from e_jepa_ttc.training.correspondence_pretext_v11 import PretextTextures
from e_jepa_ttc.training.stage70_fixed_fit import FixedFit, deterministic_runtime
from e_jepa_ttc.training.stage70_update_ledger import FitSpec, UpdateLedger
from e_jepa_ttc.training.stage72_architecture import SupervisedBatches, initialize_architecture
from e_jepa_ttc.training.stage72_supervision import Supervision
from e_jepa_ttc.training.stage73_factorial import PlannedFit, treatment


@dataclass(frozen=True)
class FitJob:
    """Resolved immutable job identity; the stage executor authorizes its concrete source view."""

    plan: PlannedFit
    training_lock_sha256: str
    training_view_sha256: str
    normalization_sha256: str
    supervision_sha256: str
    source_role_sha256: str
    allowed_train_groups: tuple[str, ...]
    excluded_groups: tuple[str, ...]
    source_manifest_sha256: str
    stage_authorization_sha256: str | None = None

    def __post_init__(self) -> None:
        for value in (
            self.training_lock_sha256,
            self.training_view_sha256,
            self.normalization_sha256,
            self.supervision_sha256,
            self.source_role_sha256,
            self.source_manifest_sha256,
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("resolved job requires exact lock/input/schema hashes")
        if not self.allowed_train_groups or set(self.allowed_train_groups) & set(
            self.excluded_groups
        ):
            raise ValueError("job TRAIN and excluded acquisitions overlap or are empty")
        if self.plan.stage == "S75" and not self.excluded_groups:
            raise ValueError("nested job cannot omit its transitive exclusions")
        if self.plan.stage in ("S74", "S75"):
            value = self.stage_authorization_sha256
            if value is None or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("conditional fit requires its irreversible stage decision hash")
        elif self.stage_authorization_sha256 is not None:
            raise ValueError("initial matrix must not be conditioned on scientific outcomes")

    @property
    def identity(self) -> str:
        """Bind all phases of this architecture endpoint to the same fixed job."""
        return object_digest(asdict(self))


def require_scientific_lock(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Reject unfinished preflight, changed scientific bytes or an uncommitted recipe."""
    record = json.loads(path.read_text())
    signature = record.pop("artifact_sha256", None)
    if signature != expected_sha256 or object_digest(record) != signature:
        raise ValueError("scientific lock content hash changed")
    if (
        record.get("kind") != "STAGE70_76_TRAINING_LOCK"
        or record.get("tracked_worktree_clean_at_freeze") is not True
        or record.get("implementation_complete_before_first_fit") is not True
        or record.get("scientific_updates_at_freeze") != 0
        or len(record.get("code_commit", "")) != 40
    ):
        raise ValueError("scientific execution requires the complete committed pre-fit lock")
    required = {
        "handoff_integrity",
        "historical_audit",
        "timing_charter",
        "raw_histogram",
        "real_gpu_exact_resume",
        "availability_resolution",
        "baseline_failure_id_comparison",
        "lint_types",
        "cli_powershell",
        "provenance_attacks",
        "conditional_stage_integration",
        "diagnostic_and_packaging_schemas",
        "resource_plan",
    }
    receipts = record.get("preflight_receipts", {})
    if set(receipts) != required or not record.get("scientific_files"):
        raise ValueError("mandatory pre-fit integration evidence is missing")
    for receipt in (*receipts.values(), *record["scientific_files"]):
        verify(receipt)
    return record


def reserve_composite_job(
    ledger: UpdateLedger,
    job: FitJob,
    pretext_view_sha256: str | None,
) -> tuple[FitSpec | None, FitSpec]:
    """Reserve BOTH complete phases before either may consume an optimizer update."""
    recipe = treatment(job.plan.treatment_name)
    supervised = FitSpec(
        job.plan.job_id + "_supervised",
        job.plan.stage,
        recipe.architecture,
        job.plan.outer_fold,
        job.plan.seed,
        recipe.supervised_updates,
        job.identity,
        job.training_view_sha256,
    )
    pretext = None
    if recipe.pretext_updates:
        if pretext_view_sha256 is None:
            raise ValueError("pretext-enabled treatment lacks its role-bound texture pool")
        pretext = FitSpec(
            job.plan.job_id + "_pretext",
            job.plan.stage,
            "CORRESPONDENCE",
            job.plan.outer_fold,
            job.plan.seed,
            recipe.pretext_updates,
            job.identity,
            pretext_view_sha256,
        )
    elif pretext_view_sha256 is not None:
        raise ValueError("scratch treatment cannot silently inherit a pretext producer")
    # An interrupted reservation may leave a RESERVED entry; it cannot start a fit
    # until this complete, idempotent function succeeds on resume.
    ledger.reserve(supervised)
    if pretext is not None:
        ledger.reserve(pretext)
    return pretext, supervised


def _publish_weights(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    """Publish closed inference-only CPU tensors once, preserving any completed artifact."""
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        expected = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if (
            not isinstance(saved, dict)
            or set(saved) != set(expected)
            or any(
                not isinstance(saved[key], torch.Tensor)
                or saved[key].dtype != value.dtype
                or not torch.equal(saved[key], value)
                for key, value in expected.items()
            )
        ):
            raise ValueError("orphan inference weights disagree with the fixed endpoint")
        return binding(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return binding(path)


def execute_architecture_job(
    job: FitJob,
    *,
    lock_path: Path,
    view: TrainingView,
    normalization: RateNormalization,
    supervision: Supervision,
    ledger: UpdateLedger,
    owner: ResourceOwner,
    output_root: Path,
    stage_authorization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a whole prescribed endpoint synchronously; no outer evaluation or best checkpoint."""
    lock = require_scientific_lock(lock_path, job.training_lock_sha256)
    if (
        view.identity != job.training_view_sha256
        or normalization.training_view_sha256 != view.identity
        or object_digest(asdict(normalization)) != job.normalization_sha256
        or supervision.identity_sha256 != job.supervision_sha256
        or supervision.tokens != view.tokens
    ):
        raise ValueError("resolved job tensors or TRAIN normalizer/prior identity differs")
    sources = lock.get("authorized_source_manifests", {})
    if job.source_manifest_sha256 not in sources:
        raise ValueError("job source view is not authorized by the frozen stage graph")
    source = json.loads(verify(sources[job.source_manifest_sha256]).read_text())
    source_signature = source.pop("artifact_sha256", None)
    if source_signature != job.source_manifest_sha256 or object_digest(source) != source_signature:
        raise ValueError("resolved source-view manifest signature differs")
    for field, expected in {
        "training_view_sha256": view.identity,
        "ordered_tokens": list(view.tokens),
        "allowed_train_groups": list(job.allowed_train_groups),
        "excluded_groups": list(job.excluded_groups),
        "source_role_sha256": job.source_role_sha256,
    }.items():
        if source.get(field) != expected:
            raise ValueError(f"resolved source-view field differs: {field}")
    if job.source_role_sha256 != lock.get("source_role_sha256"):
        raise ValueError("job acquisition roles differ from the scientific lock")
    if job.stage_authorization_sha256 is not None:
        if stage_authorization is None:
            raise ValueError("conditional fit has no physically bound stage decision")
        from e_jepa_ttc.artifacts.architecture_stage_decisions_v11 import verify_stage_decision

        verified_decision, verified_plans = verify_stage_decision(
            stage_authorization,
            {**lock, "artifact_sha256": job.training_lock_sha256},
            job.plan.stage,
        )
        if (
            job.plan not in verified_plans
            or verified_decision["artifact_sha256"] != job.stage_authorization_sha256
        ):
            raise ValueError("conditional job differs from its source-bound canonical decision")
        decision = json.loads(verify(stage_authorization).read_text())
        decision_signature = decision.pop("artifact_sha256", None)
        if (
            decision_signature != job.stage_authorization_sha256
            or object_digest(decision) != decision_signature
            or decision.get("training_lock_sha256") != job.training_lock_sha256
            or decision.get("stage") != job.plan.stage
            or job.plan.job_id not in decision.get("authorized_job_ids", [])
        ):
            raise ValueError("conditional job is not authorized by its frozen evidence decision")
    elif stage_authorization is not None:
        raise ValueError("initial matrix must not be selected using an outcome decision")
    root = output_root.resolve() / job.plan.job_id
    if root.parent != output_root.resolve():
        raise ValueError("job output escaped its owned root")
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "FIT_ENDPOINT.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        signature = result.pop("artifact_sha256")
        if object_digest(result) != signature or result["job_sha256"] != job.identity:
            raise ValueError("completed architecture endpoint identity differs")
        verify(result["checkpoint"])
        verify(result["inference_weights"])
        return {**result, "artifact_sha256": signature}
    recipe = treatment(job.plan.treatment_name)
    pool = (
        PretextTextures(view, normalization, root / "pretext/TEXTURE_TRACE.jsonl")
        if recipe.pretext_updates
        else None
    )
    pretext_spec, supervised_spec = reserve_composite_job(
        ledger, job, None if pool is None else pool.identity
    )
    if not torch.cuda.is_available():
        raise RuntimeError("scientific architecture execution requires the measured CUDA runtime")
    torch.set_num_threads(2)
    pretext_endpoint = None
    encoder_state = None
    if pretext_spec is not None:
        assert pool is not None
        deterministic_runtime(job.plan.seed)
        encoder = ContextEncoder().cuda()
        fit = FixedFit(encoder, pretext_spec, ledger, owner, root / "pretext", warmup=150)
        state = ledger.state["jobs"][pretext_spec.job_id]
        if state["status"] != "COMPLETE":
            fit.resume()
            if fit.update == 0 and state["checkpoint"] is None:
                fit.save()
            fit.run(
                lambda update, index, batch, generator: pool.objective(
                    encoder, update=update, microbatch_index=index, batch=batch, generator=generator
                )
            )
        pretext_endpoint = ledger.state["jobs"][pretext_spec.job_id]["endpoint"]
        payload = torch.load(verify(pretext_endpoint), map_location="cpu", weights_only=False)
        if payload["update"] != 3000 or payload["spec"] != asdict(pretext_spec):
            raise ValueError("pretext transfer requires the complete registered encoder endpoint")
        encoder_state = payload["model"]
        del payload, fit, encoder
        torch.cuda.empty_cache()
    model = initialize_architecture(job.plan.seed, supervision, encoder_state=encoder_state).cuda()
    del encoder_state
    fit = FixedFit(
        model, supervised_spec, ledger, owner, root / "supervised", warmup=recipe.supervised_warmup
    )
    state = ledger.state["jobs"][supervised_spec.job_id]
    if state["status"] != "COMPLETE":
        fit.resume()
        if fit.update == 0 and state["checkpoint"] is None:
            fit.save()
        fit.run(
            SupervisedBatches(
                model,
                view,
                normalization,
                supervision,
                recipe.architecture,
                job.plan.seed,
                root / "supervised/TRAIN_TRACE.jsonl",
            )
        )
    checkpoint = ledger.state["jobs"][supervised_spec.job_id]["endpoint"]
    payload = torch.load(verify(checkpoint), map_location="cpu", weights_only=False)
    if payload["update"] != recipe.supervised_updates or payload["spec"] != asdict(supervised_spec):
        raise ValueError("partial or mismatched supervised checkpoint cannot become an endpoint")
    model.cpu().load_state_dict(payload["model"], strict=True)
    del payload, fit
    result = {
        "kind": "STANDALONE_ARCHITECTURE_ENDPOINT",
        "job_sha256": job.identity,
        "job": asdict(job),
        "training_code_commit": lock["code_commit"],
        "checkpoint": checkpoint,
        "pretext_checkpoint": pretext_endpoint,
        "inference_weights": _publish_weights(model, root / "inference_weights.pt"),
        "completed_supervised_updates": recipe.supervised_updates,
        "completed_pretext_updates": recipe.pretext_updates,
        "scientific_endpoint": True,
        "outer_evaluated": False,
    }
    result["artifact_sha256"] = object_digest(result)
    atomic_json(result_path, result)
    return result
