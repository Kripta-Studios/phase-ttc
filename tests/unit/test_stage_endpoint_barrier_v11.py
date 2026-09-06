"""Synthetic endpoint tampering tests; fixtures never execute an optimizer step."""

import json
from dataclasses import asdict, replace

import pytest
import torch

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import (
    _verify_job_source,
    freeze_initial_stage,
    validate_stage_membership,
    verify_fit_endpoint,
    verify_initial_barrier,
)
from e_jepa_ttc.training.stage70_update_ledger import FitSpec
from e_jepa_ttc.training.stage72_jobs import FitJob
from e_jepa_ttc.training.stage73_factorial import Availability, initial_jobs, treatment

COMMIT = "7" * 40
AVAILABILITY = Availability(True, True, "a" * 64, "b" * 64)


def jobs(stage):
    return tuple(
        FitJob(
            plan, "a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64, ("train",), ("outer",), "f" * 64
        )
        for plan in initial_jobs(AVAILABILITY)
        if plan.stage == stage
    )


def signed(path, record):
    atomic_json(path, {**record, "artifact_sha256": object_digest(record)})


def synthetic_endpoint(root, job):
    """Deliberately fabricated test fixture, confined to pytest's temporary directory."""
    recipe = treatment(job.plan.treatment_name)
    directory = root / job.plan.job_id
    (directory / "supervised").mkdir(parents=True)
    checkpoint = directory / "supervised" / f"checkpoint_{recipe.supervised_updates:06d}.pt"
    weights = directory / "inference_weights.pt"
    spec = FitSpec(
        job.plan.job_id + "_supervised",
        job.plan.stage,
        recipe.architecture,
        job.plan.outer_fold,
        job.plan.seed,
        recipe.supervised_updates,
        job.identity,
        job.training_view_sha256,
    )
    runtime = {
        "warmup": recipe.supervised_warmup,
        "effective_batch": 16,
        "precision": "float32_TF32_off",
        "deterministic": True,
        "checkpoint_interval": 250,
        "cublas_workspace_config": ":4096:8",
        "microbatch": 2,
    }
    payload = {
        "spec": asdict(spec),
        "update": spec.endpoint,
        "optimizer": {},
        "rng": {},
        "runtime": runtime,
        "history": [{"update": u} for u in range(1, spec.endpoint + 1)],
        "identity": object_digest({"spec": asdict(spec), "runtime": runtime}),
        "model": {"weight": torch.tensor([1.0, 2.0])},
    }
    torch.save(payload, checkpoint)
    torch.save(payload["model"], weights)
    record = {
        "kind": "STANDALONE_ARCHITECTURE_ENDPOINT",
        "job_sha256": job.identity,
        "job": asdict(job),
        "training_code_commit": COMMIT,
        "scientific_endpoint": True,
        "outer_evaluated": False,
        "completed_supervised_updates": recipe.supervised_updates,
        "completed_pretext_updates": recipe.pretext_updates,
        "checkpoint": binding(checkpoint),
        "inference_weights": binding(weights),
        "pretext_checkpoint": None,
    }
    path = directory / "FIT_ENDPOINT.json"
    signed(path, record)
    return path, record, payload


@pytest.mark.parametrize("stage,count", [("S72", 12), ("S73", 15)])
def test_every_prescribed_endpoint_required_even_when_one_model_is_better(stage, count):
    planned = jobs(stage)
    assert len(planned) == count
    validate_stage_membership(stage, planned, AVAILABILITY)
    for invalid in (planned[:-1], planned + planned[:1], planned[:-1] + planned[:1]):
        with pytest.raises(ValueError, match="all prescribed"):
            validate_stage_membership(stage, invalid, AVAILABILITY)


def test_verified_endpoint_requires_exact_weight_values_and_physical_bytes(tmp_path):
    job = jobs("S72")[0]
    path, record, _ = synthetic_endpoint(tmp_path, job)
    result = verify_fit_endpoint(job, tmp_path, COMMIT)
    assert result["endpoint"] == binding(path)
    weights = path.parent / "inference_weights.pt"
    torch.save({"weight": torch.tensor([1.0, 3.0])}, weights)
    with pytest.raises(ValueError, match="physical identity"):
        verify_fit_endpoint(job, tmp_path, COMMIT)
    record["inference_weights"] = binding(weights)
    signed(path, record)
    with pytest.raises(ValueError, match="inference weights differ"):
        verify_fit_endpoint(job, tmp_path, COMMIT)


@pytest.mark.parametrize("damage", ["partial", "history", "runtime", "fold", "identity"])
def test_resigned_checkpoint_still_cannot_change_prescribed_contract(tmp_path, damage):
    job = jobs("S72")[0]
    path, record, payload = synthetic_endpoint(tmp_path, job)
    if damage == "partial":
        payload["update"] -= 1
    elif damage == "history":
        payload["history"].pop(20)
    elif damage == "runtime":
        payload["runtime"]["effective_batch"] = 8
    elif damage == "fold":
        payload["spec"]["fold"] = 1
    else:
        payload["identity"] = "0" * 64
    checkpoint = path.parent / "supervised" / "checkpoint_008000.pt"
    torch.save(payload, checkpoint)
    record["checkpoint"] = binding(checkpoint)
    signed(path, record)
    with pytest.raises(ValueError):
        verify_fit_endpoint(job, tmp_path, COMMIT)


def test_same_weight_bytes_from_other_job_are_rejected(tmp_path):
    first, second = jobs("S72")[:2]
    path, record, _ = synthetic_endpoint(tmp_path, first)
    other, _, _ = synthetic_endpoint(tmp_path, second)
    record["inference_weights"] = binding(other.parent / "inference_weights.pt")
    signed(path, record)
    with pytest.raises(ValueError, match="owned path"):
        verify_fit_endpoint(first, tmp_path, COMMIT)


def test_pretext_treatment_cannot_claim_completion_without_encoder_parent(tmp_path):
    job = next(j for j in jobs("S73") if treatment(j.plan.treatment_name).pretext_updates)
    synthetic_endpoint(tmp_path, job)
    with pytest.raises(ValueError, match="parent encoder"):
        verify_fit_endpoint(job, tmp_path, COMMIT)


def test_source_fold_assignment_and_acquisition_exclusions_are_rechecked(tmp_path):
    job = jobs("S72")[0]
    normalization = {"training_view_sha256": job.training_view_sha256}
    norm_path = tmp_path / "NORMALIZATION.json"
    supervision_path = tmp_path / "SUPERVISION.json"
    atomic_json(norm_path, normalization)
    atomic_json(
        supervision_path,
        {
            "training_view_sha256": job.training_view_sha256,
            "identity_sha256": job.supervision_sha256,
            "tokens": ["train0"],
        },
    )
    job = replace(job, normalization_sha256=object_digest(normalization))
    source = {
        "outer_fold": job.plan.outer_fold,
        "training_view_sha256": job.training_view_sha256,
        "allowed_train_groups": list(job.allowed_train_groups),
        "excluded_groups": list(job.excluded_groups),
        "source_role_sha256": job.source_role_sha256,
        "include_expansion": False,
        "ordered_tokens": ["train0"],
        "normalization": binding(norm_path),
        "supervision": binding(supervision_path),
    }
    path = tmp_path / "SOURCE_VIEW.json"
    signed(path, source)
    job = replace(job, source_manifest_sha256=object_digest(source))
    lock = {
        "initial_source_manifest_by_job": {job.plan.job_id: job.source_manifest_sha256},
        "authorized_source_manifests": {job.source_manifest_sha256: binding(path)},
        "source_role_sha256": job.source_role_sha256,
    }
    _verify_job_source(job, lock)
    with pytest.raises(ValueError, match="excluded_groups"):
        _verify_job_source(replace(job, excluded_groups=("other",)), lock)
    with pytest.raises(ValueError, match="normalization or supervision"):
        _verify_job_source(replace(job, normalization_sha256="0" * 64), lock)
    lock["initial_source_manifest_by_job"][job.plan.job_id] = "0" * 64
    with pytest.raises(ValueError, match="fold assignment"):
        _verify_job_source(job, lock)


def test_self_signed_barrier_cannot_omit_any_stage_endpoint(tmp_path):
    path = tmp_path / "BARRIER.json"
    record = {
        "kind": "ALL_INITIAL_STAGE_ENDPOINTS_FROZEN",
        "stage": "S72",
        "training_lock_sha256": "a" * 64,
        "outer_inference_authorized": True,
        "selection_using_outer_results": False,
        "availability": asdict(AVAILABILITY),
        "endpoints": [{"job_id": job.plan.job_id} for job in jobs("S72")[:-1]],
        "required_endpoint_count": 11,
    }
    signed(path, record)
    with pytest.raises(ValueError, match="omits or duplicates"):
        verify_initial_barrier(path, expected_stage="S72", expected_lock="a" * 64)
    value = json.loads(path.read_text())
    value["outer_inference_authorized"] = False
    atomic_json(path, value)
    with pytest.raises(ValueError, match="invalid initial-stage"):
        verify_initial_barrier(path, expected_stage="S72", expected_lock="a" * 64)


def test_complete_synthetic_stage_publishes_once_and_rechecks_every_payload(tmp_path):
    planned = jobs("S72")
    sources = {}
    resolved = []
    for job in planned:
        directory = tmp_path / "sources" / job.plan.job_id
        directory.mkdir(parents=True)
        norm = {"training_view_sha256": job.training_view_sha256}
        atomic_json(directory / "norm.json", norm)
        atomic_json(
            directory / "supervision.json",
            {
                "training_view_sha256": job.training_view_sha256,
                "identity_sha256": job.supervision_sha256,
                "tokens": ["train0"],
            },
        )
        source = {
            "outer_fold": job.plan.outer_fold,
            "training_view_sha256": job.training_view_sha256,
            "allowed_train_groups": list(job.allowed_train_groups),
            "excluded_groups": list(job.excluded_groups),
            "source_role_sha256": job.source_role_sha256,
            "include_expansion": False,
            "ordered_tokens": ["train0"],
            "normalization": binding(directory / "norm.json"),
            "supervision": binding(directory / "supervision.json"),
        }
        path = directory / "source.json"
        signed(path, source)
        sha = object_digest(source)
        sources[sha] = binding(path)
        resolved.append(
            replace(job, source_manifest_sha256=sha, normalization_sha256=object_digest(norm))
        )
    receipt = tmp_path / "synthetic_receipt.json"
    atomic_json(receipt, {"kind": "SYNTHETIC_TEST_ONLY", "optimizer_updates": 0})
    required = (
        "handoff_integrity historical_audit timing_charter raw_histogram real_gpu_exact_resume "
        "availability_resolution baseline_failure_id_comparison lint_types cli_powershell "
        "provenance_attacks conditional_stage_integration diagnostic_and_packaging_schemas "
        "resource_plan"
    ).split()
    root = tmp_path / "synthetic_endpoints"
    lock = {
        "kind": "STAGE70_76_TRAINING_LOCK",
        "tracked_worktree_clean_at_freeze": True,
        "implementation_complete_before_first_fit": True,
        "scientific_updates_at_freeze": 0,
        "code_commit": COMMIT,
        "preflight_receipts": {key: binding(receipt) for key in required},
        "scientific_files": [binding(receipt)],
        "endpoint_root": str(root.resolve()),
        "availability": asdict(AVAILABILITY),
        "source_role_sha256": "e" * 64,
        "authorized_source_manifests": sources,
        "initial_source_manifest_by_job": {
            j.plan.job_id: j.source_manifest_sha256 for j in resolved
        },
    }
    lock_path = tmp_path / "SYNTHETIC_LOCK.json"
    signed(lock_path, lock)
    lock_sha = object_digest(lock)
    resolved = tuple(replace(j, training_lock_sha256=lock_sha) for j in resolved)
    for job in resolved:
        synthetic_endpoint(root, job)
    output = tmp_path / "SYNTHETIC_BARRIER.json"
    first = freeze_initial_stage(
        "S72",
        resolved,
        AVAILABILITY,
        lock_path=lock_path,
        lock_sha256=lock_sha,
        endpoint_root=root,
        output=output,
    )
    assert first["required_endpoint_count"] == 12
    original = binding(output)
    assert (
        freeze_initial_stage(
            "S72",
            resolved,
            AVAILABILITY,
            lock_path=lock_path,
            lock_sha256=lock_sha,
            endpoint_root=root,
            output=output,
        )
        == first
    )
    assert binding(output) == original
    weights = root / resolved[-1].plan.job_id / "inference_weights.pt"
    with weights.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="physical identity"):
        verify_initial_barrier(output, expected_stage="S72", expected_lock=lock_sha)
