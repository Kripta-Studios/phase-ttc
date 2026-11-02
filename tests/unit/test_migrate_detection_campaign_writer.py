from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path

import pytest
import torch


def load_script():
    path = Path(__file__).parents[2] / "scripts/migrate_detection_campaign_writer.py"
    spec = importlib.util.spec_from_file_location("migrate_detection_campaign_writer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def old_trainer() -> Path:
    path = (
        Path(__file__).parents[2]
        / "tests/fixtures/detection_candidate_sources/writer_train_detection_campaign.py.txt"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "168b1041de16e1d6adce10f00cdce6b2e7c02ecbdee2ef08e0a4f6a9c00bc8da"
    )
    return path


def test_current_cold_restart_trainer_is_rejected_by_writer_only_migrator() -> None:
    module = load_script()
    new = Path(__file__).parents[2] / "scripts/train_detection_campaign.py"
    with pytest.raises(ValueError, match="leave the allowlist"):
        module.verify_writer_only_ast(old_trainer(), new)


def test_streaming_serializer_is_byte_exact_and_rejects_nonfinite() -> None:
    module = load_script()
    payload = {
        "metrics": {"AP": 0.02064, "curve": [0, 0.5, 1]},
        "predictions": {"token": [{"score": 0.25, "class_name": "car"}]},
        "unicode": "fase",
    }
    receipt = module.verify_serializer_parity(payload)
    assert receipt["status"] == "PASS"
    with pytest.raises(ValueError):
        module.serializer_bytes({"bad": float("nan")}, streaming=False)
    with pytest.raises(ValueError):
        module.serializer_bytes({"bad": float("nan")}, streaming=True)


def make_prereg(module, code_root: Path, trainer: Path) -> dict:
    trainer_destination = code_root / module.TRAINER_RELATIVE
    trainer_destination.parent.mkdir(parents=True)
    shutil.copy2(trainer, trainer_destination)
    portable = code_root / "portable_source/model.py"
    portable.parent.mkdir(parents=True)
    portable.write_text("VALUE = 1\n", encoding="utf-8")
    code = {module.TRAINER_RELATIVE: module.sha256_file(trainer_destination)}
    portable_hashes = {"model.py": module.sha256_file(portable)}
    source_receipt = {"upstream_revision": "fixed", "patch_sha256": "a" * 64}
    effective_model = {"num_classes": 8, "num_queries": 50, "device": "cuda"}
    return {
        "schema": "eap_detection_campaign_runtime_preregistration_v1",
        "config": {"seed": 7, "updates": 8192},
        "config_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
        "source_receipt": source_receipt,
        "source_receipt_sha256": module.canonical_sha256(source_receipt),
        "effective_model": effective_model,
        "effective_model_sha256": module.canonical_sha256(effective_model),
        "code_sha256": code,
        "code_sha256_manifest": module.canonical_sha256(code),
        "portable_source_sha256": portable_hashes,
        "portable_source_sha256_manifest": module.canonical_sha256(portable_hashes),
        "dataset_media_sha256": "d" * 64,
        "matmul_precision": "highest",
    }


def checkpoint_payload(module, provenance: dict, update: int, best_metric=None) -> dict:
    return {
        "schema": module.CHECKPOINT_SCHEMA,
        "producer": module.TRAINER_RELATIVE,
        "inference_contract": "campaign_v1_strict_loader_required",
        "config_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
        "runtime_provenance": provenance,
        "runtime_provenance_sha256": module.canonical_sha256(provenance),
        "update": update,
        "best_metric": best_metric,
        "progress": {
            "training_seconds": 12.5,
            "empty_target_exposures": 2,
            "valid_velocity_object_exposures": 9,
        },
        "data_member_sha256": {"sequence:frame": "e" * 64},
        "model": {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
        "optimizer": {"state": {0: {"step": torch.tensor(4), "exp_avg": torch.ones(2)}}},
        "scheduler": {"update": update, "base_lrs": [0.0002]},
        "sampler": {"epoch": 1, "position": 3, "order": [2, 0, 1]},
        "rng": {
            "python": (3, (1, 2, 3), None),
            "numpy": {"keys": torch.arange(4, dtype=torch.int64), "position": 1},
            "torch_cpu": torch.arange(8, dtype=torch.uint8),
            "torch_cuda": [torch.arange(4, dtype=torch.uint8)],
        },
    }


def profile_payload(module, *, admitted: bool) -> dict:
    observed_evaluator = 2_000_000
    fixed = 1_000_000
    projected_evaluator = math.ceil(observed_evaluator * (23958 / 4096) * 1.5)
    projected_total = fixed + projected_evaluator
    projection = {
        "streaming_rss_observed_sample_count": 4096,
        "full_population_sample_count": 23958,
        "projection_multiplier": 1.5,
        "measured_evaluator_peak_rss_bytes": observed_evaluator,
        "evaluator_measurement_excludes_trainer_model": True,
        "fixed_trainer_model_cache_rss_bytes": fixed,
        "projected_full_evaluator_peak_rss_bytes": projected_evaluator,
        "projected_full_total_physical_ram_bytes": projected_total,
        "observed_prestart_physical_free_ram_bytes": 100_000_000,
        "observed_runtime_min_physical_free_ram_bytes": 100_000_000,
        "physical_safety_margin_bytes": 2_000_000,
        "memory_basis": "physical_ram_not_commit_or_pagefile",
        "projection_is_measurement": False,
        "admission_mode": "migration_before_model_load_total",
    }
    return {
        "schema": "eap_detection_streaming_writer_profile_v1",
        "status": "PASS",
        "old_writer_sha256": module.OLD_WRITER_SHA256,
        "new_writer_sha256": module.NEW_WRITER_SHA256,
        "serializer_byte_parity": True,
        "metric_parity": True,
        "peak_memory_admission": admitted,
        "profile_scope": "observed_1024_4096_projected_23958",
        "observed_parity_sample_counts": [1024, 4096],
        "full_population_observed": False,
        "independent_review_status": "PASS",
        "independent_reviewer": "fixture-reviewer",
        "profiler_sha256": "3" * 64,
        "source_selection_sha256": "4" * 64,
        "protocol_sha256": "5" * 64,
        "serializer_sha256_by_sample_count": {
            "1024": {"old": "f" * 64, "new": "f" * 64},
            "4096": {"old": "e" * 64, "new": "e" * 64},
        },
        "metrics_sha256_by_sample_count": {
            "1024": {"old": "1" * 64, "new": "1" * 64},
            "4096": {"old": "2" * 64, "new": "2" * 64},
        },
        **projection,
        "reviewed_projection_sha256": module.canonical_sha256(projection),
    }


def attach_profile_evidence(root: Path, module, profile: dict) -> dict:
    evidence_values = {
        "serialization_1024": {
            "status": "PASS",
            "frame_count": 1024,
            "byte_exact_match": True,
            "payload_sha256_both_modes": profile["serializer_sha256_by_sample_count"]["1024"][
                "old"
            ],
        },
        "stream_worker_4096": {
            "status": "PASS",
            "frame_count": 4096,
            "serialized_sha256": profile["serializer_sha256_by_sample_count"]["4096"]["new"],
            "process_peak_rss_bytes": profile["measured_evaluator_peak_rss_bytes"],
            "executed_script_sha256": profile["profiler_sha256"],
            "source_sha256": profile["source_selection_sha256"],
            "protocol_sha256": profile["protocol_sha256"],
        },
        "legacy_worker_4096": {
            "status": "PASS",
            "frame_count": 4096,
            "serialized_sha256": profile["serializer_sha256_by_sample_count"]["4096"]["old"],
            "process_peak_rss_bytes": 3_000_000,
            "executed_script_sha256": profile["profiler_sha256"],
            "source_sha256": profile["source_selection_sha256"],
            "protocol_sha256": profile["protocol_sha256"],
        },
        "metric_parity_receipt": {
            "status": "PASS",
            "serializer_sha256_by_sample_count": profile["serializer_sha256_by_sample_count"],
            "metrics_sha256_by_sample_count": profile["metrics_sha256_by_sample_count"],
        },
        "admission_receipt": {
            "status": "CONDITIONALLY_ADMIT_STREAMING_ONLY",
            "executed_profiler_sha256": profile["profiler_sha256"],
            "source_selection_sha256": profile["source_selection_sha256"],
            "projection_23958": {
                "fixed_trainer_model_cache": {
                    "windows_peak_working_set_bytes": 900_000,
                    "conservative_allowance_bytes": profile["fixed_trainer_model_cache_rss_bytes"],
                }
            },
        },
    }
    evidence = {}
    for name, value in evidence_values.items():
        path = root / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        evidence[name] = {"path": path.name, "sha256": module.sha256_file(path)}
    for mode, worker_name in (
        ("stream", "stream_worker_4096"),
        ("legacy", "legacy_worker_4096"),
    ):
        worker_sha = evidence[worker_name]["sha256"]
        watchdog_name = f"{mode}_watchdog_4096"
        value = {
            "status": "PASS",
            "returncode": 0,
            "abort_reason": None,
            "observed_owned_pids": [1, 2],
            "worker_receipt_sha256": worker_sha,
            "prestart_host_available_bytes": profile["observed_prestart_physical_free_ram_bytes"],
            "minimum_host_available_bytes": profile["observed_runtime_min_physical_free_ram_bytes"],
        }
        path = root / f"{watchdog_name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        evidence[watchdog_name] = {"path": path.name, "sha256": module.sha256_file(path)}
    profile["evidence"] = evidence
    return profile


def test_abandoned_migration_cannot_accept_cold_restart_trainer(tmp_path: Path) -> None:
    module = load_script()
    old_run = tmp_path / "old_run"
    new_freeze = tmp_path / "new_freeze"
    old_code, new_code = old_run / "code", new_freeze / "code"
    old_prereg = make_prereg(module, old_code, old_trainer())
    new_trainer = Path(__file__).parents[2] / "scripts/train_detection_campaign.py"
    new_prereg = make_prereg(module, new_code, new_trainer)
    old_prereg_path = old_run / "runtime_preregistration.json"
    new_prereg_path = new_freeze / "runtime_preregistration.json"
    old_prereg_path.write_text(json.dumps(old_prereg, indent=2) + "\n", encoding="utf-8")
    new_prereg_path.write_text(json.dumps(new_prereg, indent=2) + "\n", encoding="utf-8")
    (old_run / "failure.json").write_text(
        json.dumps(
            {
                "status": "FAILED",
                "error_type": "OSError",
                "error": "image file is truncated",
            }
        ),
        encoding="utf-8",
    )
    (old_run / "preregistration.json").write_text(
        json.dumps({"parent_run": None, "parent_run_chain": []}), encoding="utf-8"
    )
    with (old_run / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for update in range(1, 726):
            stream.write(json.dumps({"update": update}) + "\n")
    old_provenance = module._runtime_provenance(old_prereg_path, old_prereg)
    checkpoint_dir = old_run / "checkpoints"
    checkpoint_dir.mkdir()
    torch.save(
        checkpoint_payload(module, old_provenance, 512, best_metric=None),
        checkpoint_dir / "last.pt",
    )
    old_hash = module.sha256_file(checkpoint_dir / "last.pt")
    producer = tmp_path / "producer.json"
    producer.write_text(
        json.dumps(
            {
                "schema": "detection_campaign_producer_freeze_v1",
                "git_commit": module.OLD_GIT_COMMIT,
                "source_sha256": old_prereg["code_sha256"],
            }
        ),
        encoding="utf-8",
    )
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            attach_profile_evidence(tmp_path, module, profile_payload(module, admitted=True))
        ),
        encoding="utf-8",
    )
    setattr(  # noqa: B010 - dynamically loaded module is intentionally patched.
        module, "AUTHORIZED_PROFILE_SHA256", module.sha256_file(profile)
    )
    authority = tmp_path / "source_authority.json"
    authority.write_text(
        json.dumps(
            {
                "schema": "eap_detection_failed_checkpoint_authority_v1",
                "status": "PASS_INDEPENDENT_REVIEW",
                "old_run": str(old_run.resolve()),
                "failure_receipt_sha256": module.sha256_file(old_run / "failure.json"),
                "preregistration_sha256": module.sha256_file(old_run / "preregistration.json"),
                "runtime_preregistration_sha256": module.sha256_file(old_prereg_path),
                "metrics_jsonl_sha256": module.sha256_file(old_run / "metrics.jsonl"),
                "durable_last_checkpoint_sha256": old_hash,
                "durable_update": 512,
                "last_logged_update": 725,
                "selected_checkpoint": {
                    "status": "NOT_AVAILABLE_NO_SCHEDULED_SELECTION",
                    "sha256": None,
                    "best_metric": None,
                },
                "independent_reviewer": "fixture-reviewer",
            }
        ),
        encoding="utf-8",
    )
    setattr(  # noqa: B010 - dynamically loaded module is intentionally patched.
        module,
        "AUTHORIZED_FAILED_SOURCE_RECEIPT_SHA256",
        module.sha256_file(authority),
    )
    output = tmp_path / "migrated"
    with pytest.raises(ValueError, match="leave the allowlist"):
        module.migrate(
            old_run=old_run,
            producer_receipt_path=producer,
            new_prereg_path=new_prereg_path,
            new_code_root=new_code,
            new_trainer_path=new_trainer,
            stream_profile_path=profile,
            source_checkpoint_receipt_path=authority,
            output=output,
        )
    assert module.sha256_file(checkpoint_dir / "last.pt") == old_hash
    assert not output.exists()


def test_profile_gate_blocks_migration_before_output_creation(tmp_path: Path) -> None:
    module = load_script()
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            attach_profile_evidence(tmp_path, module, profile_payload(module, admitted=False))
        ),
        encoding="utf-8",
    )
    setattr(  # noqa: B010 - dynamically loaded module is intentionally patched.
        module, "AUTHORIZED_PROFILE_SHA256", module.sha256_file(profile)
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="profile has not passed"):
        module.migrate(
            old_run=tmp_path / "absent",
            producer_receipt_path=tmp_path / "absent.json",
            new_prereg_path=tmp_path / "absent-new.json",
            new_code_root=tmp_path / "absent-code",
            new_trainer_path=tmp_path / "absent-trainer.py",
            stream_profile_path=profile,
            source_checkpoint_receipt_path=tmp_path / "absent-authority.json",
            output=output,
        )
    assert not output.exists()


def test_profile_gate_rejects_tampered_evidence(tmp_path: Path) -> None:
    module = load_script()
    profile_value = attach_profile_evidence(
        tmp_path, module, profile_payload(module, admitted=True)
    )
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(profile_value), encoding="utf-8")
    setattr(  # noqa: B010 - dynamically loaded module is intentionally patched.
        module, "AUTHORIZED_PROFILE_SHA256", module.sha256_file(profile)
    )
    evidence_path = tmp_path / profile_value["evidence"]["stream_worker_4096"]["path"]
    evidence_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence bytes differ"):
        module._verify_profile(profile)


@pytest.mark.parametrize(
    "field,required_from",
    [
        (
            "observed_prestart_physical_free_ram_bytes",
            lambda value: (
                value["projected_full_total_physical_ram_bytes"]
                + value["physical_safety_margin_bytes"]
            ),
        ),
        (
            "observed_runtime_min_physical_free_ram_bytes",
            lambda value: (
                value["projected_full_evaluator_peak_rss_bytes"]
                + value["physical_safety_margin_bytes"]
            ),
        ),
    ],
)
def test_profile_gate_rejects_each_physical_memory_scope(
    tmp_path: Path, field: str, required_from
) -> None:
    module = load_script()
    profile_value = profile_payload(module, admitted=True)
    profile_value[field] = required_from(profile_value) - 1
    projection_fields = {
        key: profile_value[key]
        for key in (
            "streaming_rss_observed_sample_count",
            "full_population_sample_count",
            "projection_multiplier",
            "measured_evaluator_peak_rss_bytes",
            "evaluator_measurement_excludes_trainer_model",
            "fixed_trainer_model_cache_rss_bytes",
            "projected_full_evaluator_peak_rss_bytes",
            "projected_full_total_physical_ram_bytes",
            "observed_prestart_physical_free_ram_bytes",
            "observed_runtime_min_physical_free_ram_bytes",
            "physical_safety_margin_bytes",
            "memory_basis",
            "projection_is_measurement",
            "admission_mode",
        )
    }
    profile_value["reviewed_projection_sha256"] = module.canonical_sha256(projection_fields)
    attach_profile_evidence(tmp_path, module, profile_value)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(profile_value), encoding="utf-8")
    setattr(  # noqa: B010 - dynamically loaded module is intentionally patched.
        module, "AUTHORIZED_PROFILE_SHA256", module.sha256_file(profile)
    )
    with pytest.raises(ValueError, match="memory arithmetic failed admission"):
        module._verify_profile(profile)


def test_source_authority_rejects_checkpoint_changed_after_review(tmp_path: Path) -> None:
    module = load_script()
    old_run = tmp_path / "failed"
    old_run.mkdir()
    for name, value in (
        ("failure.json", {"status": "FAILED"}),
        ("preregistration.json", {"parent_run_chain": []}),
        ("runtime_preregistration.json", {}),
        ("metrics.jsonl", None),
    ):
        path = old_run / name
        path.write_text('{"update": 1}\n' if value is None else json.dumps(value), encoding="utf-8")
    checkpoint = old_run / "checkpoints/last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"reviewed checkpoint")
    source_lineage = {
        "failure_receipt": {"sha256": module.sha256_file(old_run / "failure.json")},
        "preregistration_sha256": module.sha256_file(old_run / "preregistration.json"),
        "metrics_jsonl_sha256": module.sha256_file(old_run / "metrics.jsonl"),
        "last_logged_update": 1,
    }
    authority_value = {
        "schema": "eap_detection_failed_checkpoint_authority_v1",
        "status": "PASS_INDEPENDENT_REVIEW",
        "old_run": str(old_run.resolve()),
        "failure_receipt_sha256": source_lineage["failure_receipt"]["sha256"],
        "preregistration_sha256": source_lineage["preregistration_sha256"],
        "runtime_preregistration_sha256": module.sha256_file(
            old_run / "runtime_preregistration.json"
        ),
        "metrics_jsonl_sha256": source_lineage["metrics_jsonl_sha256"],
        "durable_last_checkpoint_sha256": module.sha256_file(checkpoint),
        "durable_update": 1,
        "last_logged_update": 1,
        "selected_checkpoint": {
            "status": "NOT_AVAILABLE_NO_SCHEDULED_SELECTION",
            "sha256": None,
            "best_metric": None,
        },
        "independent_reviewer": "fixture-reviewer",
    }
    authority = tmp_path / "authority.json"
    authority.write_text(json.dumps(authority_value), encoding="utf-8")
    setattr(  # noqa: B010 - dynamically loaded module is intentionally patched.
        module,
        "AUTHORIZED_FAILED_SOURCE_RECEIPT_SHA256",
        module.sha256_file(authority),
    )
    checkpoint.write_bytes(b"tampered checkpoint")
    with pytest.raises(ValueError, match="differs from source bytes"):
        module._verify_source_checkpoint_authority(
            authority,
            old_run=old_run,
            last_source=checkpoint,
            selected_source=old_run / "checkpoints/selected.pt",
            durable_update=1,
            best_metric=None,
            source_lineage=source_lineage,
        )
