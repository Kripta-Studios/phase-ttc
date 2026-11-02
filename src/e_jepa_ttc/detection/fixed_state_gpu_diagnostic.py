"""Contracts for a reset-per-trial GPU reader-equivalence diagnostic.

This module is intentionally CPU-only.  It seals the batches and comparison
policy before a separate reviewed executor may import torch or reserve a GPU.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from e_jepa_ttc.detection.campaign_training import canonical_sha256, sha256_file

SCHEMA = "eap_detection_fixed_state_gpu_diagnostic_v1"
RELATIVE_BATCHES = (1, 6, 11, 16)
CONTROL_REPEATS = 5


def build_plan(
    *,
    old_metrics_path: Path,
    checkpoint_path: Path,
    selected_path: Path,
    original_producer: Path,
    optimized_producer: Path,
    original_runtime_preregistration: Path,
    optimized_runtime_preregistration: Path,
    performance_config: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Build a self-hashed, result-blind diagnostic plan from old-A metrics."""
    rows = [
        json.loads(line)
        for line in old_metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != 16:
        raise ValueError("The frozen control trace must contain exactly 16 updates")
    batches = []
    for relative in RELATIVE_BATCHES:
        row = rows[relative - 1]
        tokens = row.get("sample_tokens")
        if not isinstance(tokens, list) or len(tokens) != 4 or len(set(tokens)) != 4:
            raise ValueError("Each fixed batch must contain four unique sample tokens")
        batches.append(
            {
                "relative_update": relative,
                "historical_update": row["update"],
                "sample_tokens": tokens,
                "learning_rates": row["learning_rates"],
            }
        )
    raw = {
        "schema": SCHEMA,
        "status": "PREREGISTERED_NOT_EXECUTED",
        "claim_scope": "SINGLE_UPDATE_READER_EQUIVALENCE_ONLY",
        "prior_16_update_admission_status": "FAIL_UNCHANGED",
        "selection_reason": "FOUR_ORDERED_COVERAGE_POINTS_INCLUDING_BOTH_ENDPOINTS",
        "relative_batches": list(RELATIVE_BATCHES),
        "control_repeats_per_batch": CONTROL_REPEATS,
        "candidate_repeats_per_batch": CONTROL_REPEATS,
        "restore_before_every_trial": [
            "model",
            "optimizer",
            "scheduler",
            "sampler",
            "python_rng",
            "numpy_rng",
            "torch_cpu_rng",
            "torch_cuda_rng_all",
        ],
        "cpu_parity": {
            "raw_member_sha256_exact": True,
            "token_order_exact": True,
            "prepared_tensor_bitwise_exact": True,
            "targets_and_masks_bitwise_exact": True,
        },
        "pre_backward_parity": {
            "model_output_tensor_bitwise_exact": True,
            "matcher_assignments_exact": True,
            "loss_keys_exact": True,
            "loss_tensors_bitwise_exact": True,
            "on_control_nonexact": "INCONCLUSIVE_DO_NOT_RUN_CANDIDATE",
        },
        "control_envelope_policy": {
            "source": "ORIGINAL_REPEATS_ONLY",
            "seal_before_candidate": True,
            "tensor_distances": {
                "d2": "norm2(a-b)/max(norm2(a),norm2(b),1e-12)",
                "dinf": "norminf(a-b)/max(norminf(a),norminf(b),1e-12)",
                "scalar": "abs(a-b)/max(abs(a),abs(b),1e-12)",
            },
            "control_radius": "MAX_PAIRWISE_DISTANCE_ACROSS_FIVE_ORIGINAL_RESETS",
            "candidate_gate": "min(candidate_hard_cap,max(floor,2*control_radius))",
            "candidate_multiplier": 2.0,
            "normalized_l2_floor": 1e-7,
            "normalized_max_abs_floor": 1e-7,
            "control_normalized_l2_hard_cap": 1e-4,
            "control_normalized_max_abs_hard_cap": 1e-4,
            "candidate_normalized_l2_hard_cap": 2e-4,
            "candidate_normalized_max_abs_hard_cap": 2e-4,
            "set_distance": "SYMMETRIC_HAUSDORFF_OVER_FIVE_RESET_TRIALS",
            "fields": [
                "gradient_per_parameter_l2",
                "gradient_per_parameter_max_abs",
                "parameter_step_delta_l2",
                "parameter_step_delta_max_abs",
                "gradient_post_clip_per_parameter_l2",
                "gradient_post_clip_per_parameter_max_abs",
                "clip_coefficient",
                "adamw_exp_avg_step_delta",
                "adamw_exp_avg_sq_step_delta",
                "adamw_step_delta",
                "model_buffer_step_delta",
                "scheduler_step_delta",
                "post_trial_rng_state",
                "gradient_norm_before_clip",
            ],
            "no_post_candidate_recalibration": True,
            "on_control_cap_exceeded": "INCONCLUSIVE_DO_NOT_RUN_CANDIDATE",
            "on_candidate_cap_exceeded": "FAIL_THIS_DIAGNOSTIC_ONLY",
        },
        "batches": batches,
        "artifacts": {
            name: {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}
            for name, path in {
                "old_metrics": old_metrics_path,
                "checkpoint": checkpoint_path,
                "selected": selected_path,
                "original_producer": original_producer,
                "optimized_producer": optimized_producer,
                "original_runtime_preregistration": original_runtime_preregistration,
                "optimized_runtime_preregistration": optimized_runtime_preregistration,
                "performance_config": performance_config,
                "config": config_path,
            }.items()
        },
        "execution_precondition": "R0_ORIGINAL_UPDATE_8192_COMPLETE_AND_SCIENCE_REVIEW_SHIP",
        "resources": {
            "num_cpu_threads": 1,
            "minimum_free_output_bytes": 85899345920,
            "trial_tensor_storage": "WRITE_ONCE_CPU_TENSORS_RELEASED_AFTER_EACH_TRIAL",
            "maximum_live_trial_tensor_sets": 2,
        },
        "maximum_claim": (
            "EXACT_INPUTS_AND_OBSERVED_SINGLE_STEP_EQUIVALENCE_FOR_FOUR_FIXED_BATCHES;"
            "NOT_GLOBAL_MATHEMATICAL_PARITY_OR_TRAJECTORY_EQUIVALENCE"
        ),
    }
    repository_root = Path(__file__).resolve().parents[3]
    for name, path in {
        "diagnostic_contract": Path(__file__),
        "trial_executor": Path(__file__).with_name("fixed_state_trial_executor.py"),
        "diagnostic_runner": repository_root / "scripts/run_fixed_state_gpu_diagnostic.py",
        "plan_builder": repository_root / "scripts/prepare_fixed_state_gpu_diagnostic.py",
        "campaign_training": repository_root / "src/e_jepa_ttc/detection/campaign_training.py",
        "indexed_rgb_data": repository_root / "src/e_jepa_ttc/detection/indexed_rgb_data.py",
        "indexed_rgb_store": repository_root / "src/e_jepa_ttc/detection/indexed_rgb_store.py",
        "optimized_campaign": repository_root / "src/e_jepa_ttc/detection/optimized_campaign.py",
        "eap_rgb_data": repository_root / "src/e_jepa_ttc/detection/eap_rgb_data.py",
        "monodgp_eap_adapter": repository_root / "src/e_jepa_ttc/detection/monodgp_eap_adapter.py",
    }.items():
        raw["artifacts"][name] = {
            "path": str(path.resolve(strict=True)),
            "sha256": sha256_file(path),
        }
    return raw | {"plan_sha256": canonical_sha256(raw)}


def validate_plan(plan: Mapping[str, Any]) -> None:
    """Fail closed on policy drift or artifact mutation without importing torch."""
    if plan.get("schema") != SCHEMA or plan.get("status") != "PREREGISTERED_NOT_EXECUTED":
        raise ValueError("Unknown fixed-state diagnostic plan")
    unhashed = dict(plan)
    claimed = unhashed.pop("plan_sha256", None)
    if claimed != canonical_sha256(unhashed):
        raise ValueError("Fixed-state diagnostic plan hash differs")
    if tuple(plan.get("relative_batches", ())) != RELATIVE_BATCHES:
        raise ValueError("Fixed batches differ from preregistration")
    if plan.get("control_repeats_per_batch") != CONTROL_REPEATS:
        raise ValueError("Control repeat count differs")
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TypeError("Diagnostic artifacts are missing")
    for artifact in artifacts.values():
        if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("Diagnostic artifact changed after preregistration")


__all__ = ["CONTROL_REPEATS", "RELATIVE_BATCHES", "SCHEMA", "build_plan", "validate_plan"]
