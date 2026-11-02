"""Execute the preregistered reset-per-trial reader diagnostic after R0 completes."""

# ruff: noqa: E402 -- direct execution from the worktree is intentional.

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from e_jepa_ttc.detection.campaign_training import (
    ResumablePermutationSampler,
    UpdateLRScheduler,
    load_campaign_config,
    optimizer_parameter_groups,
    restore_rng_state,
    sha256_file,
    validate_campaign_manifest,
    validate_manifest_against_metadata,
)
from e_jepa_ttc.detection.fixed_state_gpu_diagnostic import validate_plan
from e_jepa_ttc.detection.fixed_state_trial_executor import (
    capture_state,
    exact_training_operation,
    restore_state,
    verify_frozen_operation_source,
)
from e_jepa_ttc.detection.indexed_rgb_store import IndexedRGBStore
from e_jepa_ttc.detection.optimized_campaign import PerformanceConfig


def _load_frozen_producer(path: Path) -> Any:
    frozen_root = path.resolve().parents[1]
    sys.path.insert(0, str(frozen_root / "scripts"))
    sys.path.insert(0, str(frozen_root / "src"))
    sys.modules.pop("smoke_monodgp_portable", None)
    spec = importlib.util.spec_from_file_location("fixed_state_frozen_producer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen producer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_indexed_store(config: Any, performance: PerformanceConfig) -> IndexedRGBStore:
    manifest = validate_campaign_manifest(config)
    metadata = validate_manifest_against_metadata(manifest, config)
    admitted = dict(metadata["media_admission"])
    receipt = json.loads(Path(admitted["receipt_path"]).read_text(encoding="utf-8"))
    final_media = receipt["final_media"]
    ownership = {
        token: (
            str(row["rgb_shard_path"]),
            str(row["rgb_member_path"]),
            str(row["member_sha256"]),
        )
        for token, row in final_media.items()
    }
    inventory = json.loads(
        (ROOT / "data/manifests/eap_detection_public_files_v1.json").read_text(encoding="utf-8")
    )["files"]
    shard_hashes = {
        path: str(inventory[path]["sha256"])
        for path in {contract[0] for contract in ownership.values()}
    }
    return IndexedRGBStore(
        config.data.root,
        ownership,
        shard_hashes,
        max_open_handles=performance.max_open_handles,
    )


def _write_once(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _exclusive_gpu_preflight() -> list[dict[str, str | int]]:
    completed = subprocess.run(
        ["nvidia-smi", "-q", "-x"], check=True, capture_output=True, text=True
    )
    root = ET.fromstring(completed.stdout)
    processes = []
    for item in root.findall(".//process_info"):
        row: dict[str, str | int] = {
            "pid": int(item.findtext("pid") or "-1"),
            "type": item.findtext("type") or "UNKNOWN",
            "name": item.findtext("process_name") or "",
        }
        processes.append(row)
    competitors = [row for row in processes if row["type"] == "C"]
    if competitors:
        raise RuntimeError(f"Competing compute processes are active: {competitors}")
    return processes


def _validate_full_evaluation(path: Path, config: Any, ordinary: Any) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics_v2") if isinstance(payload, dict) else None
    protocol = ordinary.load_protocol(config.evaluation.protocol)
    full_tokens = json.loads(Path(config.data.full_selection_tokens).read_text(encoding="utf-8"))[
        "sample_tokens"
    ]
    if (
        not isinstance(metrics, dict)
        or metrics.get("schema") != "phase_ttc_detection_evaluation_result_v2"
        or metrics.get("protocol_sha256") != protocol.sha256
        or metrics.get("official_eap_score") is not False
        or metrics.get("sample_count") != len(full_tokens)
        or not isinstance(payload.get("predictions"), dict)
        or set(payload["predictions"]) != set(full_tokens)
    ):
        raise ValueError("R0 full-selection evaluation contract differs")
    return payload


def _tensor_sha(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _flatten_tensors(value: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().cpu()}
    result: dict[str, torch.Tensor] = {}
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            result.update(_flatten_tensors(value[key], f"{prefix}/{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(_flatten_tensors(item, f"{prefix}/{index}"))
    return result


def _fingerprint(value: Any) -> dict[str, str]:
    return {key: _tensor_sha(tensor) for key, tensor in _flatten_tensors(value).items()}


def _state_digest(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return {
                "tensor_sha256": _tensor_sha(item),
                "dtype": str(item.dtype),
                "shape": list(item.shape),
            }
        if isinstance(item, np.ndarray):
            return {
                "numpy_sha256": hashlib.sha256(item.tobytes()).hexdigest(),
                "dtype": str(item.dtype),
                "shape": list(item.shape),
            }
        if isinstance(item, dict):
            return {str(key): normalize(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, (list, tuple)):
            return [normalize(part) for part in item]
        return item

    return hashlib.sha256(
        json.dumps(normalize(value), sort_keys=True, default=repr).encode()
    ).hexdigest()


def _normalized(a: torch.Tensor, b: torch.Tensor, *, infinity: bool) -> float:
    left, right = a.double().reshape(-1), b.double().reshape(-1)
    if infinity:
        numerator = float((left - right).abs().max()) if left.numel() else 0.0
        denominator = max(
            float(left.abs().max()) if left.numel() else 0.0,
            float(right.abs().max()) if right.numel() else 0.0,
            1e-12,
        )
    else:
        numerator = float(torch.linalg.vector_norm(left - right))
        denominator = max(
            float(torch.linalg.vector_norm(left)),
            float(torch.linalg.vector_norm(right)),
            1e-12,
        )
    return numerator / denominator


def _trial_vectors(result: dict[str, Any]) -> dict[str, torch.Tensor]:
    vectors: dict[str, torch.Tensor] = {}
    for group in (
        "gradients_before_clip",
        "gradients_after_clip",
        "parameter_step_deltas",
        "buffer_step_deltas",
    ):
        for name, tensor in result[group].items():
            vectors[f"{group}/{name}"] = tensor.detach().cpu()
    vectors["gradient_norm_before_clip"] = result["gradient_norm_before_clip"].reshape(1).cpu()
    vectors["clip_coefficient"] = result["clip_coefficient"].reshape(1).cpu()
    for key, tensor in result["optimizer_step_deltas"].items():
        vectors[f"optimizer_delta/{key}"] = tensor.detach().cpu()
    return vectors


def _trial_receipt(trial: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in trial.items() if key != "vectors"} | {
        "vectors": {
            key: {
                "sha256": _tensor_sha(tensor),
                "l2": float(torch.linalg.vector_norm(tensor.double().reshape(-1))),
                "max_abs": (float(tensor.double().abs().max()) if tensor.numel() else 0.0),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            for key, tensor in sorted(trial["vectors"].items())
        }
    }


def _cpu_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_cpu_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _cpu_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _load_trial_vectors(trial: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "vectors" in trial:
        return trial["vectors"]
    return torch.load(trial["vector_path"], map_location="cpu", weights_only=True)


def _same_vector_keys(trials: list[dict[str, Any]]) -> bool:
    if not trials:
        return False
    expected = set(_load_trial_vectors(trials[0]))
    return all(set(_load_trial_vectors(trial)) == expected for trial in trials[1:])


def _pairwise_radius(trials: list[dict[str, Any]], key: str, infinity: bool) -> float:
    radius = 0.0
    for index in range(len(trials)):
        left = _load_trial_vectors(trials[index])[key]
        for other in range(index):
            right = _load_trial_vectors(trials[other])[key]
            radius = max(radius, _normalized(left, right, infinity=infinity))
            del right
        del left
    return radius


def _hausdorff(
    control: list[dict[str, Any]], candidate: list[dict[str, Any]], key: str, infinity: bool
) -> float:
    def directed(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
        maximum = 0.0
        for left_trial in left:
            left_tensor = _load_trial_vectors(left_trial)[key]
            nearest = math.inf
            for right_trial in right:
                right_tensor = _load_trial_vectors(right_trial)[key]
                nearest = min(nearest, _normalized(left_tensor, right_tensor, infinity=infinity))
                del right_tensor
            maximum = max(maximum, nearest)
            del left_tensor
        return maximum

    return max(directed(control, candidate), directed(candidate, control))


def run(plan_path: Path, performance_path: Path, r0_summary: Path, output: Path) -> dict[str, Any]:
    """Run control-first, seal its envelope, then run the indexed candidate."""
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    if (
        performance_path.resolve()
        != Path(plan["artifacts"]["performance_config"]["path"]).resolve()
    ):
        raise ValueError("Performance config path differs from the fixed plan")
    ordinary = _load_frozen_producer(Path(plan["artifacts"]["original_producer"]["path"]))
    summary = json.loads(r0_summary.read_text(encoding="utf-8"))
    completed_status = summary.get("status") in {"COMPLETE", "COMPLETED"}
    bounded_complete = (
        summary.get("status") == "BUDGET_STOPPED"
        and summary.get("external_execution_stop_requested") == 8192
        and summary.get("stop_reason") == "external_execution_stop_after_completed_update"
    )
    if summary.get("final_update") != 8192 or not (completed_status or bounded_complete):
        raise ValueError("Original R0 must complete update 8192 before this diagnostic")
    full_evaluation = r0_summary.parent / "evaluation/full_selection_update_008192.json"
    if not full_evaluation.is_file():
        raise ValueError("Original R0 lacks the required full-selection update-8192 evaluation")
    if shutil.disk_usage(output.parent).free < plan["resources"]["minimum_free_output_bytes"]:
        raise RuntimeError("Output volume lacks preregistered trial-tensor headroom")
    performance, performance_receipt = PerformanceConfig.load(performance_path)
    reader_path = ROOT / "src/e_jepa_ttc/detection/indexed_rgb_data.py"
    store_path = ROOT / "src/e_jepa_ttc/detection/indexed_rgb_store.py"
    if sha256_file(reader_path) != performance.expected_indexed_reader_sha256:
        raise ValueError("Indexed reader differs from performance preregistration")
    if sha256_file(store_path) != performance.expected_indexed_store_sha256:
        raise ValueError("Indexed store differs from performance preregistration")
    config = load_campaign_config(Path(plan["artifacts"]["config"]["path"]))
    if (
        summary.get("schema") != "eap_detection_campaign_run_v1"
        or summary.get("config_sha256") != config.sha256
        or summary.get("evaluation_protocol_sha256")
        != ordinary.load_protocol(config.evaluation.protocol).sha256
        or summary.get("runtime_provenance", {}).get("runtime_preregistration_sha256")
        != plan["artifacts"]["original_runtime_preregistration"]["sha256"]
    ):
        raise ValueError("Completed R0 summary lineage differs from the fixed plan")
    _validate_full_evaluation(full_evaluation, config, ordinary)
    proof = verify_frozen_operation_source(plan["artifacts"]["original_producer"]["path"])
    if proof["producer_sha256"] != plan["artifacts"]["original_producer"]["sha256"]:
        raise ValueError("Frozen producer differs from the AST proof")
    gpu_processes = _exclusive_gpu_preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required after CPU preflight")
    torch.set_num_threads(1)
    resource_receipt = ordinary.resource_admission(config)
    torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True)
    _write_once(
        output / "preflight.json",
        {
            "schema": "eap_detection_fixed_state_preflight_v1",
            "status": "PASS",
            "plan_sha256": plan["plan_sha256"],
            "plan_file_sha256": sha256_file(plan_path),
            "r0_summary_sha256": sha256_file(r0_summary),
            "r0_full_evaluation_sha256": sha256_file(full_evaluation),
            "performance_config": performance_receipt,
            "source_ast_proof": proof,
            "num_cpu_threads": torch.get_num_threads(),
            "resource_admission": resource_receipt,
            "gpu_processes_before_cuda_context": gpu_processes,
            "planned_resources": plan["resources"],
        },
    )
    ordinary.prepare_source(
        Path(config.model.source_root),
        Path(config.model.portable_root),
        output / "upstream.patch",
        predict_velocity=config.model.predict_velocity,
    )
    model, criterion, _ = ordinary.build_detector(
        Path(config.model.portable_root),
        config.model.group_num,
        predict_velocity=config.model.predict_velocity,
        velocity_loss_coef=config.model.velocity_loss_coef,
    )
    groups, _ = optimizer_parameter_groups(model, config.optimization)
    optimizer = torch.optim.AdamW(groups)
    scheduler = UpdateLRScheduler(
        optimizer,
        total_updates=config.optimization.updates,
        kind=config.optimization.scheduler,
        warmup_fraction=config.optimization.warmup_fraction,
        minimum_ratio=config.optimization.minimum_lr_ratio,
    )
    checkpoint = torch.load(
        plan["artifacts"]["checkpoint"]["path"], map_location="cpu", weights_only=True
    )
    if checkpoint.get("update") != 1024 or checkpoint.get("config_sha256") != config.sha256:
        raise ValueError("Diagnostic checkpoint differs from update-1024 campaign state")
    original_prereg_sha = plan["artifacts"]["original_runtime_preregistration"]["sha256"]
    if (
        checkpoint.get("runtime_provenance", {}).get("runtime_preregistration_sha256")
        != original_prereg_sha
    ):
        raise ValueError("Checkpoint differs from the bound original runtime")
    sampler = ResumablePermutationSampler(4096, config.seed)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    sampler.load_state_dict(checkpoint["sampler"])
    restore_rng_state(checkpoint["rng"])
    initial = capture_state(model, optimizer, scheduler, sampler)
    tokens = [token for batch in plan["batches"] for token in batch["sample_tokens"]]
    original_dataset = ordinary.open_campaign_dataset(config.data.root, tokens, config)
    store = _build_indexed_store(config, performance)
    indexed_dataset = store.dataset_view(
        tokens, max_label_sequences=config.data.label_cache_sequences
    )
    member_hashes = dict(checkpoint["data_member_sha256"])
    original_cache = ordinary.PreparedCache(original_dataset, config, copy.deepcopy(member_hashes))
    indexed_cache = ordinary.PreparedCache(indexed_dataset, config, copy.deepcopy(member_hashes))

    def prepared_rows(cache: Any, batch_index: int) -> list[Any]:
        return [cache.get(batch_index * 4 + offset) for offset in range(4)]

    def execute_trial(rows: list[Any], batch_spec: dict[str, Any]) -> dict[str, Any]:
        restore_state(initial, model, optimizer, scheduler, sampler)
        applied = scheduler.apply(int(batch_spec["historical_update"]))
        if applied != batch_spec["learning_rates"]:
            raise ValueError("Trial learning rate differs from the frozen trace")
        cuda_batch = ordinary.batch_to_cuda(rows, config, training=True)
        matcher_observations: list[Any] = []
        matcher = criterion.matcher
        original_matcher_forward = matcher.forward

        def observed_matcher(*args: Any, **kwargs: Any) -> Any:
            matched = original_matcher_forward(*args, **kwargs)
            matcher_observations.append(copy.deepcopy(matched))
            return matched

        matcher.forward = observed_matcher
        try:
            result = exact_training_operation(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                batch=cuda_batch,
                clip=config.optimization.gradient_clip_norm,
                finite_validator=ordinary.require_finite,
            )
        finally:
            matcher.forward = original_matcher_forward
        return {
            "output_sha256": _fingerprint(result["outputs"]),
            "loss_sha256": _fingerprint(result["losses"]),
            "matcher_sha256": _fingerprint(matcher_observations),
            "scheduler_state_sha256": _state_digest(scheduler.state_dict()),
            "post_rng_sha256": _state_digest(result["post_rng"]),
            "vectors": _trial_vectors(result),
        }

    def persist_trial(
        rows: list[Any], batch_spec: dict[str, Any], phase: str, batch: int, trial: int
    ) -> dict[str, Any]:
        value = execute_trial(rows, batch_spec)
        receipt = _trial_receipt(value)
        vector_path = output / "trial_tensors" / phase / f"batch_{batch:02d}_trial_{trial:02d}.pt"
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        if vector_path.exists():
            raise FileExistsError(vector_path)
        torch.save(value["vectors"], vector_path)
        persisted = {key: item for key, item in value.items() if key != "vectors"}
        persisted.update(
            vector_path=str(vector_path.resolve()),
            vector_sha256=sha256_file(vector_path),
            receipt=receipt,
        )
        del value
        torch.cuda.empty_cache()
        return persisted

    try:
        control: dict[int, list[dict[str, Any]]] = {}
        candidate: dict[int, list[dict[str, Any]]] = {}
        cpu_receipt = []
        for index, batch_spec in enumerate(plan["batches"]):
            ordinary_rows, indexed_rows = (
                prepared_rows(original_cache, index),
                prepared_rows(indexed_cache, index),
            )
            if [row[1].sample_token for row in ordinary_rows] != batch_spec["sample_tokens"]:
                raise ValueError("Original fixed batch order differs")
            if [row[1].member_sha256 for row in ordinary_rows] != [
                row[1].member_sha256 for row in indexed_rows
            ]:
                raise ValueError("Reader member hashes differ")
            if any(
                not _cpu_equal(
                    {
                        "image": left[0].image,
                        "calibration": left[0].calibration,
                        "image_size": left[0].image_size,
                        "target": left[0].target,
                    },
                    {
                        "image": right[0].image,
                        "calibration": right[0].calibration,
                        "image_size": right[0].image_size,
                        "target": right[0].target,
                    },
                )
                for left, right in zip(ordinary_rows, indexed_rows, strict=True)
            ):
                raise ValueError("Prepared CPU samples differ bitwise")
            cpu_receipt.append(
                {
                    "relative_update": batch_spec["relative_update"],
                    "sample_tokens": batch_spec["sample_tokens"],
                    "member_sha256": [row[1].member_sha256 for row in ordinary_rows],
                    "status": "PASS_BITWISE",
                }
            )
            control[index] = [
                persist_trial(ordinary_rows, batch_spec, "control", index, trial)
                for trial in range(5)
            ]
        _write_once(
            output / "control_trials.json",
            {
                "status": "CONTROL_COMPLETE_CANDIDATE_NOT_RUN",
                "trials": {
                    str(index): [trial["receipt"] for trial in trials]
                    for index, trials in control.items()
                },
            },
        )
        envelope: dict[str, Any] = {}
        for index, trials in control.items():
            if not _same_vector_keys(trials):
                result = {
                    "status": "INCONCLUSIVE_CONTROL_VECTOR_KEYS_DIFFER",
                    "batch": index,
                }
                _write_once(output / "result.json", result)
                return result
            if any(
                trial["output_sha256"] != trials[0]["output_sha256"]
                or trial["loss_sha256"] != trials[0]["loss_sha256"]
                or trial["matcher_sha256"] != trials[0]["matcher_sha256"]
                or trial["scheduler_state_sha256"] != trials[0]["scheduler_state_sha256"]
                or trial["post_rng_sha256"] != trials[0]["post_rng_sha256"]
                for trial in trials[1:]
            ):
                result = {
                    "status": "INCONCLUSIVE_CONTROL_PREBACKWARD_NONEXACT",
                    "batch": index,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
                _write_once(output / "result.json", result)
                return result
            keys = set(_load_trial_vectors(trials[0]))
            batch_envelope = {}
            for key in sorted(keys):
                d2 = _pairwise_radius(trials, key, False)
                dinf = _pairwise_radius(trials, key, True)
                if d2 > 1e-4 or dinf > 1e-4:
                    result = {
                        "status": "INCONCLUSIVE_CONTROL_CAP_EXCEEDED",
                        "batch": index,
                        "tensor": key,
                        "d2": d2,
                        "dinf": dinf,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    }
                    _write_once(output / "result.json", result)
                    return result
                batch_envelope[key] = {"d2_radius": d2, "dinf_radius": dinf}
            envelope[str(index)] = batch_envelope
        _write_once(
            output / "control_envelope.json",
            {
                "status": "SEALED_BEFORE_CANDIDATE",
                "plan_sha256": plan["plan_sha256"],
                "cpu_parity": cpu_receipt,
                "envelope": envelope,
            },
        )
        for index, batch_spec in enumerate(plan["batches"]):
            rows = prepared_rows(indexed_cache, index)
            candidate[index] = [
                persist_trial(rows, batch_spec, "candidate", index, trial) for trial in range(5)
            ]
        _write_once(
            output / "candidate_trials.json",
            {
                "status": "CANDIDATE_COMPLETE_UNSCORED",
                "trials": {
                    str(index): [trial["receipt"] for trial in trials]
                    for index, trials in candidate.items()
                },
            },
        )
        failures = []
        for index in control:
            if not _same_vector_keys(candidate[index]) or set(
                _load_trial_vectors(candidate[index][0])
            ) != set(_load_trial_vectors(control[index][0])):
                failures.append({"batch": index, "reason": "VECTOR_KEY_SET_DIFF"})
                continue
            if any(
                trial["output_sha256"] != control[index][0]["output_sha256"]
                or trial["loss_sha256"] != control[index][0]["loss_sha256"]
                or trial["matcher_sha256"] != control[index][0]["matcher_sha256"]
                or trial["scheduler_state_sha256"] != control[index][0]["scheduler_state_sha256"]
                or trial["post_rng_sha256"] != control[index][0]["post_rng_sha256"]
                for trial in candidate[index]
            ):
                failures.append({"batch": index, "reason": "PREBACKWARD_NONEXACT"})
                continue
            for key, radii in envelope[str(index)].items():
                for metric, infinity, cap in (("d2", False, 2e-4), ("dinf", True, 2e-4)):
                    observed = _hausdorff(control[index], candidate[index], key, infinity)
                    allowed = min(cap, max(1e-7, 2 * radii[f"{metric}_radius"]))
                    if observed > allowed:
                        failures.append(
                            {
                                "batch": index,
                                "tensor": key,
                                "metric": metric,
                                "observed": observed,
                                "allowed": allowed,
                            }
                        )
        result = {
            "schema": "eap_detection_fixed_state_result_v1",
            "status": "FAIL_THIS_DIAGNOSTIC_ONLY" if failures else "PASS_OBSERVED_SINGLE_STEP_ONLY",
            "prior_16_update_admission_status": "FAIL_UNCHANGED",
            "global_or_trajectory_equivalence_claimed": False,
            "failures": failures,
            "control_envelope_sha256": sha256_file(output / "control_envelope.json"),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        _write_once(output / "result.json", result)
        return result
    finally:
        original_dataset.close()
        indexed_dataset.close()
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--performance-config", type=Path, required=True)
    parser.add_argument("--r0-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-gpu", action="store_true", required=True)
    args = parser.parse_args()
    try:
        run(args.plan, args.performance_config, args.r0_summary, args.output)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = args.output / "failure.json"
        if not failure.exists():
            _write_once(
                failure,
                {
                    "schema": "eap_detection_fixed_state_failure_v1",
                    "status": "FAILED_NO_RETRY",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "plan_sha256": (
                        json.loads(args.plan.read_text(encoding="utf-8")).get("plan_sha256")
                        if args.plan.is_file()
                        else None
                    ),
                },
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
