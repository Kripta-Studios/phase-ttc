"""Target-free, fold-bound inference after the complete initial-stage endpoint barrier."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import verify_initial_barrier
from e_jepa_ttc.data.event_context128 import RateNormalization, four_contexts
from e_jepa_ttc.data.event_context128_reader import CountReader
from e_jepa_ttc.models.projective_phase_network import ProjectivePhaseNetwork
from e_jepa_ttc.training.stage70_fixed_fit import deterministic_runtime
from e_jepa_ttc.training.stage72_augmentation import time_derangement
from e_jepa_ttc.training.stage72_jobs import FitJob, require_scientific_lock
from e_jepa_ttc.training.stage73_factorial import treatment


def validate_outer_view(
    job: FitJob, reader: CountReader, tokens: tuple[str, ...], lock: dict[str, Any]
) -> None:
    """Require exact frozen original-fold rows and every held-out acquisition exclusion."""
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError("outer inference requires unique nonempty ordered tokens")
    if reader.identity != lock.get("outer_count_cache_sha256"):
        raise ValueError("outer inference cache differs from the scientific lock")
    expected = lock.get("outer_tokens_by_fold", {}).get(str(job.plan.outer_fold))
    if list(tokens) != expected:
        raise ValueError("outer inference token order differs from the frozen fold")
    rows = reader.metadata
    fold_rows = rows.loc[rows.outer_fold == job.plan.outer_fold]
    if tuple(fold_rows.sample_token.astype(str)) != tokens:
        raise ValueError("outer inference must cover the entire original fold exactly once")
    mapping = lock.get("sequence_to_group", {})
    sequences = set(fold_rows.sequence_id.astype(str))
    if not sequences <= set(mapping):
        raise ValueError("outer inference sequence has no frozen acquisition role")
    groups = {mapping[sequence] for sequence in sequences}
    if not groups <= set(job.excluded_groups) or groups & set(job.allowed_train_groups):
        raise ValueError("outer acquisition was not excluded from its encoder and head producer")


def _publish_arrays(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Close and flush a numeric-only shard before hashing; never overwrite an endpoint."""
    if path.exists():
        raise ValueError("inference shard already exists without a verified publication receipt")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("inference arrays cannot contain pickle or object payloads")
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, allow_pickle=False, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return binding(path)


def verify_inference_shards(
    shards: list[dict[str, Any]], tokens: tuple[str, ...], identity_sha: str, output: Path
) -> None:
    """Reject partial, reordered, duplicated or foreign shards even in a re-signed receipt."""
    if len(shards) != (len(tokens) + 31) // 32:
        raise ValueError("completed inference omitted prescribed row shards")
    for number, row in enumerate(shards):
        unsigned = {key: value for key, value in row.items() if key != "artifact_sha256"}
        first = number * 32
        expected_path = output.absolute() / f"inference_{number:05d}.npz"
        if (
            object_digest(unsigned) != row.get("artifact_sha256")
            or row.get("identity_sha256") != identity_sha
            or row.get("first_row") != first
            or row.get("tokens") != list(tokens[first : first + 32])
            or Path(row["arrays"]["path"]).resolve() != expected_path
        ):
            raise ValueError("inference shard order, source or owned path changed")
        verify(row["arrays"])


def infer_initial_endpoint(
    job: FitJob,
    *,
    barrier_path: Path,
    lock_path: Path,
    reader: CountReader,
    tokens: tuple[str, ...],
    normalization: RateNormalization,
    output: Path,
    owner: ResourceOwner,
) -> dict[str, Any]:
    """Infer immutable 32-row shards with batch one and exact token-fixed TIMEPERM.

    The caller holds the single GPU writer and reserves this endpoint's entire
    output budget. No targets, old expert predictions, augmentations or fitted
    calibration enter this path. Evaluation joins targets only after publication.
    """
    owner.check()
    barrier = verify_initial_barrier(
        barrier_path, expected_stage=job.plan.stage, expected_lock=job.training_lock_sha256
    )
    lock = require_scientific_lock(lock_path, job.training_lock_sha256)
    validate_outer_view(job, reader, tokens, lock)
    matches = [row for row in barrier["endpoints"] if row["job_sha256"] == job.identity]
    if len(matches) != 1:
        raise ValueError("inference job is absent from the complete stage barrier")
    if (
        object_digest(asdict(normalization)) != job.normalization_sha256
        or normalization.training_view_sha256 != job.training_view_sha256
    ):
        raise ValueError("outer inference normalizer differs from its TRAIN producer")
    expected_output = Path(lock["outer_inference_root"]) / job.plan.job_id
    if output.absolute() != expected_output or output.resolve() != expected_output:
        raise ValueError("outer inference output escaped its frozen owned directory")
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "job_sha256": job.identity,
        "stage_barrier": binding(barrier_path),
        "weights": matches[0]["inference_weights"],
        "count_cache_sha256": reader.identity,
        "ordered_tokens": list(tokens),
        "normalization_sha256": job.normalization_sha256,
        "precision": "float32_TF32_off",
        "batch": 1,
        "shard_rows": 32,
        "targets_read": False,
        "optimizer_updates": 0,
    }
    identity_sha = object_digest(identity)
    receipt_path = output / "INFERENCE_COMPLETE.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        unsigned = {key: value for key, value in receipt.items() if key != "artifact_sha256"}
        if (
            object_digest(unsigned) != receipt.get("artifact_sha256")
            or receipt.get("identity_sha256") != identity_sha
        ):
            raise ValueError("completed inference identity changed")
        if receipt.get("identity") != identity or receipt.get("rows") != len(tokens):
            raise ValueError("completed inference recipe or full row count changed")
        verify_inference_shards(receipt["shards"], tokens, identity_sha, output)
        return receipt
    if not torch.cuda.is_available():
        raise RuntimeError("scientific inference requires the registered CUDA runtime")
    deterministic_runtime(job.plan.seed)
    torch.set_num_threads(2)
    model = ProjectivePhaseNetwork(checkpoint_sweep=False)
    model.load_state_dict(
        torch.load(verify(matches[0]["inference_weights"]), map_location="cpu", weights_only=True),
        strict=True,
    )
    model.cuda().eval()
    recipe = treatment(job.plan.treatment_name)
    shards = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            for first in range(0, len(tokens), 32):
                owner.check()
                selected = tokens[first : first + 32]
                shard_path = output / f"inference_{first // 32:05d}.npz"
                shard_receipt = shard_path.with_suffix(".json")
                if shard_receipt.exists():
                    record = json.loads(shard_receipt.read_text(encoding="utf-8"))
                    signature = record.pop("artifact_sha256")
                    if (
                        object_digest(record) != signature
                        or record.get("identity_sha256") != identity_sha
                        or record.get("first_row") != first
                        or record.get("tokens") != list(selected)
                        or record["arrays"]["path"] != str(shard_path.resolve())
                    ):
                        raise ValueError(
                            "inference resume shard belongs to another row or producer"
                        )
                    verify(record["arrays"])
                    shards.append({**record, "artifact_sha256": signature})
                    continue
                outputs: dict[str, list[np.ndarray]] = {}
                for token in selected:
                    observation = reader.observation(token)
                    sample = four_contexts(
                        observation.counts,
                        observation.durations_s,
                        observation.support,
                        observation.midpoint_twice_relative_us,
                        normalization,
                    )
                    permutation = None
                    if recipe.architecture == "PPS-TIMEPERM":
                        permutation = torch.tensor(
                            [time_derangement(token, job.plan.seed)],
                            device="cuda",
                            dtype=torch.int64,
                        )
                    result = model(
                        sample.contexts[None].cuda(),
                        sample.times[None].cuda(),
                        arm=recipe.architecture,
                        permutation=permutation,
                    )
                    if torch.cuda.max_memory_allocated() >= (
                        0.9 * torch.cuda.get_device_properties(0).total_memory
                    ):
                        raise RuntimeError("RESOURCE_BLOCKED: inference reached 90 percent VRAM")
                    for key, value in result.items():
                        outputs.setdefault(key, []).append(value.cpu().numpy().copy())
                arrays = {key: np.concatenate(values) for key, values in outputs.items()}
                arrays["phase_edges"] = model.edges.cpu().numpy().copy()
                record = {
                    "identity_sha256": identity_sha,
                    "first_row": first,
                    "tokens": list(selected),
                    "arrays": _publish_arrays(shard_path, arrays),
                }
                record["artifact_sha256"] = object_digest(record)
                atomic_json(shard_receipt, record)
                shards.append(record)
                atomic_json(
                    output / "INFERENCE_PROGRESS.json",
                    {
                        "identity_sha256": identity_sha,
                        "completed_rows": first + len(selected),
                        "total_rows": len(tokens),
                        "shards": len(shards),
                        "optimizer_updates": 0,
                    },
                )
        torch.cuda.synchronize()
        verify_inference_shards(shards, tokens, identity_sha, output)
        record = {
            "kind": "FIXED_ENDPOINT_OUTER_INFERENCE",
            "identity": identity,
            "identity_sha256": identity_sha,
            "shards": shards,
            "rows": len(tokens),
            "elapsed_this_invocation_seconds": time.monotonic() - started,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
            "total_model_parameters": sum(p.numel() for p in model.parameters()),
        }
        record["artifact_sha256"] = object_digest(record)
        atomic_json(receipt_path, record)
        return record
    finally:
        del model
        torch.cuda.empty_cache()
