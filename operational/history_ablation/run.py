"""CLI for the history-length ablation (see docs/sota_20261118/history/PROTOCOL.md).

Usage (PYTHONPATH must contain the repository root and ``src``)::

    python -m operational.history_ablation.run manifest
    python -m operational.history_ablation.run fidelity
    python -m operational.history_ablation.run contexts --seed 7 --modality event --k 8
    python -m operational.history_ablation.run fusion --seed 7
    python -m operational.history_ablation.report
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from operational.final_refit.training import _model

from . import archive
from .fitting import (
    context_outputs,
    context_stage,
    fit_stage,
    fusion_inputs,
    fusion_predict,
    fusion_stage,
    normalize_fragment,
    scaled_updates,
)
from .truncation import truncate_mapping

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "sota_20261118" / "history"
KS = (1, 2, 4, 8)
ARMS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 2),
    (4, 4),
    (8, 8),
    (1, 8),
    (2, 8),
    (4, 8),
    (8, 1),
    (8, 2),
    (8, 4),
)
RELEASED_UPDATES = {0: 2500, 1: 2434, 2: 2479, 3: 2402, 4: 2230}
BLOCKS = 5


def arm_name(k_rgb: int, k_event: int) -> str:
    """Stable arm label ``R{K_RGB}_E{K_EVENT}``."""
    return f"R{k_rgb}_E{k_event}"


def fusion_train_blocks(test_block: int) -> tuple[int, ...]:
    """Fusion training blocks for a test block: exclude it and the block its refiner saw.

    Rotation ``r`` generates block ``B_r`` with a refiner trained on ``B_{r+1}``, so the
    context features of ``B_{b-1}`` were produced by a refiner fitted on ``B_b``.
    """
    return tuple((test_block + offset) % BLOCKS for offset in (1, 2, 3))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _write_npz(path: Path, values: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)  # type: ignore[arg-type]
    os.replace(temporary, path)
    return _sha256(path)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _environment(device: torch.device) -> dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "source_sha256": {p.name: _sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }


def _manifest() -> dict[str, Any]:
    path = OUT / "INPUT_MANIFEST.json"
    if not path.is_file():
        raise FileNotFoundError("Run the 'manifest' command first")
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value["members"])


def _reader() -> archive.ArchiveReader:
    return archive.ArchiveReader(manifest=_manifest())


def _device(name: str) -> torch.device:
    if name == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def _rotation_inputs(reader: archive.ArchiveReader, rotation: int, modality: str) -> dict[str, Any]:
    member = archive.rotation_member
    return {
        "cache": reader.arrays(member(rotation, modality, "cache")),
        "normalizer": reader.json(member(rotation, modality, "normalizer")),
        "generation": reader.arrays(member(rotation, modality, "generation")),
    }


def _native_tokens() -> tuple[np.ndarray, np.ndarray]:
    frame = archive.supervision()
    return frame.sample_token.to_numpy(str), frame.group_id.to_numpy(str)


def _place(tokens: np.ndarray, part_tokens: np.ndarray) -> np.ndarray:
    lookup = {token: index for index, token in enumerate(tokens.tolist())}
    return np.asarray([lookup[token] for token in part_tokens.astype(str).tolist()])


# ----------------------------------------------------------------------------- commands


def cmd_manifest(_: argparse.Namespace) -> None:
    """Bind every archived input to RESTORE_INDEX and record supervision row hashes."""
    value = archive.build_manifest()
    value["supervision_rows_sha256"] = archive.rows_sha256()
    _write_json(OUT / "INPUT_MANIFEST.json", value)
    print(f"bound {len(value['members'])} members")


def cmd_fidelity(args: argparse.Namespace) -> None:
    """Reproduce archived generation outputs from archived checkpoints (no training)."""
    device = _device(args.device)
    reader = _reader()
    tokens, groups = _native_tokens()
    stitched = reader.arrays(archive.stitch_member("PHASE20_CONTEXT"))
    stitched_index = _place(tokens, stitched["sample_token"])
    released = np.empty_like(stitched["features"])
    released[stitched_index] = stitched["features"]
    released_width = np.empty_like(stitched["rgb_width_logits"])
    released_width[stitched_index] = stitched["rgb_width_logits"]
    raw = {
        m: reader.arrays(archive.stitch_member(i))
        for m, i in (("event", "PHASE17_RAW_EVENT"), ("rgb", "PHASE17_RAW_RGB"))
    }
    report: dict[str, Any] = {"schema": "history_ablation_fidelity_v1", "rotations": {}}
    frame = archive.supervision()
    report["rgb_observation_span_us"] = {
        "median": float(frame.actual_span_us.median()),
        "p05": float(frame.actual_span_us.quantile(0.05)),
        "p95": float(frame.actual_span_us.quantile(0.95)),
        "producer_frame_count_values": sorted(set(frame.producer_frame_count.astype(int))),
    }
    covered = np.zeros(len(tokens), dtype=np.int64)
    for rotation in archive.ROTATIONS:
        entry: dict[str, Any] = {}
        for modality in archive.MODALITIES:
            inputs = _rotation_inputs(reader, rotation, modality)
            cache, generation = inputs["cache"], inputs["generation"]
            population = len(cache["target_phase"])
            payload = reader.read_bytes(archive.rotation_member(rotation, modality, "checkpoint"))
            checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
            stage = context_stage(archive.STAGE[modality], RELEASED_UPDATES[rotation])
            model = _model(stage)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model = model.to(device).float().eval().requires_grad_(False)
            values = normalize_fragment(generation, inputs["normalizer"])
            states, widths = context_outputs(model, values, device=device)
            index = _place(tokens, generation["sample_token"])
            columns = slice(0, 10) if modality == "rgb" else slice(10, 20)
            raw_index = _place(raw[modality]["sample_token"], generation["sample_token"])
            entry[modality] = {
                "refiner_population": population,
                "refiner_groups": sorted(set(cache["group_id"].astype(str).tolist())),
                "generation_groups": sorted(set(generation["group_id"].astype(str).tolist())),
                "scaled_updates": scaled_updates(population),
                "released_updates": RELEASED_UPDATES[rotation],
                "checkpoint_completed_updates": int(checkpoint["completed_updates"]),
                "checkpoint_device": checkpoint["identity"].get("device"),
                "checkpoint_seed": checkpoint["identity"].get("seed"),
                "generation_equals_stitched_phase17": bool(
                    np.array_equal(raw[modality]["features"][raw_index], generation["features"])
                    and np.array_equal(raw[modality]["valid"][raw_index], generation["valid"])
                    and np.array_equal(raw[modality]["timing"][raw_index], generation["timing"])
                ),
                "max_abs_diff_vs_released_phase20": float(
                    np.abs(states - released[index][:, columns]).max()
                ),
                "max_abs_diff_width_logits": (
                    float(np.abs(widths - released_width[index]).max())
                    if modality == "rgb"
                    else None
                ),
                "refiner_groups_disjoint_from_generation": not (
                    set(cache["group_id"].astype(str).tolist())
                    & set(generation["group_id"].astype(str).tolist())
                ),
            }
            if modality == "event":
                covered[index] += 1
        report["rotations"][str(rotation)] = entry
    report["every_query_generated_exactly_once"] = bool(np.all(covered == 1))
    report["environment"] = _environment(device)
    _write_json(OUT / "FIDELITY.json", report)
    print(json.dumps(report["rotations"], indent=1)[:4000])


def cmd_contexts(args: argparse.Namespace) -> None:
    """Train five rotation refiners at truncation K and stitch OOF context outputs."""
    device = _device(args.device)
    torch.set_num_threads(args.threads)
    reader = _reader()
    tokens, groups = _native_tokens()
    modality, k, seed = args.modality, args.k, args.seed
    output = OUT / "contexts" / f"seed{seed}" / f"{modality}_K{k}.npz"
    if output.exists() and not args.force:
        print(f"exists: {output}")
        return
    states = np.full((len(tokens), 10), np.nan, dtype=np.float32)
    widths = np.full((len(tokens), 2), np.nan, dtype=np.float32)
    rotations: dict[str, Any] = {}
    started = datetime.now(UTC).isoformat()
    for rotation in archive.ROTATIONS:
        inputs = _rotation_inputs(reader, rotation, modality)
        cache = truncate_mapping(inputs["cache"], k)
        updates = scaled_updates(len(cache["target_phase"]))
        if updates != RELEASED_UPDATES[rotation]:
            raise ValueError("Refiner exposure differs from the released rotation")
        clock = time.perf_counter()
        fit = fit_stage(
            context_stage(archive.STAGE[modality], updates), cache, seed=seed, device=device
        )
        generation = truncate_mapping(
            normalize_fragment(inputs["generation"], inputs["normalizer"]), k
        )
        part_states, part_widths = context_outputs(fit.model, generation, device=device)
        index = _place(tokens, inputs["generation"]["sample_token"])
        states[index], widths[index] = part_states, part_widths
        rotations[str(rotation)] = {
            "updates": updates,
            "seconds": time.perf_counter() - clock,
            "loss_first100": float(np.mean(fit.losses[:100])),
            "loss_last100": float(np.mean(fit.losses[-100:])),
            "refiner_valid_slots_max": int(cache["valid"].sum(1).max()),
            "generation_valid_slots_max": int(generation["valid"].sum(1).max()),
            "generation_rows": int(len(index)),
        }
        print(modality, k, seed, rotation, rotations[str(rotation)], flush=True)
    if not np.isfinite(states).all():
        raise ValueError("Some queries were not generated by exactly one rotation")
    digest = _write_npz(
        output,
        {"states": states, "width_logits": widths, "sample_token": tokens, "group_id": groups},
    )
    _write_json(
        output.with_suffix(".json"),
        {
            "schema": "history_ablation_context_outputs_v1",
            "modality": modality,
            "k": k,
            "seed": seed,
            "output_sha256": digest,
            "fields": "frozen_context_state block[:,160:170]: raw_location, point_phase, "
            "experts[3], timing_last[4], valid_count/8",
            "rotations": rotations,
            "start_time": started,
            "end_time": datetime.now(UTC).isoformat(),
            "targets_used_for_outputs": False,
            "environment": _environment(device),
        },
    )


def _load_context(seed: int, modality: str, k: int) -> dict[str, np.ndarray]:
    path = OUT / "contexts" / f"seed{seed}" / f"{modality}_K{k}.npz"
    receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if receipt["output_sha256"] != _sha256(path):
        raise ValueError(f"Context output changed: {path}")
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def _blocks() -> dict[int, list[str]]:
    review = json.loads(
        (REPO / "docs/final_refit_20261017/NESTED_ROTATION_REVIEW.json").read_text(encoding="utf-8")
    )
    result = {}
    for rotation in review["rotations"]:
        r = int(rotation["rotation"])
        result[r] = sorted(rotation["generation_groups"])
        expected_refiner = sorted(review["rotations"][(r + 1) % BLOCKS]["generation_groups"])
        if sorted(rotation["refiner_groups"]) != expected_refiner:
            raise ValueError("Rotation refiner block is not the next generation block")
    return result


def _oof_fusion(
    features340: np.ndarray,
    widths: np.ndarray,
    target_phase: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    blocks = _blocks()
    location = np.full(len(groups), np.nan, dtype=np.float32)
    folds: dict[str, Any] = {}
    for test in range(BLOCKS):
        train_groups = {g for b in fusion_train_blocks(test) for g in blocks[b]}
        train = np.isin(groups, sorted(train_groups))
        test_rows = np.isin(groups, blocks[test])
        if train_groups & set(blocks[test]) or train_groups & set(blocks[(test - 1) % BLOCKS]):
            raise AssertionError("Fusion fold leaks its test block or the refiner-seen block")
        values = {
            "features": features340[train],
            "rgb_width_logits": widths[train],
            "target_phase": target_phase[train],
            "group_id": groups[train],
        }
        updates = scaled_updates(int(train.sum()))
        fit = fit_stage(fusion_stage(updates), values, seed=seed, device=device)
        location[test_rows] = fusion_predict(
            fit.model, features340[test_rows], widths[test_rows], device=device
        )
        folds[str(test)] = {
            "train_blocks": list(fusion_train_blocks(test)),
            "train_rows": int(train.sum()),
            "test_rows": int(test_rows.sum()),
            "updates": updates,
            "loss_last100": float(np.mean(fit.losses[-100:])),
        }
    if not np.isfinite(location).all():
        raise ValueError("Fusion did not predict every query exactly once")
    return location, folds


def cmd_fusion(args: argparse.Namespace) -> None:
    """Refit the PHASE_ONLY head OOF for every arm (and the archived reference)."""
    device = _device(args.device)
    torch.set_num_threads(args.threads)
    frame = archive.supervision()
    tokens = frame.sample_token.to_numpy(str)
    groups = frame.group_id.to_numpy(str)
    target_phase = frame.target_phase.to_numpy(np.float32)
    seed = args.seed
    arms: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    if args.reference:
        reader = _reader()
        stitched = reader.arrays(archive.stitch_member("PHASE20_CONTEXT"))
        index = _place(tokens, stitched["sample_token"])
        released = np.empty_like(stitched["features"])
        released[index] = stitched["features"]
        released_width = np.empty_like(stitched["rgb_width_logits"])
        released_width[index] = stitched["rgb_width_logits"]
        arms.append(
            (
                "ARCHIVED_RELEASED",
                released[:, :10],
                released[:, 10:],
                released_width,
                released[:, 0],
            )
        )
    for k_rgb, k_event in ARMS:
        rgb = _load_context(seed, "rgb", k_rgb)
        event = _load_context(seed, "event", k_event)
        for name, values in (("rgb", rgb), ("event", event)):
            if not np.array_equal(values["sample_token"], tokens):
                raise ValueError(f"{name} context rows differ from the native TRAIN40 order")
        arms.append(
            (
                arm_name(k_rgb, k_event),
                rgb["states"],
                event["states"],
                rgb["width_logits"],
                rgb["states"][:, 0],
            )
        )
    for name, rgb10, event10, widths, _ in arms:
        output = OUT / "predictions" / f"seed{seed}" / f"{name}.npz"
        if output.exists() and not args.force:
            print(f"exists: {output}")
            continue
        clock = time.perf_counter()
        features340 = fusion_inputs(rgb10, event10)
        location, folds = _oof_fusion(
            features340, widths, target_phase, groups, seed=seed, device=device
        )
        digest = _write_npz(
            output,
            {
                "fused_location": location,
                "rgb_ctx_location": np.ascontiguousarray(rgb10[:, 0]),
                "event_ctx_location": np.ascontiguousarray(event10[:, 0]),
                "sample_token": tokens,
                "group_id": groups,
            },
        )
        _write_json(
            output.with_suffix(".json"),
            {
                "schema": "history_ablation_fusion_predictions_v1",
                "arm": name,
                "seed": seed,
                "output_sha256": digest,
                "folds": folds,
                "seconds": time.perf_counter() - clock,
                "fusion_rule": "train on blocks b+1..b+3; exclude test block b and block b-1",
                "environment": _environment(device),
            },
        )
        print(name, seed, f"{time.perf_counter() - clock:.0f}s", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments and dispatch a command."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest", help="bind archived inputs to RESTORE_INDEX")
    fidelity = sub.add_parser("fidelity", help="reproduce archived outputs from checkpoints")
    fidelity.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    contexts = sub.add_parser("contexts", help="train truncated refiners for one modality/K")
    contexts.add_argument("--seed", type=int, required=True)
    contexts.add_argument("--modality", choices=archive.MODALITIES, required=True)
    contexts.add_argument("--k", type=int, choices=KS, required=True)
    contexts.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    contexts.add_argument("--threads", type=int, default=4)
    contexts.add_argument("--force", action="store_true")
    fusion = sub.add_parser("fusion", help="OOF PHASE_ONLY refits for all arms")
    fusion.add_argument("--seed", type=int, required=True)
    fusion.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    fusion.add_argument("--threads", type=int, default=4)
    fusion.add_argument(
        "--reference",
        action="store_true",
        help="also refit on the archived released PHASE20 features",
    )
    fusion.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    handlers = {
        "manifest": cmd_manifest,
        "fidelity": cmd_fidelity,
        "contexts": cmd_contexts,
        "fusion": cmd_fusion,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
