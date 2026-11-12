"""Measure native RAW-to-host inference on a fixed physical eAP Test12 prefix.

The PHASE runtime reuses computed causal observations within each sequence.
Garl consumes the matching supplied current pair. No ground truth is opened.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
sys.dont_write_bytecode = True

import numpy as np
import torch
from torch import nn

from e_jepa_ttc.data.garl_official_preprocessing import official_resize_roi, official_square_box
from e_jepa_ttc.efficient_context.garl_input import inference_record
from e_jepa_ttc.rgb_port.data import TarFrameReader
from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
from operational.evttc_rgb_transfer.inputs import _normalized_rgb
from operational.final_refit.test12 import TRUSTED_INPUTS_SHA256
from operational.garl_evttc_competition.model import OfficialGarlRuntime
from operational.local_comparison.core import assert_parity, statistics, without_decoders
from operational.local_comparison.eap import current_garl_row, select_physical
from operational.phase_robust.bundle import verified_upstream_paths
from scripts.audit_phase_native_source_closure import DEPENDENCIES
from scripts.benchmark_local_comparison import append, sha, telemetry, write
from scripts.final_train40_transfer import load_models
from scripts.release_native_local_latency import timed_predict
from scripts.release_native_raw_latency import prepare, raw_native
from scripts.release_research import verify_complete


def prepare_current(
    row: dict[str, Any], media: Path, reader: TarFrameReader, pool: ReaderPool
) -> np.ndarray:
    """Execute the parity-audited native RGB normalization and 40-plane event mapping."""
    images = [
        reader.read_uint8(s, m)
        for s, m in zip(row["rgb_shard_paths"], row["rgb_member_paths"], strict=True)
    ]
    normalized = np.concatenate([_normalized_rgb(image) for image in images], axis=0)
    square = official_square_box([tuple(b) for b in row["boxes_xyxy"]], 1)
    rgb = official_resize_roi(normalized, square, (128, 128)).numpy()
    event = inference_record(row, pool, media / "data/test").numpy()
    return np.concatenate((rgb, event), axis=0).astype(np.float32, copy=False)


def main() -> None:
    """Verify real sensor reconstruction before running three alternating-order passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--garl-freeze", type=Path, required=True)
    parser.add_argument("--phase-bundle", type=Path, required=True)
    parser.add_argument("--media-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sequence", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.per_sequence < 1:
        parser.error("Counts must be positive")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise ValueError("Use a new experiment directory")
    if sha(args.public) != TRUSTED_INPUTS_SHA256:
        raise ValueError("Public input identity changed")
    data = json.loads(args.public.read_text(encoding="utf-8"))
    selected = select_physical(data, args.per_sequence)
    rows = {q["sample_token"]: current_garl_row(data, q) for q in selected}
    verify_complete(args.media_freeze.parent)
    media_receipt = json.loads(args.media_freeze.read_text(encoding="utf-8"))
    media_by_path = {
        str(Path(path).resolve()): media_receipt["inputs"][key]
        for key, path in media_receipt["input_paths"].items()
        if key.startswith(("raw:", "rgb:"))
    }
    selected_media: set[Path] = set()
    for q in selected:
        selected_media.update(
            Path(data["event_observations"][i]["raw_path"]).resolve() for i in q["event_history"]
        )
        selected_media.update(
            (Path(data["media_root"]) / shard).resolve()
            for i in q["rgb_history"]
            for shard in data["rgb_observations"][i]["producer_shards"]
        )
    media_bindings = [
        {
            "path": str(path),
            "historically_verified_sha256": media_by_path[str(path)],
            "current_bytes": path.stat().st_size,
            "current_mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(selected_media)
    ]
    model_order = ("phase_native_raw", "garl_full_raw", "garl_ttc_only_raw")
    freeze = json.loads(args.garl_freeze.read_text(encoding="utf-8"))
    source_paths = [
        Path(__file__),
        ROOT / "operational/local_comparison/eap.py",
        ROOT / "operational/local_comparison/core.py",
        ROOT / "scripts/benchmark_local_comparison.py",
        ROOT / "scripts/release_native_raw_latency.py",
        ROOT / "scripts/release_native_local_latency.py",
        ROOT / "scripts/final_train40_transfer.py",
    ]
    source_paths.extend(ROOT / name for name in DEPENDENCIES)
    for directory in (
        "operational/final_refit",
        "operational/rgb_port",
        "operational/phase_timewarp",
        "operational/garl_evttc_competition",
        "operational/phase_robust",
        "src/e_jepa_ttc/efficient_context",
        "src/e_jepa_ttc/data",
        "src/e_jepa_ttc/rgb_port",
        "src/e_jepa_ttc/simplex_t",
    ):
        source_paths.extend((ROOT / directory).rglob("*.py"))
    protocol = {
        "experiment_id": "same_hardware_eap_20261112",
        "seed": 7,
        "start_time": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "host": socket.gethostname(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "garl_inventory": freeze["inventory"],
        "garl_freeze_sha256": sha(args.garl_freeze),
        "phase_bundle_manifest": json.loads((args.phase_bundle / "BUNDLE.json").read_text()),
        "queries": selected,
        "public_sha256": sha(args.public),
        "media_provenance": {
            "prior_freeze_sha256": sha(args.media_freeze),
            "prior_complete_sha256": sha(args.media_freeze.parent / "COMPLETE.json"),
            "files": media_bindings,
            "scope": (
                "Prior full-media hash verification; this run verifies selected event tensors "
                "and logs prepared Garl hashes, without rehashing the complete media files"
            ),
        },
        "garl_rows": rows,
        "repeats": args.repeats,
        "models": model_order,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(source_paths))},
        "contract": {
            "selection": "Chronological physical prefix per Test12 sequence; aliases removed",
            "clock": "CUDA synchronized RAW-to-host wall time; no stage instrumentation",
            "state": "Per-sequence readers and PHASE observation cache; reset each pass",
            "garl_readers": "Independent per-arm reader/frame caches; no cross-arm cache reuse",
            "exclusions": "Load, reader setup, hashes, warmup, sensor acquisition and detection",
            "inputs": "PHASE native causal history; Garl corresponding supplied current pair",
            "precision": "FP32, TF32 off, 2 CPU threads; same shared CUDA device",
            "labels": "None read; native local replay, not an official re-score or accuracy claim",
            "warmup": "Three first-query calls per arm; reset native caches before measured pass",
            "decoder_bypass": "Same Garl weights; require all-query TTC parity",
            "ttc_conversion": (
                "Native released 0.1 s annotation interval; sensor drift recorded separately"
            ),
        },
    }
    protocol["dataset_manifest_hash"] = hashlib.sha256(
        json.dumps(selected, sort_keys=True).encode()
    ).hexdigest()
    protocol["config_hash"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, default=str).encode()
    ).hexdigest()
    write(out / "protocol.json", protocol)
    if args.prepare_only:
        print(f"Validated {len(selected)} physical query mappings; no GPU work", flush=True)
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(7)
    # Reconstruct all needed event tensors and compare their frozen real-input references.
    pool = ReaderPool()
    seen = set()
    try:
        for q in selected:
            for identity in q["event_history"]:
                if identity in seen:
                    continue
                seen.add(identity)
                item = data["event_observations"][identity]
                reference = Path(item["current_tensor_path"])
                if sha(reference) != item["current_tensor_sha256"]:
                    raise ValueError("Frozen sensor reference changed")
                actual = prepare(pool, item).numpy()
                with np.load(reference, allow_pickle=False) as archive:
                    expected = archive["events"][-1:]
                    if not archive["valid"][-1] or not np.array_equal(actual, expected):
                        raise ValueError("RAW/prepared event parity failed")
                    if actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
                        raise ValueError("RAW/prepared event byte parity failed")
                append(
                    out / "input_parity.jsonl",
                    {"identity": identity, "exact": True, "reference_sha256": sha(reference)},
                )
    finally:
        pool.close()
    models, normalizers = load_models(verified_upstream_paths(args.phase_bundle), "cuda")
    garl = OfficialGarlRuntime(
        Path(freeze["inventory"]["upstream"]["path"]), freeze["inventory"], "cuda"
    )
    if any(row["native_delta_t_s"] != float(garl.models["full"].dT) for row in rows.values()):
        raise ValueError("Input conversion differs from released model.dT")
    write(out / "loading.json", {"garl": garl.bindings, "telemetry": telemetry()})
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in selected:
        groups[query["sequence_id"]].append(query)
    records = []
    prior: dict[tuple[str, str], float | None] = {}
    with torch.inference_mode():
        for repeat in range(args.repeats):
            for sequence, queries in groups.items():
                append(
                    out / "telemetry.jsonl", {"repeat": repeat, "sequence": sequence, **telemetry()}
                )
                native, pool = raw_native(models, normalizers, data)
                readers = {
                    name: TarFrameReader(
                        Path(data["media_root"]), max_open=2, max_frame_cache_bytes=128 * 1024**2
                    )
                    for name in model_order[1:]
                }
                garl_pools = {name: ReaderPool() for name in model_order[1:]}

                # Called synchronously within this sequence; reset resources are intentional.
                def call(name: str, q: dict[str, Any]) -> dict[str, Any]:
                    if name == "phase_native_raw":
                        result = timed_predict(native, q, data, False)  # noqa: B023
                        prediction = float(result["prediction"])
                        if not np.isfinite(prediction):
                            raise ValueError("Nonfinite PHASE prediction")
                        return {
                            "ttc_seconds": prediction,
                            "total_ms": float(result["elapsed_ms"]),
                            "status": "ok",
                            "reason": "",
                            "diagnostics": result,
                        }
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    sensor = prepare_current(
                        rows[q["sample_token"]],
                        Path(data["media_root"]),
                        readers[name],  # noqa: B023
                        garl_pools[name],  # noqa: B023
                    )
                    context = (
                        without_decoders(cast(nn.Module, garl.models["full"]))
                        if name == "garl_ttc_only_raw"
                        else contextlib.nullcontext()
                    )
                    with context:
                        prediction = garl.predict(
                            "full", sensor, rows[q["sample_token"]]["native_delta_t_s"]
                        )
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    return {
                        "ttc_seconds": prediction.model_ttc,
                        "total_ms": elapsed,
                        "status": prediction.status,
                        "reason": prediction.reason,
                        "heights": prediction.heights,
                        "prepared_input_sha256": hashlib.sha256(sensor.tobytes()).hexdigest(),
                    }

                try:
                    for name in model_order:
                        for _ in range(3):
                            call(name, queries[0])
                    native.close()
                    pool.close()
                    native, pool = raw_native(models, normalizers, data)
                    for resource in (*readers.values(), *garl_pools.values()):
                        resource.close()
                    readers = {
                        name: TarFrameReader(
                            Path(data["media_root"]),
                            max_open=2,
                            max_frame_cache_bytes=128 * 1024**2,
                        )
                        for name in model_order[1:]
                    }
                    garl_pools = {name: ReaderPool() for name in model_order[1:]}
                    for ordinal, query in enumerate(queries):
                        offset = (ordinal + repeat) % len(model_order)
                        predictions = {}
                        for name in model_order[offset:] + model_order[:offset]:
                            try:
                                measured = call(name, query)
                            except Exception as error:
                                append(
                                    out / "errors.jsonl",
                                    {
                                        "model": name,
                                        "query": query["sample_token"],
                                        "repeat": repeat,
                                        "error": repr(error),
                                        "telemetry": telemetry(),
                                    },
                                )
                                raise
                            prediction = measured["ttc_seconds"]
                            key = name, query["sample_token"]
                            if key in prior:
                                assert_parity(prior[key], prediction)
                            prior[key] = prediction
                            predictions[name] = measured
                            record = {
                                "model": name,
                                "query": query["sample_token"],
                                "sequence": sequence,
                                "repeat": repeat,
                                "order_position": (
                                    model_order[offset:] + model_order[:offset]
                                ).index(name),
                                **measured,
                            }
                            records.append(record)
                            append(out / "trials.jsonl", record)
                        full, bypass = (
                            predictions["garl_full_raw"],
                            predictions["garl_ttc_only_raw"],
                        )
                        for field in ("status", "reason", "prepared_input_sha256", "heights"):
                            if full[field] != bypass[field]:
                                raise ValueError(f"Garl bypass changed {field}")
                        assert_parity(full["ttc_seconds"], bypass["ttc_seconds"])
                    print(
                        f"eAP repeat={repeat + 1} sequence={sequence} trials={len(records)}",
                        flush=True,
                    )
                finally:
                    native.close()
                    pool.close()
                    for resource in (*readers.values(), *garl_pools.values()):
                        resource.close()
    write(
        out / "summary.json",
        {
            "status": "complete",
            "end_time": datetime.now(UTC).isoformat(),
            "protocol_sha256": sha(out / "protocol.json"),
            "trials_sha256": sha(out / "trials.jsonl"),
            "models": {
                name: {
                    "attempts": sum(r["model"] == name for r in records),
                    "successes": sum(r["model"] == name and r["status"] == "ok" for r in records),
                    "total_ms": statistics([r["total_ms"] for r in records if r["model"] == name]),
                }
                for name in model_order
            },
            "parity": "RAW event inputs exact; repeat and Garl bypass outputs within 1e-5",
        },
    )


if __name__ == "__main__":
    main()
