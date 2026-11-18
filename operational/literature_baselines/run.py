"""Run one baseline over the subset manifest (inside that baseline's own venv).

``python -m operational.literature_baselines.run --method OEXP --manifest <npz> --out <dir>``
writes ``<METHOD>_OUTPUTS.npz`` (aggregated rho per query, manifest order) and
``<METHOD>_RECEIPT.json``.  ``--limit N --timing-only`` is the pre-registered timing pilot
(no rho is stored).  Partial progress is saved every ``--save-every`` queries and resumed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from operational.phase_geometry.metrics import json_safe

from .adapters import load
from .common import CROP_SIZE, affine_rho, aggregate, box_in_crop, region_mask
from .frames import TarFrameReader, crop, eap_root

AGGREGATIONS = ("median", "inner", "mean")


def aggregate_output(output: dict[str, np.ndarray], crop_box: np.ndarray) -> dict[str, float]:
    """Primary and secondary statistics of one model output inside the query-frame box."""
    region = region_mask(crop_box)
    inner = region_mask(crop_box, inner=True)
    if "rho" in output:
        rho = output["rho"]
        return {
            "median": aggregate(rho, region, "median"),
            "inner": aggregate(rho, inner, "median"),
            "mean": aggregate(rho, region, "mean"),
            "pixels": float(region.sum()),
        }
    flow = output["flow"]
    return {
        "median": affine_rho(flow, region),
        "inner": affine_rho(flow, inner),
        "mean": float("nan"),
        "pixels": float(region.sum()),
    }


def query_order(manifest: dict[str, np.ndarray]) -> np.ndarray:
    """Process by (sequence, timestamp, track) to reuse decoded frames; outputs stay aligned."""
    return np.lexsort((manifest["track_id"], manifest["timestamp_us"], manifest["sequence_id"]))


def main() -> None:
    """Run the requested method and write outputs and a receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("OEXP", "SFPP", "RAFT_AFF"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--save-every", type=int, default=250)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")
    device = torch.device("cuda")
    with np.load(args.manifest, allow_pickle=False) as archive:
        manifest = {k: archive[k] for k in archive.files}
    total = len(manifest["sample_token"])
    if args.timing_only:
        order = np.arange(min(args.limit or 40, total))
    else:
        order = query_order(manifest)
        if args.limit:
            order = order[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.method}_PILOT" if args.timing_only else f"{args.method}"
    partial = args.out / f"{stem}_PARTIAL.npz"
    results = {k: np.full(total, np.nan) for k in (*AGGREGATIONS, "pixels", "forward_s", "prep_s")}
    done = np.zeros(total, dtype=bool)
    if partial.exists() and not args.timing_only:
        with np.load(partial, allow_pickle=False) as archive:
            if not np.array_equal(archive["sample_token"], manifest["sample_token"]):
                raise ValueError("Partial results belong to another manifest")
            done = archive["done"].copy()
            for key in results:
                results[key] = archive[key].copy()
    predictor = load(args.method, device)
    reader = TarFrameReader(eap_root())
    torch.cuda.reset_peak_memory_stats()
    warm = reader.read(str(manifest["shard_current"][0]), str(manifest["member_current"][0]))
    warm_crop = crop(warm, tuple(manifest["square"][0]))
    predictor.run(warm_crop, warm_crop)
    torch.cuda.synchronize()
    started = time.time()
    processed = 0

    def save() -> None:
        np.savez(partial, sample_token=manifest["sample_token"], done=done, **results)

    for count, index in enumerate(order, start=1):
        if done[index]:
            continue
        t0 = time.perf_counter()
        square = tuple(float(v) for v in manifest["square"][index])
        current = reader.read(
            str(manifest["shard_current"][index]), str(manifest["member_current"][index])
        )
        previous = reader.read(
            str(manifest["shard_previous"][index]), str(manifest["member_previous"][index])
        )
        image1, image2 = crop(current, square), crop(previous, square)
        crop_box = box_in_crop(manifest["box_current"][index], square, CROP_SIZE)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        output = predictor.run(image1, image2)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        results["forward_s"][index] = t2 - t1
        results["prep_s"][index] = t1 - t0
        if not args.timing_only:
            for key, value in aggregate_output(output, crop_box).items():
                results[key][index] = value
        done[index] = True
        processed += 1
        if not args.timing_only and count % args.save_every == 0:
            save()
            print(f"{args.method}: {count}/{len(order)} {time.time() - started:.0f}s", flush=True)
    wall = time.time() - started
    forward = results["forward_s"][order]
    prep = results["prep_s"][order]
    receipt: dict[str, Any] = {
        "schema": "phase_literature_baselines_run_v1",
        "method": args.method,
        "timing_only": args.timing_only,
        "queries": int(len(order)),
        "processed_this_invocation": processed,
        "wall_seconds_this_invocation": wall,
        "forward_seconds_median": float(np.nanmedian(forward)),
        "forward_seconds_mean": float(np.nanmean(forward)),
        "prep_seconds_median": float(np.nanmedian(prep)),
        "prep_seconds_mean": float(np.nanmean(prep)),
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "provenance": predictor.provenance,
        "manifest": str(args.manifest.name),
        "crop_size": CROP_SIZE,
        "env": {
            k: os.environ.get(k, "") for k in ("PHASE_TTC_BASELINES_ROOT", "PHASE_TTC_EAP_DATASET")
        },
    }
    if args.timing_only:
        receipt["projected_hours_for_manifest"] = float(
            (np.nanmean(forward) + np.nanmean(prep)) * total / 3600.0
        )
    else:
        if not done.all():
            raise RuntimeError(f"{(~done).sum()} queries were not processed")
        np.savez_compressed(
            args.out / f"{args.method}_OUTPUTS.npz",
            sample_token=manifest["sample_token"],
            **{f"rho_{k}": results[k] for k in AGGREGATIONS},
            pixels=results["pixels"],
            forward_s=results["forward_s"],
            prep_s=results["prep_s"],
        )
        partial.unlink(missing_ok=True)
    (args.out / f"{stem}_RECEIPT.json").write_text(
        json.dumps(json_safe(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(receipt), indent=1))


if __name__ == "__main__":
    main()
