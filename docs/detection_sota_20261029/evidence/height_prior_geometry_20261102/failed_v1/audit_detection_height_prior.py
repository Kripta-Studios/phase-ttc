"""Measure the ideal-target height prior's bias on the registered TRAIN panel."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.cuboid_edge_depth import depth_from_cuboid_edges
from e_jepa_ttc.detection.eap_rgb_data import RawRGBSample, finite_matrix
from e_jepa_ttc.detection.geometry_contract import original_bundle_corners
from e_jepa_ttc.detection.monocop_campaign import PANEL_SHA256
from e_jepa_ttc.detection.monodgp_eap_adapter import prepare_sample, resized_intrinsic
from e_jepa_ttc.detection.range_campaign import digest, write_json
from e_jepa_ttc.detection.raw_diagnostics import interval


def describe(values: list[float]) -> dict:
    """Preserve signs and report absolute errors separately."""
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"count": 0}
    return {
        "count": len(a),
        "mean_signed_m": float(a.mean()),
        "mae_m": float(abs(a).mean()),
        "median_abs_m": float(np.median(abs(a))),
        "p90_abs_m": float(np.quantile(abs(a), 0.9)),
        "max_abs_m": float(abs(a).max()),
        "fraction_abs_lt": {str(t): float((abs(a) < t).mean()) for t in (0.5, 1, 2, 4)},
    }


def main() -> None:
    """Use TRAIN metadata and blank images; no model, RGB decoding or optimizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--population", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or args.output.exists():
        raise ValueError("Require CPU-only execution and a new output directory")
    torch.set_num_threads(1)
    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    start = time.perf_counter()
    if digest(args.population) != PANEL_SHA256:
        raise ValueError("Registered TRAIN panel required")
    population = json.loads(args.population.read_text())
    tokens = population["roles"]["train"]
    if len(tokens) != 4096 or {t.split(":")[0] for t in tokens} & {
        t.split(":")[0] for t in population["roles"]["selection"]
    }:
        raise ValueError("TRAIN ownership or sequence separation differs")
    media_path = args.dataset / "data/train.parquet"
    if digest(media_path) != population["source_media_sha256"]:
        raise ValueError("Canonical metadata changed")
    media = pd.read_parquet(
        media_path,
        filters=[("sample_token", "in", tokens)],
        columns=["sample_token", "sequence_id", "split", "K_event", "T_event_ego"],
    )
    if (
        len(media) != len(tokens)
        or media.sample_token.duplicated().any()
        or not media.split.eq("train").all()
    ):
        raise ValueError("Missing, duplicate or non-public TRAIN metadata")
    frames = {row["sample_token"]: row for row in media.to_dict("records")}
    by_sequence: dict[str, list[str]] = defaultdict(list)
    for token in tokens:
        by_sequence[token.split(":")[0]].append(token)
    paths = [
        Path(__file__),
        *[
            ROOT / f"src/e_jepa_ttc/detection/{name}.py"
            for name in (
                "cuboid_edge_depth",
                "geometry_contract",
                "monodgp_eap_adapter",
                "eap_rgb_data",
            )
        ],
    ]
    sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    inputs = {args.population.as_posix(): PANEL_SHA256, media_path.as_posix(): digest(media_path)}
    args.output.mkdir(parents=True)
    blank = np.zeros((384, 640, 3), dtype=np.uint8)
    rows: list[dict[str, Any]] = []
    finished = 0
    raw_count, admitted_count = 0, 0
    for sequence, sequence_tokens in sorted(by_sequence.items()):
        name = f"data/train/{sequence}/labels.parquet"
        if digest(args.dataset / name) != population["label_sha256"][name]:
            raise ValueError("TRAIN labels changed")
        inputs[(args.dataset / name).as_posix()] = population["label_sha256"][name]
        labels = pd.read_parquet(
            args.dataset / name, filters=[("sample_token", "in", sequence_tokens)]
        )
        if not labels.split.eq("train").all() or not labels.sequence_id.eq(sequence).all():
            raise ValueError("Label role/sequence mismatch")
        if labels.duplicated(["sample_token", "instance_id"]).any():
            raise ValueError("Repeated object label")
        label_frames = {
            str(k): v.to_dict("records") for k, v in labels.groupby("sample_token", sort=False)
        }
        for token in sequence_tokens:
            if time.perf_counter() - start > 600:
                raise TimeoutError("TRAIN geometry audit exceeded its ten-minute CPU bound")
            metadata = frames[token]
            original_wh = tuple(population["media"][token]["image_wh"])
            intrinsic = resized_intrinsic(
                finite_matrix(metadata["K_event"], (3, 3)), original_wh, (640, 384)
            )
            transform = finite_matrix(metadata["T_event_ego"], (4, 4))
            annotations = label_frames.get(token, [])
            prepared = prepare_sample(
                RawRGBSample(
                    token,
                    sequence,
                    blank,
                    annotations,
                    intrinsic,
                    transform,
                    0,
                    0,
                    {},
                    "blank_geometry_only",
                ),
                max_depth=300,
            )
            raw_count += len(annotations)
            admitted_count += len(prepared.native_ground_truth)
            k = prepared.calibration.numpy().astype(float)
            for index, native in enumerate(prepared.native_ground_truth):
                target = prepared.target
                depth = float(target["depth"][index, 0])
                h, w, length = target["size_3d"][index].numpy().astype(float)
                coords = target["boxes_3d"][index].numpy().astype(float)
                uv, distances = coords[:2] * [640, 384], coords[2:] * [640, 640, 384, 384]
                left, right, top, bottom = distances
                geometric_depth = h * k[1, 1] / max(top + bottom, 1)
                clipped = np.array(
                    [
                        uv[0] - left <= 1e-3,
                        uv[0] + right >= 639 - 1e-3,
                        uv[1] - top <= 1e-3,
                        uv[1] + bottom >= 383 - 1e-3,
                    ]
                )
                yaw = native["yaw_rad"]
                if not isinstance(yaw, (float, int)):
                    raise TypeError("Expected a numeric native target yaw")
                offsets = original_bundle_corners([0, 0, 0, length, h, w, float(yaw)])
                offsets = offsets @ transform[:3, :3].T
                edge_depths = depth_from_cuboid_edges(offsets, k, uv, distances)
                usable = edge_depths[~clipped]
                edge_depth = float(np.median(usable)) if len(usable) else None
                rows.append(
                    {
                        "token": token,
                        "target_index": index,
                        "class": str(native["class_name"]),
                        "optical_depth_m": depth,
                        "depth_bucket": interval(depth, (0.0, 20.0, 40.0, 60.0, 100.0, 150.0)),
                        "height_pixels_at_384": float(top + bottom),
                        "length_m": length,
                        "height_prior_m": geometric_depth,
                        "height_error_m": geometric_depth - depth,
                        "vertical_clipped": bool(clipped[2:].any()),
                        "any_clipped": bool(clipped.any()),
                        "usable_edges": int((~clipped).sum()),
                        "cuboid_edge_m": edge_depth,
                        "cuboid_edge_error_m": edge_depth - depth
                        if edge_depth is not None
                        else None,
                        "one_pixel_height_sensitivity_m": abs(
                            h * k[1, 1] / (top + bottom + 1) - geometric_depth
                        ),
                    }
                )
            finished += 1
        write_json(
            args.output / "STATUS.json",
            {
                "status": "CPU_TRAIN_GEOMETRY",
                "frames": finished,
                "targets": len(rows),
                "elapsed_seconds": time.perf_counter() - start,
            },
        )
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for key in (
            "all",
            f"class/{row['class']}",
            f"depth/{row['depth_bucket']}",
            f"vertical_clipped/{row['vertical_clipped']}",
        ):
            groups[key].append(row)
    summaries = {}
    for key, members in groups.items():
        matched = [r for r in members if r["cuboid_edge_error_m"] is not None]
        summaries[key] = {
            "height_prior": describe([r["height_error_m"] for r in members]),
            "height_prior_on_edge_valid_subset": describe([r["height_error_m"] for r in matched]),
            "cuboid_edges_on_same_subset": describe([r["cuboid_edge_error_m"] for r in matched]),
            "median_one_pixel_sensitivity_m": float(
                np.median([r["one_pixel_height_sensitivity_m"] for r in members])
            ),
        }
    if sources != {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}:
        raise ValueError("Diagnostic source changed during execution")
    data = "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows).encode()
    (args.output / "ROWS.jsonl.gz").write_bytes(gzip.compress(data, mtime=0))
    result = {
        "schema": "phase_train_height_prior_audit_v1",
        "status": "VERIFIED_TRAIN_DIAGNOSTIC_ONLY",
        "frames": finished,
        "raw_targets": raw_count,
        "admitted_targets": admitted_count,
        "groups": summaries,
        "input_sha256": inputs,
        "source_sha256": sources,
        "rows_sha256": digest(args.output / "ROWS.jsonl.gz"),
        "elapsed_seconds": time.perf_counter() - start,
        "cuda_initialized": torch.cuda.is_initialized(),
        "model_inference": False,
        "training": False,
        "test_labels_read": False,
        "limitations": [
            "Uses perfect TRAIN target size, yaw and 2D edges; not predicted quality.",
            "Cuboid inversion proves pinhole consistency, not sensor rectification.",
            "Clipped edges are excluded; the compared subset is explicit.",
            "One-pixel sensitivity is not an accuracy bound: regressors can predict subpixels.",
            "No existing model, producer, checkpoint or promotion rule changed.",
        ],
    }
    write_json(args.output / "RESULTS.json", result)
    write_json(
        args.output / "STATUS.json",
        {"status": "COMPLETE", "frames": finished, "elapsed_seconds": result["elapsed_seconds"]},
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "frames": finished,
                "targets": len(rows),
                "all": summaries["all"],
            }
        )
    )


if __name__ == "__main__":
    main()
