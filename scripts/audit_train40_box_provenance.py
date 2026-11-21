"""Compare public TRAIN40 image boxes to the projection of annotated camera-space corners."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_boxes(
    corners: np.ndarray, intrinsics: np.ndarray, supplied: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return valid flags, continuous IoU, edge error and truncated-box equality per query."""
    projected = corners @ intrinsics.transpose(0, 2, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = projected[:, :, :2] / projected[:, :, 2:3]
    bounds = np.concatenate([xy.min(axis=1), xy.max(axis=1)], axis=1)
    bounds = np.clip(bounds, 0, [1279, 719, 1279, 719])
    area_a = np.prod(np.maximum(bounds[:, 2:] - bounds[:, :2], 0), axis=1)
    area_b = np.prod(np.maximum(supplied[:, 2:] - supplied[:, :2], 0), axis=1)
    valid = (
        np.isfinite(xy).all(axis=(1, 2))
        & (corners[:, :, 2] > 0).all(axis=1)
        & np.isfinite(supplied).all(axis=1)
        & (area_a > 0)
        & (area_b > 0)
    )
    overlap = np.minimum(bounds[:, 2:], supplied[:, 2:]) - np.maximum(
        bounds[:, :2], supplied[:, :2]
    )
    inter = np.prod(np.maximum(overlap, 0), axis=1)
    iou = np.divide(inter, area_a + area_b - inter, out=np.zeros_like(inter), where=valid)
    return (
        valid,
        iou,
        np.abs(bounds - supplied).max(axis=1),
        (np.trunc(bounds) == supplied).all(axis=1),
    )


def summarize(values: list[tuple[float, float, bool]]) -> dict[str, Any]:
    if not values:
        return {"n_valid": 0}
    iou, error, exact = np.asarray(values, dtype=float).T
    return {
        "n_valid": len(values),
        "iou_median": float(np.median(iou)),
        "iou_p05": float(np.quantile(iou, 0.05)),
        "max_edge_error_px_median": float(np.median(error)),
        "max_edge_error_px_p95": float(np.quantile(error, 0.95)),
        "n_within_one_px": int(np.sum(error <= 1)),
        "n_within_two_px": int(np.sum(error <= 2)),
        "n_truncated_exact": int(np.sum(exact)),
    }


def audit(garl_root: Path, eap_root: Path) -> dict[str, Any]:
    """Evaluate last-frame boxes on the complete public training population without TTC columns."""
    paths = {
        "boxes": garl_root / "data" / "train.parquet",
        "corners": garl_root / "annotations" / "train.parquet",
        "media": eap_root / "data" / "train.parquet",
    }
    cameras: dict[tuple[str, str, str], tuple[float, ...]] = {}
    media_cols = ["sequence_id", "rgb_shard_path", "rgb_member_path", "K_event"]
    for batch in pq.ParquetFile(paths["media"]).iter_batches(columns=media_cols, batch_size=512):
        for row in batch.to_pylist():
            key = (row["sequence_id"], row["rgb_shard_path"], row["rgb_member_path"])
            k = tuple(np.asarray(row["K_event"], dtype=float).ravel())
            if key in cameras and cameras[key] != k:
                raise ValueError(f"Conflicting intrinsics for {key}")
            cameras[key] = k

    labels = pq.read_table(
        paths["corners"], columns=["sample_token", "box3d_Fcam"], use_threads=False
    )
    ids = labels["sample_token"].to_pylist()
    by_token = {token: i for i, token in enumerate(ids)}
    if len(by_token) != len(ids):
        raise ValueError("Camera-space annotations contain duplicate tokens")
    input_cols = [
        "sequence_id",
        "sample_token",
        "rgb_shard_paths",
        "rgb_member_paths",
        "boxes_xyxy",
    ]
    values: dict[str, list[tuple[float, float, bool]]] = defaultdict(list)
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"offered": 0, "missing_calibration": 0}
    )
    seen: set[str] = set()
    for batch in pq.ParquetFile(paths["boxes"]).iter_batches(columns=input_cols, batch_size=512):
        matrices, boxes, indices, sequences = [], [], [], []
        for row in batch.to_pylist():
            seq, token = row["sequence_id"], row["sample_token"]
            if token in seen:
                raise ValueError(f"Duplicate input token {token}")
            seen.add(token)
            counts[seq]["offered"] += 1
            key = (seq, row["rgb_shard_paths"][-1], row["rgb_member_paths"][-1])
            if key not in cameras:
                counts[seq]["missing_calibration"] += 1
                continue
            indices.append(by_token[token])
            matrices.append(cameras[key])
            boxes.append(row["boxes_xyxy"][-1])
            sequences.append(seq)
        if indices:
            corners = np.asarray(
                [pair[-1] for pair in labels["box3d_Fcam"].take(indices).to_pylist()]
            )
            valid, iou, error, exact = compare_boxes(
                corners, np.asarray(matrices).reshape(-1, 3, 3), np.asarray(boxes, dtype=float)
            )
            for i in np.flatnonzero(valid):
                values[sequences[i]].append((iou[i], error[i], exact[i]))
    if seen != set(by_token):
        raise ValueError("Input and annotation token sets differ")
    all_values = [v for group in values.values() for v in group]
    summary = summarize(all_values)
    missing = sum(n["missing_calibration"] for n in counts.values())
    return {
        "population": "TRAIN40, last supplied image box per query; target TTC columns excluded",
        "offered": len(seen),
        "missing_calibration": missing,
        "invalid_projection_or_box": len(seen) - missing - len(all_values),
        "summary": summary,
        "by_sequence": {s: {**counts[s], **summarize(values[s])} for s in sorted(counts)},
        "input_sha256": {name: digest(path) for name, path in paths.items()},
        "script_sha256": digest(Path(__file__)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--garl-root", type=Path, required=True)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.garl_root, args.eap_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("offered", "missing_calibration", "invalid_projection_or_box", "summary")
            }
        )
    )
