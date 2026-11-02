"""Measure fixed geometric depth replacements on retained TRAIN predictions."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.benchmark_evaluation import BenchmarkEvaluator
from e_jepa_ttc.detection.eap_rgb_data import finite_matrix
from e_jepa_ttc.detection.monodgp_eap_adapter import resized_intrinsic
from e_jepa_ttc.detection.predicted_geometry_depth import replace_depths
from e_jepa_ttc.detection.range_campaign import digest, write_json


def main() -> None:
    """Use saved outputs only; no model is instantiated and no selection labels are read."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--inference", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or args.output.exists():
        raise ValueError("Require CPU-only execution and a new output directory")
    torch.set_num_threads(1)
    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    started = time.perf_counter()
    inputs: dict[str, str] = {}

    def read(path: Path, expected: str | None = None) -> dict:
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Changed input: {path}")
        inputs[path.as_posix()] = actual
        return json.loads(path.read_text())

    complete = read(args.inference / "COMPLETE.json")
    intake = read(args.inference / "INTAKE.json", complete["intake_sha256"])
    panel = read(args.panel / "PANEL.json", intake["panel_sha256"])
    gt = read(args.panel / "gt_train.json", panel["gt_train_sha256"])
    split = read(ROOT / "data/splits/eap_detection_candidate_v1.json", panel["split_sha256"])
    prereg = read(Path(panel["release_root"]) / "detector/runtime_preregistration.json")
    if (
        complete["role"] != "train"
        or intake["role"] != "train"
        or gt["role"] != "train"
        or complete["frames"] != 4096
        or intake["checkpoint_sha256"] != panel["checkpoint_sha256"]
        or panel["test_labels_read"]
        or prereg["config"]["model"]["width"] != 640
        or prereg["config"]["model"]["height"] != 384
    ):
        raise ValueError("Retained inference ownership or image dimensions differ")
    tokens = panel["roles"]["train"]
    if len(tokens) != 4096 or len(set(tokens)) != 4096 or set(tokens) != set(gt["ground_truth"]):
        raise ValueError("TRAIN population differs")
    if not {t.split(":")[0] for t in tokens} <= set(split["assignments"]["train"]):
        raise ValueError("Non-TRAIN sequence entered the diagnostic")
    media_path = args.dataset / "data/train.parquet"
    if digest(media_path) != panel["source_media_sha256"]:
        raise ValueError("Canonical metadata changed")
    inputs[media_path.as_posix()] = panel["source_media_sha256"]
    metadata = pd.read_parquet(
        media_path,
        filters=[("sample_token", "in", tokens)],
        columns=["sample_token", "sequence_id", "K_event", "T_event_ego"],
    )
    frames = {row["sample_token"]: row for row in metadata.to_dict("records")}
    if len(metadata) != len(tokens) or set(frames) != set(tokens):
        raise ValueError("Missing or repeated metadata frame")
    names = [
        Path(__file__).relative_to(ROOT).as_posix(),
        *[
            f"src/e_jepa_ttc/detection/{name}.py"
            for name in (
                "predicted_geometry_depth",
                "cuboid_edge_depth",
                "monodgp_eap_adapter",
                "geometry_contract",
                "eap_rgb_data",
                "benchmark_evaluation",
                "range_campaign",
                "development_metrics",
            )
        ],
    ]
    sources = {name: digest(ROOT / name) for name in names}
    args.output.mkdir(parents=True)
    contract = {
        "schema": "phase_predicted_geometry_train_diagnostic_v1",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sources": sources,
        "inputs_at_start": dict(inputs),
        "arms": ["baseline", "height_only", "cuboid_edges_only"],
        "boundary_exclusion_pixels": 1,
        "invalid_geometry": "retain_original_detection",
        "residual_policy": "replacement arms omit the learned residual entirely",
        "maximum_seconds": 600,
        "fit_parameters": False,
        "new_detector_inference": False,
        "selection_labels": False,
        "test_labels": False,
        "score": "original_class_sigmoid_without_ranker",
        "promotion": False,
        "scope": "TRAIN diagnostic; cannot establish generalization or justify promotion",
    }
    write_json(args.output / "PLAN.json", contract)
    for name in names:
        path = args.output / "code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    evaluators = {arm: BenchmarkEvaluator() for arm in contract["arms"]}
    counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    seen: set[str] = set()
    output_shards = {}
    try:
        for name, expected in sorted(complete["shards"].items()):
            folder = args.inference / "shards" / name
            receipt = read(folder / "COMPLETE.json", expected)
            predictions = read(folder / "predictions.json", receipt["files"]["predictions.json"])
            sidecar = read(folder / "sidecar.json", receipt["files"]["sidecar.json"])
            if (
                receipt["intake_sha256"] != complete["intake_sha256"]
                or predictions["targets_argument"] is not None
                or predictions["annotations_returned"] != 0
                or predictions["checkpoint_sha256"] != intake["checkpoint_sha256"]
                or sidecar["checkpoint_sha256"] != intake["checkpoint_sha256"]
                or sidecar["prediction_output_sha256"] != receipt["files"]["predictions.json"]
                or set(predictions["predictions"]) != set(receipt["tokens"])
                or seen.intersection(receipt["tokens"])
            ):
                raise ValueError("Retained shard lineage differs")
            exposed = {f["sample_token"]: f["tensors"] for f in sidecar["frames"]}
            if len(sidecar["frames"]) != len(exposed) or set(exposed) != set(receipt["tokens"]):
                raise ValueError("Sidecar population differs")
            saved_variants = {}
            for token in receipt["tokens"]:
                if time.perf_counter() - started > 600:
                    raise TimeoutError("TRAIN predicted-geometry diagnostic budget exceeded")
                row, media = frames[token], panel["media"][token]
                if media["role"] != "train" or row["sequence_id"] != media["sequence_id"]:
                    raise ValueError("Metadata ownership differs")
                width, height = media["image_wh"]
                k = resized_intrinsic(
                    finite_matrix(row["K_event"], (3, 3)), (width, height), (640, 384)
                )
                calibration = np.zeros((3, 4), dtype=np.float32)
                calibration[:, :3] = k
                transform = finite_matrix(row["T_event_ego"], (4, 4))
                outputs = {
                    key: torch.tensor(value["values"], dtype=torch.float32)[None]
                    for key, value in exposed[token].items()
                }
                original = predictions["predictions"][token]
                variants, stat = replace_depths(
                    outputs,
                    calibration,
                    transform,
                    np.array([640, 384], dtype=np.float32),
                    original,
                )
                counts.update(stat)
                saved_variants[token] = variants
                targets = gt["ground_truth"][token]
                class_counts.update(b["class_name"] for b in targets)
                for arm, boxes in {"baseline": original, **variants}.items():
                    evaluators[arm].add_frame(token, boxes, targets)
                seen.add(token)
            output_path = args.output / "shards" / f"{name}.json.gz"
            output_path.parent.mkdir(exist_ok=True)
            output_path.write_bytes(
                gzip.compress(json.dumps(saved_variants, allow_nan=False).encode(), mtime=0)
            )
            output_shards[output_path.name] = digest(output_path)
            write_json(
                args.output / "STATUS.json",
                {
                    "status": "CPU_TRAIN_DIAGNOSTIC",
                    "frames": len(seen),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
        if seen != set(tokens) or dict(class_counts) != panel["raw_train_class_counts"]:
            raise ValueError("Evaluated TRAIN population differs")
        if torch.cuda.is_initialized() or sources != {name: digest(ROOT / name) for name in names}:
            raise ValueError("CUDA initialized or source changed")
        results = {name: evaluator.summarize() for name, evaluator in evaluators.items()}
        write_json(args.output / "RESULTS.json", results)
        write_json(
            args.output / "COMPLETE.json",
            {
                "status": "VERIFIED_TRAIN_DIAGNOSTIC_ONLY",
                "frames": len(seen),
                "counts": dict(counts),
                "input_sha256": inputs,
                "output_shards": output_shards,
                "plan_sha256": digest(args.output / "PLAN.json"),
                "results_sha256": digest(args.output / "RESULTS.json"),
                "elapsed_seconds": time.perf_counter() - started,
                "cuda_initialized": False,
                "baseline_decode_exact": True,
                "promotion": False,
            },
        )
        write_json(args.output / "STATUS.json", {"status": "COMPLETE", "frames": len(seen)})
        print(json.dumps({name: value["mAP"] for name, value in results.items()}))
    except Exception as error:
        write_json(args.output / "FAILURE.json", {"error": str(error), "frames": len(seen)})
        write_json(args.output / "STATUS.json", {"status": "FAILED", "frames": len(seen)})
        raise


if __name__ == "__main__":
    main()
