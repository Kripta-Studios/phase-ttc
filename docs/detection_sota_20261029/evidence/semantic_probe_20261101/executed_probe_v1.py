"""Bounded TRAIN-only RGB semantic probe; no model fitting or eAP quality claim."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset  # noqa: E402
from e_jepa_ttc.detection.geometry_contract import original_bundle_corners  # noqa: E402
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402
from e_jepa_ttc.detection.raw_diagnostics import assignment  # noqa: E402


def projected_box(
    box: dict, intrinsic: np.ndarray, transform: np.ndarray, width: int, height: int
) -> list[float] | None:
    """Project the released ego cuboid; this is not a native visible-object 2D label."""
    x, y, z = box["translation_m"]
    length, width_m, height_m = box["size_lwh_m"]
    corners = original_bundle_corners([x, y, z, length, height_m, width_m, box["yaw_rad"]])
    camera = corners @ transform[:3, :3].T + transform[:3, 3]
    if (camera[:, 2] <= 0.1).any():
        return None
    pixels = camera @ intrinsic.T
    pixels = pixels[:, :2] / pixels[:, 2:]
    low = np.maximum(pixels.min(axis=0), [0, 0])
    high = np.minimum(pixels.max(axis=0), [width - 1, height - 1])
    if (high <= low).any():
        return None
    return [*low.tolist(), *high.tolist()]


def main() -> None:
    """Inspect sixteen fixed TRAIN images with a pinned public COCO detector on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("CPU probe requires CUDA_VISIBLE_DEVICES=-1")
    if torchvision.__version__ != "0.26.0+cu128":
        raise ValueError("This probe is pinned to the inspected torchvision build")
    args.output.mkdir(parents=True, exist_ok=False)
    panel = json.loads(args.panel.read_text())
    gt_path = args.panel.parent / "gt_train.json"
    if digest(gt_path) != panel["gt_train_sha256"] or panel["test_labels_read"] is not False:
        raise ValueError("Bound TRAIN labels differ")
    gt = json.loads(gt_path.read_text())
    if gt["role"] != "train":
        raise ValueError("Only TRAIN may be probed")
    positive = sorted(
        t
        for t, rows in gt["ground_truth"].items()
        if any(b["class_name"] == "bicycle" for b in rows)
    )
    negative = sorted(set(panel["roles"]["train"]) - set(positive))
    if len(positive) < 8 or len(negative) < 8:
        raise ValueError("Insufficient fixed probe population")
    positives = [positive[int(i)] for i in np.linspace(0, len(positive) - 1, 8)]
    negatives = random.Random(7).sample(negative, 8)
    tokens = positives + negatives
    if len(set(tokens)) != 16 or any(panel["media"][t]["role"] != "train" for t in tokens):
        raise ValueError("Probe population is not sixteen distinct TRAIN frames")
    plan = {
        "schema": "phase_semantic_cpu_probe_v1",
        "panel_sha256": digest(args.panel),
        "positive_tokens": positives,
        "negative_tokens": negatives,
        "sampling": "Eight evenly spaced sorted bicycle TRAIN tokens; "
        "eight random non-bicycle tokens, seed 7",
        "confidence": 0.5,
        "projected_cuboid_iou": 0.3,
        "maximum_inference_seconds": 600,
        "model": "torchvision fasterrcnn_resnet50_fpn_v2 COCO_V1",
        "purpose": "Check semantic signal feasibility only; no eAP mAP or generalization estimate",
        "selection_labels_read": False,
        "test_labels_read": False,
        "source_sha256": digest(Path(__file__)),
    }
    write_json(args.output / "PLAN.json", plan)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    weight_path = args.model_dir / Path(weights.url).name
    if not weight_path.exists():
        torch.hub.download_url_to_file(
            weights.url, str(weight_path), hash_prefix="dd69338a", progress=False
        )
    if not digest(weight_path).startswith("dd69338a"):
        raise ValueError("Official weight hash prefix differs")
    license_url = "https://raw.githubusercontent.com/pytorch/vision/v0.26.0/LICENSE"
    license_path = args.model_dir / "TORCHVISION_LICENSE.txt"
    if not license_path.exists():
        with urllib.request.urlopen(license_url, timeout=30) as response:
            license_path.write_bytes(response.read())
    torch.set_num_threads(2)
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=91)
    state = torch.load(weight_path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    categories = weights.meta["categories"]
    if categories[2] != "bicycle":
        raise ValueError("Inspected COCO class mapping changed")
    source_files = {
        name: digest(Path(inspect.getfile(module)))
        for name, module in (
            ("builder", fasterrcnn_resnet50_fpn_v2.__wrapped__),
            ("roi_heads", type(model.roi_heads)),
            ("rpn", type(model.rpn)),
            ("transform", type(model.transform)),
        )
    }
    transform: Any = model.transform
    write_json(
        args.output / "MODEL_INTAKE.json",
        {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "url": weights.url,
            "weights_sha256": digest(weight_path),
            "categories": categories,
            "code_source_sha256": source_files,
            "code_license_url": license_url,
            "code_license_sha256": digest(license_path),
            "pretraining_dataset": "COCO",
            "weights_redistribution_not_assessed": True,
            "benchmark_external_data_admission_certified": False,
            "image_min_size": list(transform.min_size),
            "image_max_size": transform.max_size,
            "box_score_threshold": model.roi_heads.score_thresh,
            "box_nms_threshold": model.roi_heads.nms_thresh,
            "box_detections_per_image": model.roi_heads.detections_per_img,
            "model_input": "RGB float32 CHW in[0,1]; images only, no targets or calibration",
        },
    )
    start = time.perf_counter()
    rows = []
    try:
        with EAPRGBDataset(panel["dataset_root"], tokens, max_open_shards=16) as dataset:
            for index, token in enumerate(tokens):
                if time.perf_counter() - start > plan["maximum_inference_seconds"]:
                    raise TimeoutError("CPU semantic probe budget exhausted")
                raw = dataset[index]
                if raw.member_sha256 != panel["media"][token]["member_sha256"]:
                    raise ValueError("Admitted RGB changed")
                tensor = torch.from_numpy(raw.image.copy()).permute(2, 0, 1).float() / 255
                before = time.perf_counter()
                with torch.inference_mode():
                    prediction = model([tensor])[0]
                elapsed = time.perf_counter() - before
                if not all(torch.isfinite(v).all() for v in prediction.values()):
                    raise ValueError("Nonfinite semantic detector output")
                payload = {k: v.tolist() for k, v in prediction.items()}
                write_json(
                    args.output / "predictions" / f"{index:02d}.json",
                    {
                        "token": token,
                        "input_sha256": raw.member_sha256,
                        "prediction": payload,
                        "inference_seconds": elapsed,
                        "annotations_in_forward": 0,
                    },
                )
                target_boxes = []
                for box in gt["ground_truth"][token]:
                    if box["class_name"] != "bicycle":
                        continue
                    rectangle = projected_box(
                        box,
                        raw.intrinsic,
                        raw.event_from_ego,
                        raw.image.shape[1],
                        raw.image.shape[0],
                    )
                    if rectangle is not None:
                        target_boxes.append(rectangle)
                choose = (prediction["labels"] == 2) & (prediction["scores"] >= plan["confidence"])
                boxes = prediction["boxes"][choose]
                expected = torch.tensor(target_boxes, dtype=torch.float32).reshape(-1, 4)
                iou = box_iou(boxes, expected).numpy()
                pairs = assignment(1 - iou, iou >= plan["projected_cuboid_iou"])
                rows.append(
                    {
                        "token": token,
                        "positive_frame": token in positives,
                        "raw_bicycle_gt": sum(
                            b["class_name"] == "bicycle" for b in gt["ground_truth"][token]
                        ),
                        "projectable_bicycle_gt": len(target_boxes),
                        "bicycle_detections": len(boxes),
                        "matches": len(pairs),
                        "inference_seconds": elapsed,
                        "matched_ious": [float(iou[i, j]) for i, j in pairs],
                    }
                )
                write_json(
                    args.output / "STATUS.json",
                    {
                        "status": "CPU_INFERENCE",
                        "frames": len(rows),
                        "total": len(tokens),
                        "elapsed_seconds": time.perf_counter() - start,
                    },
                )
        if torch.cuda.is_initialized() or any(p.is_cuda for p in model.parameters()):
            raise ValueError("CPU probe unexpectedly used CUDA")
        write_json(
            args.output / "RESULTS.json",
            {
                "status": "COMPLETE_CPU_PROBE",
                "plan_sha256": digest(args.output / "PLAN.json"),
                "model_intake_sha256": digest(args.output / "MODEL_INTAKE.json"),
                "rows": rows,
                "matched_bicycle_targets": sum(r["matches"] for r in rows),
                "projectable_bicycle_targets": sum(r["projectable_bicycle_gt"] for r in rows),
                "bicycle_detections_on_negative_frames": sum(
                    r["bicycle_detections"] for r in rows if not r["positive_frame"]
                ),
                "predictions": {
                    p.name: digest(p) for p in (args.output / "predictions").glob("*.json")
                },
                "cuda_initialized": False,
                "mAP_measured": False,
                "promotion": False,
                "limitations": "Sixteen enriched TRAIN frames, not held out; projected 3D "
                "cuboids are not visible 2D labels. No native tricycle or other_vehicle class in COCO.",
            },
        )
        write_json(args.output / "STATUS.json", {"status": "COMPLETE", "frames": len(rows)})
    except Exception as error:
        write_json(args.output / "FAILURE.json", {"error": str(error), "status": "FAILED"})
        raise


if __name__ == "__main__":
    main()
