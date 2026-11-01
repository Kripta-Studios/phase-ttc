"""Audit real public RGB, target encoding and target-free decode on a fixed panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_detection_benchmark_local import read, sha, write  # noqa: E402

from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset  # noqa: E402
from e_jepa_ttc.detection.geometry_contract import original_bundle_corners  # noqa: E402
from e_jepa_ttc.detection.monodgp_eap_adapter import decode_eap, prepare_sample  # noqa: E402
from e_jepa_ttc.detection.raw_diagnostics import interval  # noqa: E402


def main() -> None:
    """Select first occurrences per sequence and class/depth, then test real samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    complete = read(args.diagnostics / "COMPLETE.json")
    if complete["status"] != "VERIFIED":
        raise ValueError("Complete verified diagnostics required")
    panel = {}
    coverage = set()
    labels = {}
    for folder_name, hashes in complete["input_sha256"].items():
        path = args.shards / folder_name / "gt.json"
        if sha(path) != hashes["gt.json"]:
            raise ValueError("GT changed")
        receipt_path = args.shards / folder_name / "gt_receipt.json"
        if sha(receipt_path) != hashes["gt_receipt.json"]:
            raise ValueError("GT receipt changed")
        labels.update(read(receipt_path)["label_file_sha256"])
        for token, frame in read(path)["ground_truth_by_token"].items():
            rotation = np.asarray(frame["event_from_ego"])
            strata = {f"sequence/{token.split(':')[0]}"}
            for box in frame["eligible"] + frame["ignored"]:
                depth = (rotation[:3, :3] @ box["translation_m"] + rotation[:3, 3])[2]
                strata.add(f"{box['class_name']}/{interval(depth, (0.0, 60.0, 150.0))}")
            new = strata - coverage
            if new:
                panel[token] = {"frame": frame, "new_strata": sorted(new)}
                coverage.update(new)
    for name, digest in labels.items():
        if not name.startswith("data/train/") or sha(args.dataset / name) != digest:
            raise ValueError("Public label bytes changed")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    rows = []
    with EAPRGBDataset(args.dataset, list(panel)) as dataset:
        for index, token in enumerate(panel):
            raw = dataset[index]
            frame = panel[token]["frame"]
            if (
                not np.allclose(raw.intrinsic, frame["intrinsic"])
                or not np.allclose(raw.event_from_ego, frame["event_from_ego"])
                or list(raw.image.shape[1::-1]) != frame["image_wh"]
            ):
                raise ValueError("Calibration or image resolution differs from saved shard")
            if len(raw.annotations) != frame["raw_count"]:
                raise ValueError("Raw annotation count mismatch")
            prepared = prepare_sample(raw)
            target = prepared.target
            n = len(target["labels"])
            logits, angle = torch.full((1, n, 8), -20.0), torch.zeros(1, n, 24)
            for i in range(n):
                logits[0, i, target["labels"][i]] = 20
                heading = int(target["heading_bin"][i].item())
                angle[0, i, heading] = 20
                angle[0, i, 12 + heading] = target["heading_res"][i].item()
            outputs = {
                "pred_logits": logits,
                "pred_angle": angle,
                "pred_boxes": target["boxes_3d"][None],
                "pred_depth": torch.cat(
                    (target["depth"], torch.zeros_like(target["depth"])), dim=1
                )[None],
                "pred_3d_dim": target["size_3d"][None],
            }
            decoded, _ = decode_eap(
                outputs, 0, prepared.calibration, raw.event_from_ego, prepared.image_size
            )
            errors = []
            for original, predicted in zip(prepared.native_ground_truth, decoded, strict=True):
                if original["class_name"] != predicted["class_name"]:
                    raise ValueError("Class roundtrip failure")
                position = float(
                    np.max(np.abs(np.array(original["translation_m"]) - predicted["translation_m"]))
                )
                size = float(
                    np.max(np.abs(np.array(original["size_lwh_m"]) - predicted["size_lwh_m"]))
                )
                yaw = float(
                    abs(
                        (
                            cast(float, original["yaw_rad"])
                            - cast(float, predicted["yaw_rad"])
                            + np.pi
                        )
                        % (2 * np.pi)
                        - np.pi
                    )
                )
                if max(position, size, yaw) > 1e-4:
                    raise ValueError(f"Roundtrip failure: {token}: {position}, {size}, {yaw}")
                errors.append([position, size, yaw])
            image = Image.fromarray(raw.image)
            draw = ImageDraw.Draw(image)
            for annotation in raw.annotations:
                camera = (
                    original_bundle_corners(annotation["bbox_3d_ego"])
                    @ raw.event_from_ego[:3, :3].T
                    + raw.event_from_ego[:3, 3]
                )
                if (camera[:, 2] <= 0.1).any():
                    continue
                pixels = camera @ raw.intrinsic.T
                pixels = pixels[:, :2] / pixels[:, 2:]
                color = "lime" if camera[:, 2].mean() < 60 else "orange"
                for a, b in (
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 0),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 4),
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                ):
                    draw.line([tuple(pixels[a]), tuple(pixels[b])], fill=color, width=2)
            draw.text(
                (10, 10),
                token + "  GT: green <60m / orange >=60m",
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
            image.save(args.output / f"{index:03d}.jpg", quality=90)
            rows.append(
                {
                    "token": token,
                    "strata": panel[token]["new_strata"],
                    "rgb_sha256": raw.member_sha256,
                    "image": f"{index:03d}.jpg",
                    "admission": prepared.report,
                    "roundtrip_max_xyz_size_yaw": np.max(errors, axis=0).tolist()
                    if errors
                    else None,
                }
            )
            print(f"{index + 1}/{len(panel)} {token} admitted={n}", flush=True)
    write(
        args.output / "GEOMETRY_PANEL.json",
        {
            "schema": "eap_real_geometry_panel_v1",
            "selection": "first frame per sequence and class x depth <0/0-60/60-150/>=150",
            "frames": rows,
            "source_sha256": {
                str(p.relative_to(ROOT)): sha(p)
                for p in (Path(__file__), ROOT / "src/e_jepa_ttc/detection/monodgp_eap_adapter.py")
            },
            "limitations": [
                "Roundtrip proves internal consistency, not independent sensor alignment.",
                "Green/orange use camera corner mean depth, not admission or visibility.",
                "Visual inspection is separate; projection does not establish physical truth.",
            ],
            "test_labels_read": False,
        },
    )


if __name__ == "__main__":
    main()
