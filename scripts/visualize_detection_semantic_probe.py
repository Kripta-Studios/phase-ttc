"""Plot retained TRAIN semantic predictions beside projected cuboids for diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_detection_semantics_cpu import projected_box  # noqa: E402

from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset  # noqa: E402
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def main() -> None:
    """Render diagnostic crops only; cropping is never an inference input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = json.loads((args.probe / "RESULTS.json").read_text())
    plan = json.loads((args.probe / "PLAN.json").read_text())
    intake = json.loads((args.probe / "MODEL_INTAKE.json").read_text())
    if (
        result["status"] != "COMPLETE_CPU_PROBE"
        or result["plan_sha256"] != digest(args.probe / "PLAN.json")
        or plan["panel_sha256"] != digest(args.panel)
    ):
        raise ValueError("Probe lineage differs")
    panel = json.loads(args.panel.read_text())
    gt_path = args.panel.parent / "gt_train.json"
    if digest(gt_path) != panel["gt_train_sha256"]:
        raise ValueError("TRAIN labels changed")
    gt = json.loads(gt_path.read_text())["ground_truth"]
    tokens = plan["positive_tokens"]
    count = sum(b["class_name"] == "bicycle" for t in tokens for b in gt[t])
    fig, axes = plt.subplots(
        (count + 1) // 2, 2, figsize=(12, 3.1 * ((count + 1) // 2)), layout="constrained"
    )
    metadata = []
    position = 0
    with EAPRGBDataset(panel["dataset_root"], tokens, max_open_shards=16) as dataset:
        for index, token in enumerate(tokens):
            raw = dataset[index]
            name = f"{index:02d}.json"
            path = args.probe / "predictions" / name
            if digest(path) != result["predictions"][name]:
                raise ValueError("Saved prediction changed")
            prediction = json.loads(path.read_text())
            if prediction["token"] != token or prediction["input_sha256"] != raw.member_sha256:
                raise ValueError("RGB/semantic binding differs")
            boxes = prediction["prediction"]
            for box in gt[token]:
                if box["class_name"] != "bicycle":
                    continue
                rect = projected_box(
                    box, raw.intrinsic, raw.event_from_ego, raw.image.shape[1], raw.image.shape[0]
                )
                if rect is None:
                    continue
                ax = axes.flat[position]
                position += 1
                x0, y0, x1, y1 = rect
                padding_x, padding_y = max(48, (x1 - x0) * 0.6), max(48, (y1 - y0) * 0.6)
                left, right = max(0, x0 - padding_x), min(raw.image.shape[1] - 1, x1 + padding_x)
                top, bottom = max(0, y0 - padding_y), min(raw.image.shape[0] - 1, y1 + padding_y)
                ax.imshow(raw.image)
                ax.set_xlim(left, right)
                ax.set_ylim(bottom, top)
                ax.add_patch(
                    Rectangle(
                        (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#ff3939", linewidth=2
                    )
                )
                displayed = []
                for coords, label, score in zip(
                    boxes["boxes"], boxes["labels"], boxes["scores"], strict=True
                ):
                    a, b, c, d = coords
                    if score < plan["confidence"] or c < left or a > right or d < top or b > bottom:
                        continue
                    category = intake["categories"][label]
                    ax.add_patch(
                        Rectangle(
                            (a, b), c - a, d - b, fill=False, edgecolor="#00e1d4", linewidth=1.4
                        )
                    )
                    ax.text(
                        max(a, left),
                        max(b, top),
                        f"{category} {score:.2f}",
                        color="black",
                        fontsize=8,
                        bbox={"facecolor": "#00e1d4", "alpha": 0.8, "pad": 1},
                    )
                    displayed.append({"category": category, "score": score, "xyxy": coords})
                ax.set_title(token, fontsize=9)
                ax.tick_params(labelsize=7)
                metadata.append(
                    {
                        "token": token,
                        "gt_projected_xyxy": rect,
                        "ego_center_m": box["translation_m"],
                        "displayed_coco": displayed,
                    }
                )
    for ax in list(axes.flat)[position:]:
        ax.set_visible(False)
    fig.suptitle(
        "TRAIN diagnostic: red = projected bicycle cuboid; cyan = COCO detection (score ≥ 0.5)",
        fontsize=12,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    fig.savefig(args.output / "bicycle_crops.png", dpi=150)
    plt.close(fig)
    write_json(
        args.output / "VISUALS.json",
        {
            "results_sha256": digest(args.probe / "RESULTS.json"),
            "panels": metadata,
            "image_sha256": digest(args.output / "bicycle_crops.png"),
            "source_sha256": digest(Path(__file__)),
            "diagnostic_only": True,
        },
    )


if __name__ == "__main__":
    main()
