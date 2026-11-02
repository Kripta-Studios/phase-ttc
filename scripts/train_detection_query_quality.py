"""Fit the two registered quality heads using verified TRAIN predictions and raw labels."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.query_quality import (  # noqa: E402
    ARMS,
    fit_head,
    quality_targets,
    query_features,
)
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def main() -> None:
    """Require TRAIN ownership and immutable inputs; save failures without promotion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    panel = json.loads(args.panel.read_text())
    plan = json.loads(args.plan.read_text())
    if (
        panel["plan_sha256"] != digest(args.plan)
        or plan["seed"] != 7
        or plan["arms"] != list(ARMS)
        or plan["train_frames"] != 4096
    ):
        raise ValueError("Plan differs from preregistered campaign")
    split_path = ROOT / "data/splits/eap_detection_candidate_v1.json"
    if digest(split_path) != panel["split_sha256"]:
        raise ValueError("Split differs")
    train_sequences = json.loads(split_path.read_text())["assignments"]["train"]
    tokens = panel["roles"]["train"]
    if (
        len(tokens) != 4096
        or len(tokens) != len(set(tokens))
        or any(t.split(":")[0] not in train_sequences for t in tokens)
    ):
        raise ValueError("Training population ownership differs")
    gt_path = args.panel.parent / "gt_train.json"
    if digest(gt_path) != panel["gt_train_sha256"]:
        raise ValueError("Raw TRAIN labels differ")
    gt = json.loads(gt_path.read_text())
    if gt["role"] != "train" or set(gt["ground_truth"]) != set(tokens):
        raise ValueError("Labels belong to another population")
    intake_path = args.inference / "INTAKE.json"
    intake = json.loads(intake_path.read_text())
    complete_path = args.inference / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    if (
        intake["role"] != "train"
        or intake["panel_sha256"] != digest(args.panel)
        or complete["intake_sha256"] != digest(intake_path)
        or complete["frames"] != len(tokens)
    ):
        raise ValueError("TRAIN inference contract differs")
    sources = {
        p.relative_to(ROOT).as_posix(): digest(p)
        for p in [
            Path(__file__),
            *[
                ROOT / f"src/e_jepa_ttc/detection/{name}.py"
                for name in (
                    "query_quality",
                    "quality_ranking",
                    "benchmark_evaluation",
                    "range_campaign",
                )
            ],
        ]
    }
    args.output.mkdir(parents=True)
    for name in sources:
        target = args.output / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    provenance = {
        "role": "train",
        "panel_sha256": digest(args.panel),
        "plan_sha256": digest(args.plan),
        "checkpoint_sha256": panel["checkpoint_sha256"],
        "split_sha256": panel["split_sha256"],
        "gt_train_sha256": digest(gt_path),
        "inference_complete_sha256": digest(complete_path),
        "sources": sources,
        "train_frames": len(tokens),
        "test_labels_read": False,
    }
    write_json(args.output / "CONTRACT.json", provenance)
    matrices, targets, seen = [], [], set()
    for name, expected in sorted(complete["shards"].items()):
        folder = args.inference / "shards" / name
        if digest(folder / "COMPLETE.json") != expected:
            raise ValueError("Inference shard receipt differs")
        shard = json.loads((folder / "COMPLETE.json").read_text())
        for filename, sha in shard["files"].items():
            if digest(folder / filename) != sha:
                raise ValueError("Inference shard bytes differ")
        payload = json.loads((folder / "predictions.json").read_text())
        sidecar = json.loads((folder / "sidecar.json").read_text())
        predictions = payload["predictions"]
        if (
            set(predictions) != set(shard["tokens"])
            or seen.intersection(predictions)
            or not set(predictions).issubset(tokens)
            or sidecar["prediction_output_sha256"] != shard["files"]["predictions.json"]
            or sidecar["checkpoint_sha256"] != panel["checkpoint_sha256"]
            or payload["checkpoint_sha256"] != panel["checkpoint_sha256"]
            or payload["annotations_returned"] != 0
            or payload["targets_argument"] is not None
        ):
            raise ValueError("Inference population, checkpoint or label-free binding differs")
        _, features = query_features(predictions, sidecar)
        matrices.append(features)
        targets.append(
            quality_targets(predictions, {t: gt["ground_truth"][t] for t in predictions})
        )
        seen.update(predictions)
    if seen != set(tokens):
        raise ValueError("Incomplete TRAIN prediction coverage")
    x, y = np.concatenate(matrices), np.concatenate(targets)
    del matrices, targets, gt
    torch.set_num_threads(4)
    write_json(
        args.output / "ENVIRONMENT.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "host": platform.node(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "cpu_threads": 4,
        },
    )
    summaries = {}
    for arm in ARMS:
        folder = args.output / arm
        folder.mkdir()

        def progress(row: dict[str, object], log_dir: Path = folder, name: str = arm) -> None:
            with (log_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps({"arm": name, **row}), flush=True)

        try:
            bundle = fit_head(
                x,
                y,
                arm=arm,
                provenance=provenance,
                progress=progress,
                maximum_seconds=plan["training_max_seconds_per_arm"],
            )
        except Exception as error:
            write_json(
                folder / "FAILED.json", {"error": type(error).__name__, "message": str(error)}
            )
            raise
        for name, expected in sources.items():
            if digest(ROOT / name) != expected:
                raise ValueError("Training source changed")
        torch.save(bundle, folder / "head.pt")
        summary = {
            "arm": arm,
            "queries": len(x),
            "frames": len(tokens),
            "positive_weight": bundle["positive_weight"].tolist(),
            "target_mass": bundle["target_mass"].tolist(),
            "history": bundle["history"],
            "head_sha256": digest(folder / "head.pt"),
            "promotion": False,
        }
        write_json(folder / "COMPLETE.json", summary)
        summaries[arm] = summary
    write_json(
        args.output / "COMPLETE.json",
        {"contract_sha256": digest(args.output / "CONTRACT.json"), "arms": summaries},
    )


if __name__ == "__main__":
    main()
