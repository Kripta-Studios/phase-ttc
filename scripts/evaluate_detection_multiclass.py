"""Evaluate all saved class logits on fixed geometry, with resumable match records."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_detection_benchmark_local import verify_shard_coverage  # noqa: E402

from e_jepa_ttc.detection.benchmark_evaluation import (  # noqa: E402
    CLASSES,
    PROFILE_SHA256,
    BenchmarkEvaluator,
)
from e_jepa_ttc.detection.multiclass_proposals import (  # noqa: E402
    expand_class_hypotheses,
    select_global_top50,
)
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def main() -> None:
    """Use every public selection frame; do not fit thresholds or access private labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["recipe"] not in (
        "all_8_sigmoid_classes_per_existing_geometry_v1",
        "global_top50_sigmoid_classes_per_existing_geometry_v1",
    ):
        raise ValueError("Unexpected expansion recipe")
    top50 = plan["recipe"].startswith("global_top50_")
    if top50 and plan.get("fixed_topk") != 50:
        raise ValueError("Only the preregistered fixed top-50 budget is admitted")
    folders, historical = verify_shard_coverage(args.shards)
    aligned_path = args.aligned / "COMPARISON.json"
    aligned = json.loads(aligned_path.read_text())
    if aligned["profile_sha256"] != PROFILE_SHA256 or aligned["samples"] != historical["frames"]:
        raise ValueError("Aligned reference differs")
    sources = [
        Path(__file__),
        ROOT / "scripts/evaluate_detection_benchmark_local.py",
        *[
            ROOT / f"src/e_jepa_ttc/detection/{name}.py"
            for name in (
                "multiclass_proposals",
                "quality_ranking",
                "benchmark_evaluation",
                "range_campaign",
            )
        ],
    ]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    intake = {
        "schema": "phase_multiclass_replay_v1",
        "plan_sha256": digest(args.plan),
        "aligned_sha256": digest(aligned_path),
        "checkpoint_sha256": aligned["checkpoint_sha256"],
        "profile_sha256": PROFILE_SHA256,
        "sources": source_hashes,
        "recipe": plan["recipe"],
        "test_labels_read": False,
        "new_training": False,
        "new_inference": False,
        "new_geometry": False,
        "changes_class_hypothesis_pool": True,
        "argmax_confidence": "exact historical float32 value",
        "other_confidences": "stable float64 sigmoid of saved float32 logit",
        "fixed_global_topk": 50 if top50 else None,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    intake_path = args.output / "INTAKE.json"
    if intake_path.exists():
        if json.loads(intake_path.read_text()) != intake:
            raise ValueError("Existing experiment contract differs")
    else:
        if any(args.output.iterdir()):
            raise ValueError("Unowned output directory")
        write_json(intake_path, intake)
    lock = args.output / "WRITER.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    total = BenchmarkEvaluator()
    started = time.perf_counter()
    shard_hashes = {}
    for number, folder in enumerate(folders, 1):
        old_complete = folder / "COMPLETE.json"
        if digest(old_complete) != aligned["input_sha256"][folder.name]["COMPLETE.json"]:
            raise ValueError("Historical shard receipt differs")
        old = json.loads(old_complete.read_text())
        source_inputs = {}
        for name in (
            "tokens.json",
            "gt.json",
            "gt_receipt.json",
            "predictions.json",
            "sidecar.json",
        ):
            actual = digest(folder / name)
            if actual != old["files"][name]:
                raise ValueError(f"Source bytes changed: {folder / name}")
            source_inputs[name] = actual
        tokens = json.loads((folder / "tokens.json").read_text())["sample_tokens"]
        if len(tokens) != len(set(tokens)) or len(tokens) != old["frames"]:
            raise ValueError("Invalid shard population")
        output = args.output / "shards" / folder.name
        complete_path = output / "COMPLETE.json"
        if complete_path.exists():
            complete = json.loads(complete_path.read_text())
            if (
                complete["intake_sha256"] != digest(intake_path)
                or complete["input_sha256"] != source_inputs
                or complete["tokens"] != tokens
                or complete["matches_sha256"] != digest(output / "matches.npz")
            ):
                raise ValueError("Cached match records differ")
        else:
            gt_receipt = json.loads((folder / "gt_receipt.json").read_text())
            if gt_receipt["role"] != "selection" or gt_receipt["test_labels_read"] is not False:
                raise ValueError("Only public selection is admitted")
            payload = json.loads((folder / "predictions.json").read_text())
            sidecar = json.loads((folder / "sidecar.json").read_text())
            if (
                sidecar["checkpoint_sha256"] != aligned["checkpoint_sha256"]
                or payload["checkpoint_sha256"] != aligned["checkpoint_sha256"]
                or sidecar["prediction_output_sha256"] != source_inputs["predictions.json"]
                or sidecar["claims"]["ground_truth_access"] is not False
            ):
                raise ValueError("Sidecar binding or label-free provenance differs")
            original = payload["predictions"]
            if set(original) != set(tokens):
                raise ValueError("Prediction population differs")
            expanded = expand_class_hypotheses(original, sidecar)
            del sidecar
            selected = select_global_top50(expanded) if top50 else expanded
            gt = json.loads((folder / "gt.json").read_text())["ground_truth_by_token"]
            evaluator = BenchmarkEvaluator()
            for token in tokens:
                frame = gt[token]
                raw = sorted(frame["eligible"] + frame["ignored"], key=lambda b: b["source_index"])
                if [b["source_index"] for b in raw] != list(range(frame["raw_count"])):
                    raise ValueError("Incomplete raw GT")
                # The argmax sub-pool must reproduce every original box and score exactly.
                for index, box in enumerate(original[token]):
                    block = expanded[token][index * len(CLASSES) : (index + 1) * len(CLASSES)]
                    if [b for b in block if b["class_name"] == box["class_name"]] != [box]:
                        raise ValueError("Expansion altered an original hypothesis")
                    for child in block:
                        for key in box:
                            if key not in ("class_name", "score") and child[key] != box[key]:
                                raise ValueError("Expansion changed physical geometry or velocity")
                evaluator.add_frame(token, selected[token], raw)
            arrays = {
                c: np.concatenate(rows) if rows else np.empty((0, 9))
                for c, rows in evaluator.records.items()
            }
            output.mkdir(parents=True, exist_ok=True)
            with (output / "matches.npz").open("wb") as stream:
                # Keys are the fixed class taxonomy, never the reserved allow_pickle keyword.
                np.savez_compressed(stream, **cast(dict[str, Any], arrays))
            complete = {
                "intake_sha256": digest(intake_path),
                "input_sha256": source_inputs,
                "tokens": tokens,
                "gt_counts": evaluator.counts,
                "pred_counts": evaluator.pred_counts,
                "matches_sha256": digest(output / "matches.npz"),
                "exact_argmax_subpool_parity_before_budget": True,
                "fixed_global_topk": 50 if top50 else None,
            }
            write_json(complete_path, complete)
            del expanded, selected, payload, original, gt, evaluator, arrays
        if total.tokens.intersection(tokens):
            raise ValueError("Repeated sample across shards")
        with np.load(output / "matches.npz", allow_pickle=False) as saved:
            if set(saved.files) != set(CLASSES):
                raise ValueError("Match-record classes differ")
            for cls in CLASSES:
                values = saved[cls]
                if (
                    values.shape != (complete["pred_counts"][cls], 9)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("Invalid compact match records")
                if len(values):
                    if not np.array_equal(values[:, 1], np.arange(len(values))):
                        raise ValueError("Local proposal ordering changed")
                    values[:, 1] += total.pred_counts[cls]
                    total.records[cls].append(values)
                total.counts[cls] += complete["gt_counts"][cls]
                total.pred_counts[cls] += complete["pred_counts"][cls]
        total.tokens.update(tokens)
        shard_hashes[folder.name] = digest(complete_path)
        status = {
            "status": "EVALUATING",
            "shards": number,
            "total_shards": len(folders),
            "frames": len(total.tokens),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(args.output / "STATUS.json", status)
        print(json.dumps(status), flush=True)
    if len(total.tokens) != historical["frames"]:
        raise ValueError("Incomplete evaluation")
    if source_hashes != {str(p.relative_to(ROOT)): digest(p) for p in sources}:
        raise ValueError("Source changed during replay")
    metrics = total.summarize()
    write_json(
        args.output / "RESULTS.json",
        {
            "intake_sha256": digest(intake_path),
            "shard_receipt_sha256": shard_hashes,
            "metrics": metrics,
            "reference_metrics": aligned["variants"],
            "promotion": False,
            "private_scorer_equivalence": False,
        },
    )
    write_json(
        args.output / "STATUS.json",
        {
            "status": "COMPLETE",
            "frames": len(total.tokens),
            "mAP": metrics["mAP"],
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "results_sha256": digest(args.output / "RESULTS.json"),
        },
    )
    lock.unlink()


if __name__ == "__main__":
    main()
