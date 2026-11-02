"""Compare range snapshots with frozen detector/ranker on the identical screen panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_detection_benchmark_local import verify_shard_coverage  # noqa: E402

from e_jepa_ttc.detection.benchmark_evaluation import (  # noqa: E402
    PROFILE_SHA256,
    BenchmarkEvaluator,
)
from e_jepa_ttc.detection.range_campaign import RangeConfig, digest, write_json  # noqa: E402


def main() -> None:
    """Verify external raw GT parity without executing the model or accessing test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = RangeConfig.load(args.config)
    if digest(Path(cfg.population)) != cfg.population_sha256:
        raise ValueError("Panel population differs")
    tokens = json.loads(Path(cfg.population).read_text())["roles"]["selection"]
    manifest_path = Path(cfg.release_root) / "MANIFEST.json"
    if digest(manifest_path) != cfg.release_manifest_sha256:
        raise ValueError("Release manifest differs")
    manifest = json.loads(manifest_path.read_text())
    checkpoint_sha = manifest["files"][manifest["checkpoint"]]
    wanted = set(tokens)
    inputs, ground_truth = {}, {}
    variants = {"frozen_detector": {}, "frozen_ranked": {}}
    folders, _ = verify_shard_coverage(args.shards)
    for folder in folders:
        receipt = json.loads((folder / "COMPLETE.json").read_text())
        if receipt["status"] != "VERIFIED":
            raise ValueError("Unverified historical shard")
        selected = json.loads((folder / "tokens.json").read_text())["sample_tokens"]
        overlap = wanted.intersection(selected)
        if not overlap:
            continue
        inputs[folder.name] = {}
        for name in (
            "tokens.json",
            "gt.json",
            "gt_receipt.json",
            "predictions.json",
            "learned.json",
        ):
            value = digest(folder / name)
            if value != receipt["files"][name]:
                raise ValueError("Historical bytes changed")
            inputs[folder.name][name] = value
        gt_receipt = json.loads((folder / "gt_receipt.json").read_text())
        if gt_receipt["role"] != "selection" or gt_receipt["test_labels_read"] is not False:
            raise ValueError("Wrong GT role")
        gt = json.loads((folder / "gt.json").read_text())["ground_truth_by_token"]
        for token in overlap:
            frame = gt[token]
            rows = sorted(frame["eligible"] + frame["ignored"], key=lambda r: r["source_index"])
            if token in ground_truth or [r["source_index"] for r in rows] != list(
                range(frame["raw_count"])
            ):
                raise ValueError("Duplicate frame or incomplete raw GT")
            ground_truth[token] = rows
        for variant, name in (
            ("frozen_detector", "predictions.json"),
            ("frozen_ranked", "learned.json"),
        ):
            payload = json.loads((folder / name).read_text())
            if payload["checkpoint_sha256"] != checkpoint_sha:
                raise ValueError("Frozen checkpoint differs")
            variants[variant].update({t: payload["predictions"][t] for t in overlap})
    if set(ground_truth) != wanted:
        raise ValueError("Panel missing raw GT")
    expected, experiment_hashes = {}, {}
    for path in args.evaluations:
        name = f"{path.parent.parent.name}/{path.stem}"
        if name in variants:
            raise ValueError("Duplicate evaluation name")
        value = json.loads(path.read_text())
        if (
            value["metrics"]["profile_sha256"] != PROFILE_SHA256
            or value["label_free_forward"] is not True
            or value["gt_population"] != "raw_all"
            or set(value["predictions"]) != wanted
        ):
            raise ValueError("Evaluation protocol or population differs")
        variants[name] = value["predictions"]
        expected[name] = value["metrics"]
        experiment_hashes[str(path)] = digest(path)
    results = {}
    for name, predictions in variants.items():
        if set(predictions) != wanted:
            raise ValueError("Prediction population differs")
        evaluator = BenchmarkEvaluator()
        for token in tokens:
            evaluator.add_frame(token, predictions[token], ground_truth[token])
        results[name] = evaluator.summarize()
        if name in expected and json.dumps(results[name], sort_keys=True) != json.dumps(
            expected[name], sort_keys=True
        ):
            raise ValueError(f"External raw GT report parity failed: {name}")
    write_json(
        args.output,
        {
            "schema": "phase_range_fixed_panel_comparison_v1",
            "frames": len(tokens),
            "profile_sha256": PROFILE_SHA256,
            "population_sha256": cfg.population_sha256,
            "input_shards": inputs,
            "evaluation_sha256": experiment_hashes,
            "external_raw_gt_exact_report_parity": True,
            "metrics": results,
            "source_sha256": {
                str(p.relative_to(ROOT)): digest(p)
                for p in (Path(__file__), ROOT / "src/e_jepa_ttc/detection/benchmark_evaluation.py")
            },
            "test_labels_read": False,
            "new_inference": False,
        },
    )
    print(json.dumps({name: value["mAP"] for name, value in results.items()}))


if __name__ == "__main__":
    main()
