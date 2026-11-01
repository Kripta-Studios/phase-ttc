"""Reuse fixed-panel GT and saved predictions for CPU-only error decomposition."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.benchmark_evaluation import PROFILE_SHA256, BenchmarkEvaluator
from e_jepa_ttc.detection.monocop_campaign import PANEL_SHA256
from e_jepa_ttc.detection.range_campaign import digest, write_json
from e_jepa_ttc.detection.raw_diagnostics import RawDiagnostics, oracle_variants


def main() -> None:
    """Verify external raw-GT parity, then measure counterfactuals without training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--reference-panel", type=Path, required=True)
    parser.add_argument("--historical-shards", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("CPU-only diagnostics require CUDA_VISIBLE_DEVICES=-1")
    torch.set_num_threads(1)
    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    start = time.perf_counter()
    if args.output.exists():
        raise ValueError("New diagnostic output required")
    population = json.loads(args.population.read_text())
    reference = json.loads(args.reference_panel.read_text())
    if (
        digest(args.population) != PANEL_SHA256
        or reference["population_sha256"] != PANEL_SHA256
        or reference["profile_sha256"] != PROFILE_SHA256
        or not reference["external_raw_gt_exact_report_parity"]
        or reference["test_labels_read"]
    ):
        raise ValueError("Fixed external panel reference changed")
    tokens = population["roles"]["selection"]
    wanted = set(tokens)
    if len(tokens) != 512 or len(wanted) != 512:
        raise ValueError("Expected the complete fixed panel")
    if args.cache.is_dir():
        receipt = json.loads((args.cache / "RECEIPT.json").read_text())
        if (
            receipt["reference_panel_sha256"] != digest(args.reference_panel)
            or receipt["population_sha256"] != PANEL_SHA256
            or receipt["cache_sha256"] != digest(args.cache / "CACHE.json")
        ):
            raise ValueError("Reusable GT cache changed")
        cached = json.loads((args.cache / "CACHE.json").read_text())
    else:
        frames, ranked, source_hashes = {}, {}, {}
        for folder_name, expected in reference["input_shards"].items():
            folder = args.historical_shards / folder_name
            for name in ("tokens.json", "gt.json", "gt_receipt.json", "learned.json"):
                path = folder / name
                if digest(path) != expected[name]:
                    raise ValueError(f"External raw-GT/prediction shard changed: {path}")
                source_hashes[path.as_posix()] = expected[name]
            gt_receipt = json.loads((folder / "gt_receipt.json").read_text())
            if gt_receipt["role"] != "selection" or gt_receipt["test_labels_read"]:
                raise ValueError("Non-public-selection labels refused")
            gt = json.loads((folder / "gt.json").read_text())["ground_truth_by_token"]
            pred = json.loads((folder / "learned.json").read_text())["predictions"]
            for token in wanted.intersection(gt):
                frame = gt[token]
                targets = sorted(
                    frame["eligible"] + frame["ignored"], key=lambda r: r["source_index"]
                )
                if token in frames or [r["source_index"] for r in targets] != list(
                    range(frame["raw_count"])
                ):
                    raise ValueError("Duplicate or incomplete external raw GT")
                frames[token] = {
                    "targets": targets,
                    **{k: frame[k] for k in ("intrinsic", "event_from_ego", "image_wh")},
                }
                ranked[token] = pred[token]
        if set(frames) != wanted or set(ranked) != wanted:
            raise ValueError("GT cache does not cover the full panel")
        cached = {"frames": frames, "ranked": ranked}
        args.cache.mkdir(parents=True)
        write_json(args.cache / "CACHE.json", cached)
        write_json(
            args.cache / "RECEIPT.json",
            {
                "reference_panel_sha256": digest(args.reference_panel),
                "population_sha256": PANEL_SHA256,
                "cache_sha256": digest(args.cache / "CACHE.json"),
                "source_hashes": source_hashes,
                "raw_all": True,
                "test_labels_read": False,
            },
        )
    original = json.loads(args.evaluation.read_text())
    if (
        set(original["predictions"]) != wanted
        or original["gt_population"] != "raw_all"
        or not original["label_free_forward"]
        or original["metrics"]["profile_sha256"] != PROFILE_SHA256
    ):
        raise ValueError("Candidate evaluation uses a different protocol/population")
    args.output.mkdir(parents=True)
    evaluators = {
        name: BenchmarkEvaluator()
        for name in (
            "candidate",
            "incumbent",
            "oracle_class",
            "oracle_depth",
            "oracle_rank",
            "oracle_size",
        )
    }
    diagnostics = {"candidate": RawDiagnostics(), "incumbent": RawDiagnostics()}
    for index, token in enumerate(tokens):
        if time.perf_counter() - start > 300:
            raise TimeoutError("CPU diagnostic exceeded its five-minute bound")
        frame = cached["frames"][token]
        target = frame["targets"]
        candidate, incumbent = original["predictions"][token], cached["ranked"][token]
        for name, pred in (("candidate", candidate), ("incumbent", incumbent)):
            evaluators[name].add_frame(token, pred, target)
            diagnostics[name].add_frame(token, pred, target, frame)
        for name, pred in oracle_variants(candidate, target, frame).items():
            evaluators[f"oracle_{name}"].add_frame(token, pred, target)
        if (index + 1) % 64 == 0:
            write_json(
                args.output / "STATUS.json",
                {
                    "status": "CPU_DIAGNOSING",
                    "frames": index + 1,
                    "elapsed_seconds": time.perf_counter() - start,
                },
            )
    results = {name: value.summarize() for name, value in evaluators.items()}
    for name, expected in (
        ("candidate", original["metrics"]),
        ("incumbent", reference["metrics"]["frozen_ranked"]),
    ):
        if json.loads(json.dumps(results[name])) != expected:
            raise ValueError(f"Independent raw-GT evaluator parity failed: {name}")
    for name, value in results.items():
        write_json(args.output / f"{name}.json", value)
    for name, value in diagnostics.items():
        write_json(args.output / f"{name}_diagnostics.json", value.summarize())
    sources = [
        Path(__file__),
        *[
            ROOT / f"src/e_jepa_ttc/detection/{name}.py"
            for name in ("raw_diagnostics", "benchmark_evaluation", "geometry_contract")
        ],
    ]
    receipt = {
        "schema": "phase_fixed_screen_diagnostics_v1",
        "status": "VERIFIED",
        "frames": len(tokens),
        "evaluation": args.evaluation.as_posix(),
        "evaluation_sha256": digest(args.evaluation),
        "cache_receipt_sha256": digest(args.cache / "RECEIPT.json"),
        "raw_gt_external_report_parity": True,
        "new_inference": False,
        "training": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "elapsed_seconds": time.perf_counter() - start,
        "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
        "files": {p.name: digest(p) for p in args.output.glob("*.json") if p.name != "STATUS.json"},
        "oracle_warning": "GT substitutions are diagnostics, "
        "not learned results or general upper bounds.",
    }
    write_json(args.output / "COMPLETE.json", receipt)
    write_json(
        args.output / "STATUS.json",
        {
            "status": "COMPLETE",
            "frames": len(tokens),
            "elapsed_seconds": receipt["elapsed_seconds"],
        },
    )
    print(
        json.dumps(
            {
                "status": "VERIFIED",
                "mAP": {k: v["mAP"] for k, v in results.items()},
                "seconds": receipt["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
