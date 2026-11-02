"""Stream verified candidate shards and check the registered range hypothesis on CPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.benchmark_evaluation import (  # noqa: E402
    CLASSES,
    PROFILE_SHA256,
    BenchmarkEvaluator,
    box_array,
)
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402
from e_jepa_ttc.detection.raw_diagnostics import RawDiagnostics  # noqa: E402


def main() -> None:
    """Read completed shards only; require exact full metric parity before a conclusion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--aligned", required=True, type=Path)
    parser.add_argument("--reference-diagnostics", required=True, type=Path)
    parser.add_argument("--control-reuse", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-seconds", type=float, default=7200)
    args = parser.parse_args()
    if not np.isfinite(args.maximum_seconds) or args.maximum_seconds <= 0:
        raise ValueError("Positive finite observation budget required")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    lock = args.output / "WRITER.lock"
    intake_path = args.evaluation / "INTAKE.json"
    intake = json.loads(intake_path.read_text())
    if (
        intake["schema"] != "phase_range_public_selection_v1"
        or intake["panel_only"]
        or intake["frames"] != 23958
        or intake["profile_sha256"] != PROFILE_SHA256
    ):
        raise ValueError("Only the full aligned public selection is admitted")
    aligned = json.loads((args.aligned / "COMPARISON.json").read_text())
    manifest = json.loads((args.aligned / "MANIFEST.json").read_text())
    for name in ("COMPARISON.json", "ranked.json", "baseline.json"):
        if digest(args.aligned / name) != manifest[name]:
            raise ValueError("Aligned reference changed")
    reference = args.reference_diagnostics
    receipt = json.loads((reference / "COMPLETE.json").read_text())
    if receipt["status"] != "VERIFIED" or receipt["aligned_comparison_sha256"] != digest(
        args.aligned / "COMPARISON.json"
    ):
        raise ValueError("Reference diagnostic provenance differs")
    for name in ("ERROR_DIAGNOSTICS.json", "BY_SEQUENCE.json"):
        if digest(reference / name) != receipt["files"][name]:
            raise ValueError("Reference diagnostics changed")
    reuse = json.loads(args.control_reuse.read_text())
    if (
        reuse["status"] != "PASS"
        or reuse["baseline_results_sha256"] != manifest["baseline.json"]
        or reuse["original_checkpoint_sha256"] != aligned["checkpoint_sha256"]
    ):
        raise ValueError("Control reuse proof differs")
    paths = [
        Path(__file__),
        *[
            ROOT / f"src/e_jepa_ttc/detection/{n}.py"
            for n in (
                "raw_diagnostics",
                "benchmark_evaluation",
                "geometry_contract",
                "range_campaign",
            )
        ],
    ]
    sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    contract = {
        "schema": "phase_range_candidate_diagnostics_v1",
        "sources": sources,
        "evaluation_intake_sha256": digest(intake_path),
        "reference_complete_sha256": digest(reference / "COMPLETE.json"),
        "control_reuse_sha256": digest(args.control_reuse),
        "rule": "Full mAP gain>=0.005 over control and positive over incumbent; "
        "inspect class drops>0.01 and 60-150m recall",
        "range_recall": "Greedy and maximum-cardinality at 0.5/1/2/4m; "
        "max recall ignores score ordering",
        "promotion": False,
        "test_labels_read": False,
    }
    write_json(args.output / "INTAKE.json", contract)
    diagnostics = RawDiagnostics()
    evaluator = BenchmarkEvaluator()
    sources_seen = {}
    seen = set()
    start = time.perf_counter()

    def await_path(path: Path) -> None:
        while not path.is_file():
            if time.perf_counter() - start >= args.maximum_seconds:
                raise TimeoutError("Candidate evidence observation budget exhausted")
            if (args.output / "STOP").exists():
                raise RuntimeError("Diagnostic stop requested")
            time.sleep(10)

    lock.write_text(str(os.getpid()))
    try:
        folders = sorted(p for p in args.historical.iterdir() if (p / "COMPLETE.json").is_file())
        for index, old in enumerate(folders):
            current = args.evaluation / "shards" / f"{index:05d}"
            await_path(current / "COMPLETE.json")
            record = json.loads((current / "COMPLETE.json").read_text())
            if record["intake_sha256"] != contract["evaluation_intake_sha256"]:
                raise ValueError("Candidate shard belongs to a different evaluator")
            for name, expected in record["files"].items():
                if digest(current / name) != expected:
                    raise ValueError("Candidate shard changed")
            for name in ("COMPLETE.json", "tokens.json", "gt.json", "gt_receipt.json"):
                if digest(old / name) != aligned["input_sha256"][old.name][name]:
                    raise ValueError("Historical metadata changed")
            old_receipt = json.loads((old / "gt_receipt.json").read_text())
            if old_receipt["role"] != "selection" or old_receipt["test_labels_read"] is not False:
                raise ValueError("Only public selection labels permitted")
            tokens = json.loads((old / "tokens.json").read_text())["sample_tokens"]
            if tokens != record["tokens"] or seen.intersection(tokens):
                raise ValueError("Shard ordering/population changed")
            gt = json.loads((old / "gt.json").read_text())["ground_truth_by_token"]
            predictions = json.loads((current / "predictions.json").read_text())
            targets = json.loads((current / "targets.json").read_text())
            if set(predictions) != set(tokens) or set(targets) != set(tokens):
                raise ValueError("Candidate prediction/target coverage differs")
            for token in tokens:
                frame = gt[token]
                raw = sorted(frame["eligible"] + frame["ignored"], key=lambda b: b["source_index"])
                if [b["source_index"] for b in raw] != list(range(frame["raw_count"])):
                    raise ValueError("Incomplete raw historical targets")
                if [b["class_name"] for b in raw] != [
                    b["class_name"] for b in targets[token]
                ] or not np.array_equal(
                    box_array(raw, prediction=False), box_array(targets[token], prediction=False)
                ):
                    raise ValueError("Candidate and historical raw GT differ")
                diagnostics.add_frame(token, predictions[token], raw, frame)
                evaluator.add_frame(token, predictions[token], targets[token])
            seen.update(tokens)
            sources_seen[current.name] = digest(current / "COMPLETE.json")
            write_json(
                args.output / "STATUS.json",
                {
                    "status": "DIAGNOSING",
                    "frames": len(seen),
                    "shards": index + 1,
                    "elapsed_seconds": time.perf_counter() - start,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                },
            )
        await_path(args.evaluation / "RESULTS.json")
        result = json.loads((args.evaluation / "RESULTS.json").read_text())
        metrics = evaluator.summarize()
        if (
            len(seen) != intake["frames"]
            or result["intake_sha256"] != digest(intake_path)
            or result["shard_receipt_sha256"] != sources_seen
            or json.loads(json.dumps(metrics)) != result["metrics"]
        ):
            raise ValueError("Full reconstructed candidate metrics differ")
        if sources != {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}:
            raise ValueError("Diagnostic sources changed")
        old_metrics = json.loads((args.aligned / "ranked.json").read_text())
        old_diagnostics = json.loads((reference / "ERROR_DIAGNOSTICS.json").read_text())
        summary = diagnostics.summarize()
        declines = {
            cls: metrics["classes"][cls]["AP"] - old_metrics["classes"][cls]["AP"]
            for cls in CLASSES
            if metrics["classes"][cls]["AP"] - old_metrics["classes"][cls]["AP"] < -0.01
        }
        delta_control = metrics["mAP"] - reuse["full_baseline_mAP"]
        delta_incumbent = metrics["mAP"] - old_metrics["mAP"]
        comparison = {
            "metrics": metrics,
            "incumbent_mAP": old_metrics["mAP"],
            "control_mAP": reuse["full_baseline_mAP"],
            "delta_control": delta_control,
            "delta_incumbent": delta_incumbent,
            "class_declines_over_0_01": declines,
            "macro_gate_pass": delta_control >= 0.005 and delta_incumbent > 0,
            "promotion": False,
            "coverage_requires_review": True,
            "range_recall": {
                key: {
                    "candidate": summary["groups"][key],
                    "reference": old_diagnostics["groups"][key],
                }
                for key in summary["groups"]
                if key.startswith("optical_depth_m/")
            },
            "candidate_by_sequence": result["by_sequence"],
            "incumbent_by_sequence": json.loads((reference / "BY_SEQUENCE.json").read_text()),
            "uncertainty": "Eight exposed sequences; "
            "no independence or statistical significance claim",
        }
        write_json(args.output / "ERROR_DIAGNOSTICS.json", summary)
        write_json(args.output / "COMPARISON.json", comparison)
        write_json(
            args.output / "COMPLETE.json",
            {
                **contract,
                "status": "VERIFIED",
                "frames": len(seen),
                "full_metric_exact_parity": True,
                "input_shards": sources_seen,
                "candidate_results_sha256": digest(args.evaluation / "RESULTS.json"),
                "files": {
                    name: digest(args.output / name)
                    for name in ("ERROR_DIAGNOSTICS.json", "COMPARISON.json")
                },
            },
        )
        write_json(
            args.output / "STATUS.json",
            {
                "status": "COMPLETE",
                "frames": len(seen),
                "mAP": metrics["mAP"],
                "delta_control": delta_control,
                "delta_incumbent": delta_incumbent,
                "elapsed_seconds": time.perf_counter() - start,
            },
        )
    except Exception as error:
        write_json(args.output / "FAILURE.json", {"status": "FAILED", "error": str(error)})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
