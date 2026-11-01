"""Diagnose frozen public selection shards without training or new inference."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_detection_benchmark_local import read, sha, verify_shard_coverage, write  # noqa: E402

from e_jepa_ttc.detection.benchmark_evaluation import (  # noqa: E402
    PROFILE_SHA256,
    BenchmarkEvaluator,
)
from e_jepa_ttc.detection.raw_diagnostics import RawDiagnostics, oracle_variants  # noqa: E402


def main() -> None:
    """Validate evidence, diagnose full raw selection, and write finite JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folders, historical = verify_shard_coverage(args.shards)
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = [
        Path(__file__),
        ROOT / "scripts/evaluate_detection_benchmark_local.py",
        *[
            ROOT / f"src/e_jepa_ttc/detection/{name}.py"
            for name in ("raw_diagnostics", "benchmark_evaluation", "geometry_contract")
        ],
    ]
    sources = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    manifest = read(args.release / "MANIFEST.json")
    for name, digest in manifest["files"].items():
        path = (args.release / name).resolve()
        if not path.is_relative_to(args.release.resolve()) or sha(path) != digest:
            raise ValueError(f"Release integrity failure: {name}")
    aligned = read(args.aligned / "COMPARISON.json")
    if aligned["profile_sha256"] != PROFILE_SHA256:
        raise ValueError("Unexpected evaluation profile")
    intake = {
        "schema": "phase_ttc_diagnostic_intake_v1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": sources,
        "python": platform.python_version(),
        "release_manifest_sha256": sha(args.release / "MANIFEST.json"),
        "release_files_verified": len(manifest["files"]),
        "checkpoint_sha256": manifest["files"][manifest["checkpoint"]],
        "ranker_file_sha256": manifest["files"][manifest["ranker"]],
        "aligned_comparison_sha256": sha(args.aligned / "COMPARISON.json"),
        "profile_sha256": PROFILE_SHA256,
        "private_scorer_equivalence": "NOT_CERTIFIED",
        "test_labels_read": False,
        "training": False,
        "new_inference": False,
        "cloud_compute_started": False,
    }
    write(args.output / "INTAKE_AND_PROTOCOL_RECEIPT.json", intake)
    diagnostics = RawDiagnostics()
    evaluators = {
        name: BenchmarkEvaluator()
        for name in ("ranked", "oracle_class", "oracle_size", "oracle_depth", "oracle_rank")
    }
    sequence_evaluators: dict[str, BenchmarkEvaluator] = {}
    seen: set[str] = set()
    inputs = {}
    start = time.perf_counter()
    for number, folder in enumerate(folders, 1):
        complete = read(folder / "COMPLETE.json")
        if complete["status"] != "VERIFIED":
            raise ValueError("Unverified input shard")
        inputs[folder.name] = {}
        for name in ("COMPLETE.json", "tokens.json", "gt.json", "gt_receipt.json", "learned.json"):
            digest = sha(folder / name)
            if digest != aligned["input_sha256"][folder.name][name]:
                raise ValueError(f"Aligned receipt mismatch: {folder / name}")
            if name != "COMPLETE.json" and digest != complete["files"][name]:
                raise ValueError("Shard receipt mismatch")
            inputs[folder.name][name] = digest
        receipt = read(folder / "gt_receipt.json")
        if (
            receipt["test_labels_read"] is not False
            or receipt["role"] != "selection"
            or receipt["panel"] != "selection_full"
        ):
            raise ValueError("Only public full selection admitted")
        gt = read(folder / "gt.json")["ground_truth_by_token"]
        payload = read(folder / "learned.json")
        if payload["checkpoint_sha256"] != intake["checkpoint_sha256"]:
            raise ValueError("Wrong checkpoint")
        tokens = read(folder / "tokens.json")["sample_tokens"]
        if (
            len(tokens) != len(set(tokens))
            or set(tokens) != set(gt)
            or set(tokens) != set(payload["predictions"])
            or len(tokens) != complete["frames"]
        ):
            raise ValueError("Inconsistent token population")
        for token in tokens:
            if token in seen:
                raise ValueError("Repeated token across shards")
            seen.add(token)
            frame = gt[token]
            targets = sorted(frame["eligible"] + frame["ignored"], key=lambda b: b["source_index"])
            if [b["source_index"] for b in targets] != list(range(frame["raw_count"])):
                raise ValueError("Incomplete raw ground truth")
            predictions = payload["predictions"][token]
            diagnostics.add_frame(token, predictions, targets, frame)
            evaluators["ranked"].add_frame(token, predictions, targets)
            sequence = token.split(":")[0]
            if sequence not in sequence_evaluators:
                sequence_evaluators[sequence] = BenchmarkEvaluator()
            sequence_evaluators[sequence].add_frame(token, predictions, targets)
            for name, rows in oracle_variants(predictions, targets, frame).items():
                evaluators[f"oracle_{name}"].add_frame(token, rows, targets)
        elapsed = time.perf_counter() - start
        status = {
            "status": "diagnosing",
            "shards": number,
            "total_shards": len(folders),
            "frames": len(seen),
            "elapsed_seconds": elapsed,
            "frames_per_second": len(seen) / elapsed,
            "eta_seconds_linear": (historical["frames"] - len(seen)) * elapsed / len(seen),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        write(args.output / "STATUS.json", status)
        print(json.dumps(status), flush=True)
    if len(seen) != historical["frames"]:
        raise ValueError("Incomplete selection")
    results = {name: evaluator.summarize() for name, evaluator in evaluators.items()}
    old = read(args.aligned / "ranked.json")
    if json.loads(json.dumps(results["ranked"])) != old:
        raise ValueError("Reconstructed ranked evaluator output differs from verified result")
    for name, report in results.items():
        write(args.output / f"{name}.json", report)
    write(
        args.output / "BY_SEQUENCE.json",
        {name: evaluator.summarize() for name, evaluator in sequence_evaluators.items()},
    )
    write(args.output / "ERROR_DIAGNOSTICS.json", diagnostics.summarize())
    if sources != {str(p.relative_to(ROOT)): sha(p) for p in source_paths}:
        raise ValueError("Sources changed during execution")
    write(
        args.output / "COMPLETE.json",
        {
            **intake,
            "input_sha256": inputs,
            "frames": len(seen),
            "ranked_full_report_parity": True,
            "elapsed_seconds": time.perf_counter() - start,
            "files": {
                p.name: sha(p) for p in args.output.glob("*.json") if p.name != "STATUS.json"
            },
            "status": "VERIFIED",
        },
    )
    write(
        args.output / "STATUS.json",
        {"status": "VERIFIED", "frames": len(seen), "elapsed_seconds": time.perf_counter() - start},
    )


if __name__ == "__main__":
    main()
