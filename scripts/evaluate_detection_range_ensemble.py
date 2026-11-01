"""Evaluate the preregistered, fixed 60 m combination of two frozen detectors on CPU."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.benchmark_evaluation import (  # noqa: E402
    CLASSES,
    PROFILE_SHA256,
    BenchmarkEvaluator,
    box_array,
)
from e_jepa_ttc.detection.range_ensemble import BOUNDARY_M, fuse_ranges  # noqa: E402


def digest(path: Path) -> str:
    """Hash an immutable source without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path, expected: str | None = None) -> Any:
    """Parse the same bytes whose identity was checked."""
    raw = path.read_bytes()
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Evidence changed: {path}")
    return json.loads(raw)


def write_json(path: Path, value: object) -> None:
    """Publish finite JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    """Bind inputs and recipe before scoring; fail closed on either control mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--aligned", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Run with CUDA_VISIBLE_DEVICES=-1; this is a CPU-only replay")
    inputs: dict[str, str] = {}

    def bind(path: Path) -> Any:
        expected = digest(path)
        value = read_json(path, expected)
        inputs[path.as_posix()] = expected
        return value

    manifest = bind(args.aligned / "MANIFEST.json")
    aligned = read_json(args.aligned / "COMPARISON.json", manifest["COMPARISON.json"])
    baseline = read_json(args.aligned / "baseline.json", manifest["baseline.json"])
    incumbent = read_json(args.aligned / "ranked.json", manifest["ranked.json"])
    status = bind(args.candidate / "STATUS.json")
    far_result = read_json(args.candidate / "RESULTS.json", status["results_sha256"])
    intake = read_json(args.candidate / "INTAKE.json", far_result["intake_sha256"])
    if (
        status["status"] != "COMPLETE"
        or status["frames"] != 23958
        or intake["frames"] != 23958
        or intake["panel_only"]
        or aligned["samples"] != 23958
        or aligned["profile_sha256"] != PROFILE_SHA256
        or intake["profile_sha256"] != PROFILE_SHA256
    ):
        raise ValueError("Both inputs must cover the same full aligned public selection")
    files = [
        Path(__file__),
        ROOT / "docs/detection_sota_20261029/RANGE_ENSEMBLE_PLAN_20261101.md",
        *[
            ROOT / f"src/e_jepa_ttc/detection/{name}.py"
            for name in ("range_ensemble", "benchmark_evaluation", "geometry_contract")
        ],
    ]
    sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    plan = {
        "schema": "phase_fixed_range_ensemble_v1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sources": sources,
        "input_manifests": inputs,
        "candidate_result_sha256": status["results_sha256"],
        "profile_sha256": PROFILE_SHA256,
        "frames": 23958,
        "boundary_optical_depth_m": BOUNDARY_M,
        "routing": "Predictions only: original Z<=60, range150 Z>60; preserve scores and boxes",
        "primary": "gated_60m",
        "controls": ["original_unranked", "range150_unranked", "ungated_union"],
        "maximum_proposals_per_frame": 100,
        "minimum_gain_over_incumbent_and_best_single": 0.005,
        "must_beat_ungated_union": True,
        "class_decline_review_threshold": 0.01,
        "maximum_seconds": 900,
        "exposed_selection": True,
        "gpu_inference": False,
        "test_labels_read": False,
        "new_cloud_spend": False,
        "promotion": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "PLAN.json", plan)
    for relative in sources:
        destination = args.output / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        if digest(destination) != sources[relative]:
            raise ValueError("Executed source snapshot differs")
    lock = args.output / "WRITER.lock"
    lock.write_text(str(os.getpid()))
    evaluators = {
        name: BenchmarkEvaluator()
        for name in ("original_unranked", "range150_unranked", "gated_60m", "ungated_union")
    }
    sequence_evaluators: dict[str, dict[str, BenchmarkEvaluator]] = {
        name: {} for name in ("gated_60m", "ungated_union")
    }
    totals: Counter = Counter()
    seen: set[str] = set()
    shard_hashes = {}
    started = time.perf_counter()
    try:
        folders = sorted(aligned["input_sha256"])
        if len(folders) != 47:
            raise ValueError("Expected the retained 47-shard population")
        for index, folder in enumerate(folders):
            if time.perf_counter() - started > plan["maximum_seconds"]:
                raise TimeoutError("CPU replay budget exhausted; preserve partial evidence")
            if (args.output / "STOP").exists():
                raise RuntimeError("CPU replay stop requested")
            old = args.historical / folder
            old_hashes = aligned["input_sha256"][folder]
            read_json(old / "COMPLETE.json", old_hashes["COMPLETE.json"])
            tokens = read_json(old / "tokens.json", old_hashes["tokens.json"])["sample_tokens"]
            old_receipt = read_json(old / "gt_receipt.json", old_hashes["gt_receipt.json"])
            if old_receipt["role"] != "selection" or old_receipt["test_labels_read"]:
                raise ValueError("Only the retained public selection is allowed")
            near = read_json(old / "predictions.json", old_hashes["predictions.json"])
            metadata = read_json(old / "gt.json", old_hashes["gt.json"])["ground_truth_by_token"]
            current = args.candidate / "shards" / f"{index:05d}"
            receipt = read_json(
                current / "COMPLETE.json", far_result["shard_receipt_sha256"][current.name]
            )
            if (
                receipt["intake_sha256"] != far_result["intake_sha256"]
                or receipt["tokens"] != tokens
                or seen.intersection(tokens)
                or near["checkpoint_sha256"] != aligned["checkpoint_sha256"]
                or near["targets_argument"] is not None
                or near["annotations_returned"] != 0
            ):
                raise ValueError("Checkpoint, ordering or annotation-free input differs")
            far = read_json(current / "predictions.json", receipt["files"]["predictions.json"])
            targets = read_json(current / "targets.json", receipt["files"]["targets.json"])
            if any(set(values) != set(tokens) for values in (near["predictions"], far, targets)):
                raise ValueError("Prediction/target token coverage differs")
            primary_predictions = {}
            union_fingerprint = hashlib.sha256()
            for token in tokens:
                first, second = near["predictions"][token], far[token]
                # The inference function receives only predictions and sensor calibration.
                gated = fuse_ranges(first, second, metadata[token]["event_from_ego"])
                union = [*first, *second]
                primary_predictions[token] = gated
                union_fingerprint.update(
                    json.dumps([token, union], separators=(",", ":"), allow_nan=False).encode()
                    + b"\n"
                )
                frame = metadata[token]
                raw = sorted(frame["eligible"] + frame["ignored"], key=lambda b: b["source_index"])
                if (
                    [b["source_index"] for b in raw] != list(range(frame["raw_count"]))
                    or [b["class_name"] for b in raw] != [b["class_name"] for b in targets[token]]
                    or not np.array_equal(
                        box_array(raw, prediction=False),
                        box_array(targets[token], prediction=False),
                    )
                ):
                    raise ValueError("Raw GT differs between controls")
                variants = {
                    "original_unranked": first,
                    "range150_unranked": second,
                    "gated_60m": gated,
                    "ungated_union": union,
                }
                for name, predictions in variants.items():
                    evaluators[name].add_frame(token, predictions, targets[token])
                    totals[name] += len(predictions)
                    if name in sequence_evaluators:
                        sequence = token.split(":")[0]
                        per_sequence = sequence_evaluators[name].setdefault(
                            sequence, BenchmarkEvaluator()
                        )
                        per_sequence.add_frame(token, predictions, targets[token])
            shard = args.output / "shards" / f"{index:05d}"
            shard.mkdir(parents=True)
            encoded = json.dumps(
                primary_predictions, separators=(",", ":"), allow_nan=False
            ).encode()
            (shard / "gated.json.gz").write_bytes(gzip.compress(encoded, compresslevel=1, mtime=0))
            write_json(
                shard / "COMPLETE.json",
                {
                    "frames": len(tokens),
                    "plan_sha256": digest(args.output / "PLAN.json"),
                    "near_receipt_sha256": old_hashes["COMPLETE.json"],
                    "far_receipt_sha256": far_result["shard_receipt_sha256"][current.name],
                    "gated_json_sha256": hashlib.sha256(encoded).hexdigest(),
                    "gated_gzip_sha256": digest(shard / "gated.json.gz"),
                    "ungated_token_jsonl_sha256": union_fingerprint.hexdigest(),
                },
            )
            shard_hashes[shard.name] = digest(shard / "COMPLETE.json")
            seen.update(tokens)
            write_json(
                args.output / "STATUS.json",
                {
                    "status": "SCORING",
                    "frames": len(seen),
                    "shards": index + 1,
                    "elapsed_seconds": time.perf_counter() - started,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                },
            )
        metrics = {name: json.loads(json.dumps(v.summarize())) for name, v in evaluators.items()}
        if (
            len(seen) != 23958
            or metrics["original_unranked"] != baseline
            or metrics["range150_unranked"] != far_result["metrics"]
        ):
            raise ValueError("Full control metrics failed exact replay parity")
        if sources != {p.relative_to(ROOT).as_posix(): digest(p) for p in files}:
            raise ValueError("Producer sources changed during replay")
        for name, expected in inputs.items():
            if digest(Path(name)) != expected:
                raise ValueError("Input manifest changed during replay")
        if digest(args.candidate / "RESULTS.json") != status["results_sha256"]:
            raise ValueError("Candidate results changed during replay")
        primary = metrics["gated_60m"]
        gain = primary["mAP"] - incumbent["mAP"]
        control_gain = primary["mAP"] - max(baseline["mAP"], far_result["metrics"]["mAP"])
        gate_pass = (
            gain >= 0.005
            and control_gain >= 0.005
            and primary["mAP"] > metrics["ungated_union"]["mAP"]
        )
        result = {
            "plan_sha256": digest(args.output / "PLAN.json"),
            "frames": len(seen),
            "metrics": metrics,
            "incumbent": incumbent,
            "delta_incumbent": gain,
            "delta_best_single": control_gain,
            "macro_gate_pass": gate_pass,
            "class_declines_over_0_01": {
                cls: primary["classes"][cls]["AP"] - incumbent["classes"][cls]["AP"]
                for cls in CLASSES
                if primary["classes"][cls]["AP"] - incumbent["classes"][cls]["AP"] < -0.01
            },
            "prediction_counts": dict(totals),
            "by_sequence": {
                name: {sequence: value.summarize() for sequence, value in group.items()}
                for name, group in sequence_evaluators.items()
            },
            "shards": shard_hashes,
            "control_metric_exact_parity": True,
            "promotion": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(args.output / "RESULTS.json", result)
        lines = [
            "# Combinación fija por distancia predicha",
            "",
            "23.958 imágenes de selección pública ya expuesta; GT raw, ocho clases. "
            "Sin inferencia GPU nueva. Ambos controles individuales reproducen exactamente "
            "sus métricas completas. Ver PLAN.json y el plan de investigación para los criterios.",
            "",
            "| Variante | mAP | Cajas totales |",
            "|---|---:|---:|",
        ]
        for name, value in metrics.items():
            lines.append(f"| {name} | {value['mAP']:.8f} | {totals[name]} |")
        lines += [
            f"| Incumbente con ranking | {incumbent['mAP']:.8f} | {totals['original_unranked']} |",
            "",
            f"Delta primario frente al incumbente: {gain:+.8f}; "
            f"frente al mejor control individual: {control_gain:+.8f}. "
            f"Criterio macro completo: **{gate_pass}**. Promoción: **False**.",
            "",
            "| Clase | Incumbente AP | Gated AP | Unión AP |",
            "|---|---:|---:|---:|",
        ]
        for cls in CLASSES:
            lines.append(
                f"| {cls} | {incumbent['classes'][cls]['AP']:.8f} | "
                f"{primary['classes'][cls]['AP']:.8f} | "
                f"{metrics['ungated_union']['classes'][cls]['AP']:.8f} |"
            )
        lines += [
            "",
            "La combinación conserva el score de cada detector sin recalibrarlo. "
            "Permite hasta 100 propuestas y requiere dos forwards en despliegue. "
            "No se midió esa latencia integrada. Pueden persistir duplicados cerca de la "
            "frontera. No se ajusta la frontera tras ver estos resultados. "
            "No hay prueba de generalización independiente ni nuevo score oficial.",
            "",
        ]
        (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        write_json(
            args.output / "COMPLETE.json",
            {
                "status": "VERIFIED",
                "frames": len(seen),
                "files": {n: digest(args.output / n) for n in ("RESULTS.json", "REPORT.md")},
                "plan_sha256": result["plan_sha256"],
            },
        )
        write_json(
            args.output / "STATUS.json",
            {"status": "COMPLETE", "mAP": primary["mAP"], "macro_gate_pass": gate_pass},
        )
    except Exception as error:
        write_json(args.output / "FAILURE.json", {"status": "FAILED", "error": str(error)})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
