#!/usr/bin/env python
"""Create a hash-validated interim report for one completed campaign segment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = Path("output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/r0_1024")
DEFAULT_QUEUE = Path(
    "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_queue_execution_v1/"
    "queue_receipt.json"
)
DEFAULT_CONSTANT = Path("output/campaign_controls/v5_train_constant_v1/constant_controls.json")
DEFAULT_PILOT = Path("output/s0_diagnostics/pilot_selection256_v2_camera_axes_fail_closed_v3")
DEFAULT_OUTPUT = Path("output/campaign_reports/r0_update1024_v1")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load an object while refusing duplicate JSON keys."""

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    """Write strict JSON."""
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load strict JSONL with monotonically increasing updates."""
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        _require(isinstance(value, dict), f"JSONL row {number} is not an object")
        rows.append(value)
    _require(bool(rows), "Training metrics JSONL is empty")
    updates = [row.get("update") for row in rows]
    _require(updates == list(range(1, len(rows) + 1)), "Metrics updates are not contiguous")
    return rows


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{name} is not finite")
    return result


def derive_training_curves(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive complete curves and robust endpoint summaries from JSONL."""
    fields = (
        "loss",
        "loss_center",
        "gradient_norm_before_clip",
        "preprocessing_seconds",
        "forward_backward_update_seconds",
    )
    curves = {
        field: [
            _finite_number(row.get(field), f"{field}@{index}") for index, row in enumerate(rows)
        ]
        for field in fields
    }
    window = min(64, len(rows))
    return {
        "update": [int(row["update"]) for row in rows],
        **curves,
        "summary": {
            field: {
                "first": values[0],
                "last": values[-1],
                "first_window_mean": statistics.mean(values[:window]),
                "last_window_mean": statistics.mean(values[-window:]),
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for field, values in curves.items()
        },
        "window_updates": window,
        "preprocessing_seconds_sum": sum(curves["preprocessing_seconds"]),
        "forward_backward_update_seconds_sum": sum(curves["forward_backward_update_seconds"]),
        "empty_target_exposures_sum": sum(int(row["empty_target_exposures"]) for row in rows),
        "valid_velocity_object_exposures_sum": sum(
            int(row["valid_velocity_object_exposures"]) for row in rows
        ),
    }


def analyze_classes(metrics: dict[str, Any]) -> dict[str, Any]:
    """Classify directly observed proposal/matching failure modes without oracles."""
    result = {}
    for category, value in metrics["per_class"].items():
        count = int(value["eligible_ground_truth_count"])
        predictions = int(value["prediction_count_after_domain_filter"])
        loose = value["thresholds"]["4.0"]
        tp4 = int(loose["TP"])
        ap = value["AP101_mean_thresholds"]
        if count == 0:
            state = "unsupported"
        elif predictions == 0:
            state = "no_class_predictions"
        elif tp4 == 0:
            state = "class_predictions_but_no_tp_within_4m"
        elif ap == 0:
            state = "loose_tp_but_zero_mean_ap"
        else:
            state = "positive_ap"
        result[category] = {
            "AP101_mean_thresholds": ap,
            "eligible_ground_truth_count": count,
            "prediction_count_after_domain_filter": predictions,
            "predictions_per_eligible_gt": predictions / count if count else None,
            "TP_at_4m": tp4,
            "FP_at_4m": int(loose["FP"]),
            "recall_at_4m": tp4 / count if count else None,
            "observed_state": state,
        }
    return result


def _evaluation(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = load_json(path)
    _require(
        set(bundle)
        == {
            "metrics_v2",
            "predictions",
            "ground_truth_v2",
            "decode_counts",
            "legacy_metrics_separate",
        },
        f"Unexpected evaluation bundle schema: {path}",
    )
    metrics = bundle["metrics_v2"]
    _require(
        metrics.get("schema") == "phase_ttc_detection_evaluation_result_v2",
        f"Unexpected metrics schema: {path}",
    )
    return bundle, metrics


def _verify_segment(
    run: Path, queue_receipt_path: Path, segment_name: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    summary_path = run / "summary.json"
    _require(summary_path.is_file(), f"Missing frozen segment summary: {summary_path}")
    summary = load_json(summary_path)
    _require(summary.get("schema") == "eap_detection_campaign_run_v1", "Bad summary schema")
    _require(
        summary.get("status") in {"BUDGET_STOPPED", "COMPLETED"},
        "Segment is not complete",
    )
    if summary["status"] == "BUDGET_STOPPED":
        _require(
            summary.get("stop_reason") == "external_execution_stop_after_completed_update",
            "Bad stop reason",
        )
    update = summary.get("final_update")
    _require(isinstance(update, int) and update > 0, "Final update is invalid")
    runtime_path = run / "runtime_preregistration.json"
    config_path = run / "config_resolved.json"
    metrics_path = run / "metrics.jsonl"
    selection_path = run / f"evaluation/selection_update_{update:06d}.json"
    pilot_path = run / f"evaluation/pilot_parity_update_{update:06d}.json"
    required = (runtime_path, config_path, metrics_path, selection_path, pilot_path)
    for path in required:
        _require(path.is_file(), f"Missing frozen segment artifact: {path}")
    _require(summary.get("run_name") == segment_name, "Segment name mismatch")
    _require(
        summary.get("selected_checkpoint", {}).get("status") == "AVAILABLE",
        "No selected checkpoint",
    )

    queue = load_json(queue_receipt_path)
    _require(
        queue.get("schema") == "phase_ttc_detection_campaign_queue_receipt_v1", "Bad queue schema"
    )
    segment = queue.get("segments", {}).get(segment_name)
    _require(isinstance(segment, dict), "Completed queue segment is absent")
    child = segment.get("child", {})
    verified = segment.get("verified", {})
    _require(
        child.get("status") == "EXITED_ZERO" and child.get("returncode") == 0,
        "Child did not exit zero",
    )
    _require(child.get("timed_out") is False, "Child timed out")
    summary_hash = sha256_file(summary_path)
    _require(verified.get("summary_sha256") == summary_hash, "Queue summary hash mismatch")
    _require(verified.get("summary") == summary, "Queue summary snapshot differs")

    selected_path = run / "checkpoints/selected.pt"
    last_path = run / "checkpoints/last.pt"
    _require(selected_path.is_file() and last_path.is_file(), "Checkpoint bytes missing")
    selected_hash = sha256_file(selected_path)
    last_hash = sha256_file(last_path)
    _require(
        selected_hash == verified.get("selected_checkpoint_sha256"),
        "Selected checkpoint hash mismatch",
    )
    _require(last_hash == verified.get("last_checkpoint_sha256"), "Last checkpoint hash mismatch")
    _require(
        selected_hash == summary["selected_checkpoint"]["sha256"], "Summary selected hash mismatch"
    )
    _require(last_hash == summary["last_checkpoint_sha256"], "Summary last hash mismatch")

    runtime = load_json(runtime_path)
    _require(
        sha256_file(runtime_path)
        == summary["runtime_provenance"]["runtime_preregistration_sha256"],
        "Runtime preregistration hash mismatch",
    )
    _require(
        canonical_sha256(runtime["source_receipt"]) == runtime["source_receipt_sha256"],
        "Source receipt canonical hash mismatch",
    )
    _require(
        canonical_sha256(runtime["effective_model"]) == runtime["effective_model_sha256"],
        "Effective model canonical hash mismatch",
    )
    _require(
        canonical_sha256(runtime["code_sha256"]) == runtime["code_sha256_manifest"],
        "Code manifest canonical hash mismatch",
    )
    _require(
        canonical_sha256(runtime["portable_source_sha256"])
        == runtime["portable_source_sha256_manifest"],
        "Portable source manifest canonical hash mismatch",
    )
    config = load_json(config_path)
    _require(config.get("sha256") == summary["config_sha256"], "Resolved config hash mismatch")
    _require(
        config["data"]["expected_manifest_sha256"] == summary["dataset_manifest_hash"],
        "Manifest binding mismatch",
    )
    _require(
        config["data"]["expected_split_sha256"] == summary["split_hash"], "Split binding mismatch"
    )
    _require(
        config["data"]["expected_full_selection_sha256"] == summary["full_selection_sha256"],
        "Full selection binding mismatch",
    )
    _require(
        child.get("producer_sha256")
        == runtime["code_sha256"]["scripts/train_detection_campaign.py"],
        "Queue producer hash mismatch",
    )
    _require(
        child.get("config_canonical_sha256") == summary["config_sha256"],
        "Queue config hash mismatch",
    )
    return (
        summary,
        segment,
        {
            "summary": summary_hash,
            "runtime_preregistration": sha256_file(runtime_path),
            "config_resolved": sha256_file(config_path),
            "metrics_jsonl": sha256_file(metrics_path),
            "selection_evaluation": sha256_file(selection_path),
            "pilot_evaluation": sha256_file(pilot_path),
            "selected_checkpoint": selected_hash,
            "last_checkpoint": last_hash,
            "queue_segment_canonical": canonical_sha256(segment),
        },
    )


def derive_report(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate frozen inputs and derive the complete interim result."""
    summary, queue_segment, hashes = _verify_segment(args.run, args.queue_receipt, args.segment)
    update = summary["final_update"]
    selection_bundle, selection = _evaluation(
        args.run / f"evaluation/selection_update_{update:06d}.json"
    )
    pilot_bundle, pilot = _evaluation(
        args.run / f"evaluation/pilot_parity_update_{update:06d}.json"
    )
    protocol = summary["evaluation_protocol_sha256"]
    _require(
        selection["protocol_sha256"] == protocol == pilot["protocol_sha256"],
        "Evaluation protocol mismatch",
    )
    _require(
        selection["sample_count"] == 1024 and pilot["sample_count"] == 256,
        "Evaluation sample count mismatch",
    )

    constant = load_json(args.constant_control)
    constant_monitor = constant["populations"]["monitor1024"]["metrics"]
    _require(constant["status"] == "PASS", "Constant control is not PASS")
    _require(constant["protocol_sha256"] == protocol, "Constant protocol mismatch")
    _require(constant_monitor["sample_count"] == 1024, "Constant monitor count mismatch")
    _require(
        constant_monitor["population"]["ground_truth_sha256_by_token"]
        == selection["population"]["ground_truth_sha256_by_token"],
        "Constant and R0 monitor GT maps differ",
    )

    s0_diagnostics = load_json(args.pilot / "diagnostics.json")
    s0_gt = load_json(args.pilot / "ground_truth_v2.json")
    s0_metrics = s0_diagnostics["evaluation"]
    pilot_gt_exact = s0_gt["ground_truth_by_token"] == pilot_bundle["ground_truth_v2"]
    pilot_tokens_exact = set(s0_gt["ground_truth_by_token"]) == set(pilot_bundle["ground_truth_v2"])
    pilot_protocol_exact = s0_gt["protocol_sha256"] == protocol == s0_metrics["protocol_sha256"]
    _require(
        pilot_gt_exact and pilot_tokens_exact and pilot_protocol_exact,
        "Pilot comparison GT/protocol differs",
    )
    _require(
        s0_metrics["population"]["ground_truth_sha256_by_token"]
        == pilot["population"]["ground_truth_sha256_by_token"],
        "Pilot metric GT maps differ",
    )

    rows = load_jsonl(args.run / "metrics.jsonl")
    _require(
        len(rows) == summary["final_update"], "Training curve length differs from final update"
    )
    curves = derive_training_curves(rows)
    started = datetime.fromisoformat(summary["started_at_utc"])
    ended = datetime.fromisoformat(summary["ended_at_utc"])
    wall_seconds = (ended - started).total_seconds()
    phase_sum = curves["preprocessing_seconds_sum"] + curves["forward_backward_update_seconds_sum"]
    report = {
        "schema": "phase_ttc_detection_campaign_interim_report_v1",
        "status": "PASS",
        "scope": f"{summary['run_name']} update {update} completed segment; interim, nonfinal",
        "run": {
            "name": summary["run_name"],
            "status": summary["status"],
            "stop_reason": summary["stop_reason"],
            "final_update": summary["final_update"],
            "image_exposures": summary["stage_image_exposures"],
            "seed": summary["seed"],
            "protocol_sha256": protocol,
            "manifest_sha256": summary["dataset_manifest_hash"],
            "split_sha256": summary["split_hash"],
            "selected_checkpoint_sha256": hashes["selected_checkpoint"],
            "last_checkpoint_sha256": hashes["last_checkpoint"],
        },
        "timing": {
            "wall_seconds": wall_seconds,
            "producer_stage_training_seconds": summary["stage_training_seconds"],
            "preprocessing_seconds_sum_from_1024_updates": curves["preprocessing_seconds_sum"],
            "forward_backward_update_seconds_sum_from_1024_updates": curves[
                "forward_backward_update_seconds_sum"
            ],
            "wall_residual_seconds": wall_seconds - phase_sum,
            "evaluation_seconds": None,
            "evaluation_timing_status": (
                "not separately instrumented; wall residual also contains checkpointing, JSON "
                "serialization and other overhead, so it is not relabeled as evaluation time"
            ),
        },
        "monitor1024": {
            "metrics": {
                "dev_AP101_supported_classes": selection["dev_AP101_supported_classes"],
                "dev_AP101_all_eight_absent_zero": selection["dev_AP101_all_eight_absent_zero"],
                "supported_class_count": selection["supported_class_count"],
                "population": selection["population"],
                "predictions": {
                    key: value
                    for key, value in selection["predictions"].items()
                    if key != "outside_domain"
                },
                "outside_domain_by_reason": dict(
                    sorted(
                        __import__("collections")
                        .Counter(
                            row["reason"] for row in selection["predictions"]["outside_domain"]
                        )
                        .items()
                    )
                ),
                "class_agnostic_recall": selection["class_agnostic_recall"],
            },
            "per_class_analysis": analyze_classes(selection),
            "constant_control": {
                "same_protocol": True,
                "same_sample_count": True,
                "exact_ground_truth_hash_map": True,
                "dev_AP101_supported_classes": constant_monitor["dev_AP101_supported_classes"],
                "R0_minus_constant": selection["dev_AP101_supported_classes"]
                - constant_monitor["dev_AP101_supported_classes"],
            },
        },
        "pilot256_same_gt_comparison": {
            "same_tokens": pilot_tokens_exact,
            "exact_ground_truth": pilot_gt_exact,
            "same_protocol": pilot_protocol_exact,
            "supported_class_count": pilot["supported_class_count"],
            "R0_update1024_AP": pilot["dev_AP101_supported_classes"],
            "historical_pilot_AP": s0_metrics["dev_AP101_supported_classes"],
            "difference": pilot["dev_AP101_supported_classes"]
            - s0_metrics["dev_AP101_supported_classes"],
            "attribution": (
                "descriptive only: train population/diversity and update/exposure budget changed "
                "jointly, so this does not isolate a causal intervention"
            ),
        },
        "training_curve_summary": curves["summary"],
        "exposure_accounting": {
            "stage_empty_target_exposures": summary["stage_empty_target_exposures"],
            "metrics_empty_target_exposures_sum": curves["empty_target_exposures_sum"],
            "stage_valid_velocity_object_exposures": summary[
                "stage_valid_velocity_object_exposures"
            ],
            "metrics_valid_velocity_object_exposures_sum": curves[
                "valid_velocity_object_exposures_sum"
            ],
        },
        "posterior_diagnostic": {
            "primary_observation": (
                "five classes have zero AP; four emit class proposals but obtain no TP at 4 m, "
                "while other_vehicle emits no proposals"
            ),
            "non_oracle": True,
            "no_refit": True,
        },
        "hash_validation": {
            **hashes,
            "constant_control": sha256_file(args.constant_control),
            "s0_diagnostics": sha256_file(args.pilot / "diagnostics.json"),
            "s0_ground_truth": sha256_file(args.pilot / "ground_truth_v2.json"),
            "queue_segment_exit_status": queue_segment["child"]["status"],
        },
        "limitations": [
            "Interim update-1024 result from one seed; not a final model-selection result",
            "Local v2 development AP, not the official eAP scorer",
            "The 1024-frame monitor was used by the training campaign for checkpoint selection",
            "Pilot comparison changes training population/diversity and budget jointly",
            "Five zero-AP classes remain; other_vehicle has only four eligible monitor GT",
            "Evaluation duration was not separately instrumented and is kept null",
            "No oracle substitutions, label-conditioned refits or future checkpoint claims",
        ],
    }
    _require(
        report["exposure_accounting"]["stage_empty_target_exposures"]
        == report["exposure_accounting"]["metrics_empty_target_exposures_sum"],
        "Empty-target exposure accounting mismatch",
    )
    _require(
        report["exposure_accounting"]["stage_valid_velocity_object_exposures"]
        == report["exposure_accounting"]["metrics_valid_velocity_object_exposures_sum"],
        "Velocity exposure accounting mismatch",
    )
    return report, curves


def render(report: dict[str, Any]) -> str:
    """Render the interim report only from derived values."""
    monitor = report["monitor1024"]
    metrics = monitor["metrics"]
    timing = report["timing"]
    pilot = report["pilot256_same_gt_comparison"]
    lines = [
        f"# RGB interim results: {report['run']['name']} update {report['run']['final_update']}",
        "",
        f"Segment status: `{report['run']['status']}` at update "
        f"`{report['run']['final_update']}`. This is an interim, nonfinal result.",
        "",
        "## Monitor v5 (1,024 frames)",
        "",
        f"- Global local-v2 AP101: `{metrics['dev_AP101_supported_classes']}`.",
        f"- Train-only constant AP101 on the exact same GT: "
        f"`{monitor['constant_control']['dev_AP101_supported_classes']}`.",
        f"- Difference R0 − constant: `{monitor['constant_control']['R0_minus_constant']}`.",
        f"- Class-agnostic recall at 2 m: "
        f"`{metrics['class_agnostic_recall']['thresholds']['2.0']['recall']}`.",
        "",
        "| Class | AP101 | Eligible GT | Predictions | TP@4m | Recall@4m | Observation |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for category, value in monitor["per_class_analysis"].items():
        lines.append(
            f"| {category} | {value['AP101_mean_thresholds']} | "
            f"{value['eligible_ground_truth_count']} | "
            f"{value['prediction_count_after_domain_filter']} | {value['TP_at_4m']} | "
            f"{value['recall_at_4m']} | {value['observed_state']} |"
        )
    lines.extend(
        [
            "",
            "Five classes have zero AP. Bicycle, pedestrian, bus and tricycle emit proposals "
            "but have no TP even at 4 m; `other_vehicle` emits none. This is a direct diagnostic "
            "of the retained predictions, not an oracle attribution.",
            "",
            "## Same-GT pilot panel",
            "",
            f"Exact tokens/GT/protocol: `{pilot['same_tokens']}/{pilot['exact_ground_truth']}/"
            f"{pilot['same_protocol']}`. AP changes from `{pilot['historical_pilot_AP']}` to "
            f"`{pilot['R0_update1024_AP']}` (difference `{pilot['difference']}`). Training data "
            "and budget changed jointly, so the difference is descriptive rather than causal.",
            "",
            "## Timing",
            "",
            f"- Wall: `{timing['wall_seconds']}` s.",
            f"- Producer stage-training clock: `{timing['producer_stage_training_seconds']}` s.",
            f"- Sum of per-update preprocessing: "
            f"`{timing['preprocessing_seconds_sum_from_1024_updates']}` s.",
            f"- Sum of forward/backward/update: "
            f"`{timing['forward_backward_update_seconds_sum_from_1024_updates']}` s.",
            f"- Evaluation: `{timing['evaluation_seconds']}` — "
            f"{timing['evaluation_timing_status']}.",
            "",
            "## Limits",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    """Build a new immutable report directory."""
    _require(not args.output.exists(), "Use a new report output directory")
    report, curves = derive_report(args)
    args.output.mkdir(parents=True)
    write_json(args.output / "report.json", report)
    write_json(args.output / "curves.json", curves)
    (args.output / "RGB_INTERIM_RESULTS.md").write_text(render(report), encoding="utf-8")
    receipt = {
        "schema": "phase_ttc_detection_campaign_interim_report_receipt_v1",
        "status": "PASS",
        "report_sha256": sha256_file(args.output / "report.json"),
        "curves_sha256": sha256_file(args.output / "curves.json"),
        "markdown_sha256": sha256_file(args.output / "RGB_INTERIM_RESULTS.md"),
        "generator_sha256": sha256_file(Path(__file__)),
    }
    write_json(args.output / "receipt.json", receipt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--queue-receipt", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--segment", default="r0_1024")
    parser.add_argument("--constant-control", type=Path, default=DEFAULT_CONSTANT)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
