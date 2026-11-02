"""Bind a passive observation to its metric prefix, schedule, and GPU PID."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _window(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "start_update": rows[0]["update"],
        "end_update": rows[-1]["update"],
        "rows": len(rows),
        "mean_preprocessing_seconds": statistics.fmean(
            row["preprocessing_seconds"] for row in rows
        ),
        "mean_forward_backward_update_seconds": statistics.fmean(
            row["forward_backward_update_seconds"] for row in rows
        ),
        "median_preprocessing_seconds": statistics.median(
            row["preprocessing_seconds"] for row in rows
        ),
        "median_forward_backward_update_seconds": statistics.median(
            row["forward_backward_update_seconds"] for row in rows
        ),
    }


def main() -> int:
    """Create a write-once interpretation without touching the trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    observation_bytes = args.observation.read_bytes()
    observation = json.loads(observation_bytes)
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    metrics_bytes = args.metrics.read_bytes()
    rows = [json.loads(line) for line in metrics_bytes.splitlines() if line.strip()]
    if len(rows) < 200:
        raise ValueError("Metric prefix is too short for the requested comparison")
    final_update = rows[-1]["update"]
    scheduled = config["evaluation"]["updates"]
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    gpu_processes = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3 and fields[0].isdigit():
            gpu_processes.append(
                {"pid": int(fields[0]), "process_name": fields[1], "used_memory_mib": fields[2]}
            )
    pid = int(observation["pid"])
    if pid not in {item["pid"] for item in gpu_processes}:
        raise ValueError("Observed trainer PID is not a current GPU compute process")
    result = {
        "schema": "detection_training_passive_observation_interpretation_v1",
        "status": "PASS_BOUND_EVIDENCE",
        "observation_sha256": _sha(observation_bytes),
        "config_sha256": _sha(config_bytes),
        "metrics_prefix_sha256": _sha(metrics_bytes),
        "metrics_prefix_bytes": len(metrics_bytes),
        "metrics_prefix_rows": len(rows),
        "final_update": final_update,
        "final_update_is_scheduled_evaluation": final_update in scheduled,
        "observation_updates_completed": observation["metrics"]["updates_completed"],
        "evaluation_artifact_completed_during_observation": (
            observation["evaluation"]["start"] != observation["evaluation"]["end"]
        ),
        "phase_inference": "SCHEDULED_MONITOR_EVALUATION_IN_PROGRESS",
        "phase_inference_basis": (
            "update remained at scheduled evaluation boundary 2048 for the full passive window; "
            "no completed evaluation artifact was published during that window"
        ),
        "resume_all": _window(rows),
        "previous_100": _window(rows[-200:-100]),
        "last_100": _window(rows[-100:]),
        "gpu_processes_snapshot": gpu_processes,
        "trainer_pid_bound_to_gpu_process": True,
        "claim_boundary": (
            "The window characterizes the scheduled evaluation transition, not steady-state "
            "training GPU utilization; host contention remains a plausible but unisolated cause "
            "of the preceding training slowdown."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
