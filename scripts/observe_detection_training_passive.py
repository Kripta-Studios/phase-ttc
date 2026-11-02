"""Bounded passive observation of an existing detector training process."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

GPU_FIELDS = (
    "timestamp,index,utilization.gpu,utilization.memory,power.draw,clocks.sm,"
    "clocks.mem,temperature.gpu,clocks_throttle_reasons.active"
)


def _last_metric(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 131_072))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("update"), int):
            return value
    return None


def _evaluation_snapshot(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    return [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(directory.glob("*.json"))
    ]


def _process_io(process: psutil.Process) -> dict[str, int] | None:
    try:
        counters = process.io_counters()
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return None
    return {
        "read_bytes": int(counters.read_bytes),
        "write_bytes": int(counters.write_bytes),
        "read_count": int(counters.read_count),
        "write_count": int(counters.write_count),
    }


def _delta(after: dict[str, int] | None, before: dict[str, int] | None) -> dict[str, int] | None:
    if after is None or before is None:
        return None
    return {key: after[key] - before[key] for key in before}


def _gpu_reader(process: subprocess.Popen[str], rows: list[list[str]]) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        fields = [field.strip() for field in line.strip().split(",")]
        if len(fields) == 9:
            rows.append(fields)


def _finite_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _gpu_summary(rows: list[list[str]]) -> dict[str, Any]:
    names = (
        "utilization_gpu_percent",
        "utilization_memory_percent",
        "power_draw_w",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "temperature_c",
    )
    columns = list(zip(*rows, strict=True)) if rows else []
    summary: dict[str, Any] = {"samples": len(rows)}
    for name, column in zip(names, columns[2:8], strict=False):
        values = [parsed for item in column if (parsed := _finite_float(item)) is not None]
        summary[name] = (
            {
                "min": min(values),
                "median": statistics.median(values),
                "mean": statistics.fmean(values),
                "max": max(values),
            }
            if values
            else None
        )
    throttle = [row[8] for row in rows]
    summary["throttle_masks"] = sorted(set(throttle))
    summary["nonzero_throttle_samples"] = sum(
        mask.lower() not in {"0", "0x0000000000000000"} for mask in throttle
    )
    return summary


def main() -> int:
    """Observe for at most 60 seconds and write one bounded JSON receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 5.0 <= args.duration_seconds <= 60.0:
        raise ValueError("duration must be between 5 and 60 seconds")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    trainer = psutil.Process(args.pid)
    if "python" not in trainer.name().lower():
        raise ValueError("Observed PID is not the expected Python trainer")

    started = datetime.now(UTC)
    metric_start = _last_metric(args.metrics)
    evaluation_start = _evaluation_snapshot(args.evaluation_dir)
    io_start = _process_io(trainer)
    cpu_start = trainer.cpu_times()
    system_disk_start = psutil.disk_io_counters()
    trainer.cpu_percent(None)
    process_samples: list[dict[str, Any]] = []
    gpu_rows: list[list[str]] = []
    creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    gpu_process = subprocess.Popen(
        [
            "nvidia-smi",
            f"--query-gpu={GPU_FIELDS}",
            "--format=csv,noheader,nounits",
            "--loop-ms=1000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        creationflags=creation_flags,
    )
    reader = threading.Thread(target=_gpu_reader, args=(gpu_process, gpu_rows), daemon=True)
    reader.start()
    deadline = time.monotonic() + args.duration_seconds
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(1.0, remaining))
            try:
                memory = trainer.memory_info()
                process_samples.append(
                    {
                        "elapsed_seconds": args.duration_seconds
                        - max(0.0, deadline - time.monotonic()),
                        "cpu_percent_one_core_100": trainer.cpu_percent(None),
                        "rss_bytes": memory.rss,
                        "vms_bytes": memory.vms,
                        "threads": trainer.num_threads(),
                        "metric_update": (_last_metric(args.metrics) or {}).get("update"),
                        "evaluation_files": len(_evaluation_snapshot(args.evaluation_dir)),
                    }
                )
            except psutil.NoSuchProcess:
                break
    finally:
        gpu_process.terminate()
        try:
            gpu_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            gpu_process.kill()
            gpu_process.wait(timeout=3)
        reader.join(timeout=3)

    metric_end = _last_metric(args.metrics)
    evaluation_end = _evaluation_snapshot(args.evaluation_dir)
    io_end = _process_io(trainer)
    cpu_end = trainer.cpu_times()
    system_disk_end = psutil.disk_io_counters()
    elapsed = (datetime.now(UTC) - started).total_seconds()
    update_start = metric_start.get("update") if metric_start else None
    update_end = metric_end.get("update") if metric_end else None
    evaluation_changed = evaluation_end != evaluation_start
    phase = (
        "evaluation_or_checkpoint"
        if update_start == update_end and evaluation_changed
        else "training"
        if isinstance(update_start, int)
        and isinstance(update_end, int)
        and update_end > update_start
        else "transition_or_stall"
    )
    cpu_delta = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
    disk_delta = None
    if system_disk_start is not None and system_disk_end is not None:
        disk_delta = {
            "read_bytes": system_disk_end.read_bytes - system_disk_start.read_bytes,
            "write_bytes": system_disk_end.write_bytes - system_disk_start.write_bytes,
            "read_count": system_disk_end.read_count - system_disk_start.read_count,
            "write_count": system_disk_end.write_count - system_disk_start.write_count,
        }
    receipt = {
        "schema": "detection_training_passive_observation_v1",
        "status": "PASS_PASSIVE_OBSERVATION",
        "started_utc": started.isoformat(),
        "elapsed_seconds": elapsed,
        "pid": args.pid,
        "process_create_time": trainer.create_time(),
        "phase_classification": phase,
        "metrics": {
            "path": str(args.metrics.resolve(strict=True)),
            "start_update": update_start,
            "end_update": update_end,
            "updates_completed": (
                update_end - update_start
                if isinstance(update_start, int) and isinstance(update_end, int)
                else None
            ),
            "last_preprocessing_seconds": (
                metric_end.get("preprocessing_seconds") if metric_end else None
            ),
            "last_forward_backward_update_seconds": (
                metric_end.get("forward_backward_update_seconds") if metric_end else None
            ),
        },
        "evaluation": {"start": evaluation_start, "end": evaluation_end},
        "trainer": {
            "cpu_seconds_delta": cpu_delta,
            "mean_logical_core_equivalents": cpu_delta / elapsed if elapsed else None,
            "rss_bytes": {
                "min": min(item["rss_bytes"] for item in process_samples),
                "max": max(item["rss_bytes"] for item in process_samples),
            },
            "io_delta": _delta(io_end, io_start),
            "samples": process_samples,
        },
        "gpu": _gpu_summary(gpu_rows),
        "system_disk_delta": disk_delta,
        "observer_constraints": {
            "torch_imported": False,
            "cuda_context_created": False,
            "gpu_kernels_launched": False,
            "process_control_or_priority_changed": False,
            "power_policy_changed": False,
            "other_apps_closed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "phase": phase,
                "updates": receipt["metrics"]["updates_completed"],
                "gpu": receipt["gpu"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
