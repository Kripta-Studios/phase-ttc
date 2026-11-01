"""Run the bounded 300m smoke and screen after existing GPU inference is released."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.long_range_campaign import LongRangeConfig  # noqa: E402
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def main() -> None:
    """Wait for verified GPU dependencies; no concurrent GPU launch or promotion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--query-queue", required=True, type=Path)
    parser.add_argument("--range-queue", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_dir = args.plan_dir.resolve()
    plan = json.loads((plan_dir / "PLAN.json").read_text())
    for name, key in (
        ("range300.json", "screen_config_sha256"),
        ("smoke300.json", "smoke_config_sha256"),
    ):
        if digest(plan_dir / name) != plan[key]:
            raise ValueError("Registered config changed")
        LongRangeConfig.load(plan_dir / name)
    modules = (
        "long_range_campaign",
        "range_campaign",
        "benchmark_evaluation",
        "campaign_training",
        "compact_training_cache",
        "resident_input_cache",
        "eap_rgb_data",
        "monodgp_eap_adapter",
        "geometry_contract",
        "development_metrics",
        "portable_attention",
        "evaluation_protocol",
        "velocity_extension",
    )
    scripts = (
        "run_detection_long_range_queue",
        "train_detection_long_range",
        "audit_detection_long_range_smoke",
        "audit_detection_range_smoke",
        "smoke_monodgp_portable",
    )
    sources = {
        p.relative_to(ROOT).as_posix(): digest(p)
        for p in [
            *[ROOT / f"scripts/{n}.py" for n in scripts],
            *[ROOT / f"src/e_jepa_ttc/detection/{n}.py" for n in modules],
        ]
    }
    if sources["scripts/train_detection_long_range.py"] != plan["producer_source_sha256"]:
        raise ValueError("Producer differs from registration")
    if sources["src/e_jepa_ttc/detection/long_range_campaign.py"] != plan["config_source_sha256"]:
        raise ValueError("Config implementation differs from registration")
    contract = {
        "schema": "phase_long_range_queue_v1",
        "sources": sources,
        "plan_sha256": digest(plan_dir / "PLAN.json"),
        "query_queue_sha256": digest(args.query_queue),
        "range_queue_sha256": digest(args.range_queue),
        "maximum_wait_seconds": 14400,
        "smoke_maximum_seconds": 600,
        "screen_maximum_seconds": 7200,
    }
    contract_path = root / "QUEUE.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Existing queue differs")
    write_json(contract_path, contract)
    for relative in sources:
        destination = root / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and digest(destination) != sources[relative]:
            raise ValueError("Archived source changed")
        if not destination.exists():
            destination.write_bytes((ROOT / relative).read_bytes())
    lock = root / "QUEUE.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    status_path = root / "QUEUE.status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    state: dict[str, Any] = {
        "status": "WAITING_FOR_QUERY_GPU",
        "queue_sha256": digest(contract_path),
        "completed": previous.get("completed", {}),
    }
    child = None
    try:
        write_json(status_path, state)
        start = time.perf_counter()
        while True:
            if digest(args.query_queue) != contract["query_queue_sha256"]:
                raise ValueError("Query queue changed")
            if digest(args.range_queue) != contract["range_queue_sha256"]:
                raise ValueError("Range queue changed")
            queries = json.loads(args.query_queue.with_suffix(".status.json").read_text())
            ranges = json.loads(args.range_queue.with_suffix(".status.json").read_text())
            completed = queries.get("completed", {}).get("train_inference")
            if completed:
                receipt = args.query_queue.parent / "train_inference/COMPLETE.json"
                if completed["returncode"] != 0 or digest(receipt) != completed["receipt_sha256"]:
                    raise ValueError("Query GPU completion receipt differs")
                if ranges["status"] != "COMPLETE" or args.range_queue.with_suffix(".lock").exists():
                    raise ValueError("Range GPU dependency is not complete")
                break
            if queries["status"] == "FAILED" or ranges["status"] not in ("RUNNING", "COMPLETE"):
                raise RuntimeError("An earlier GPU dependency failed")
            if time.perf_counter() - start > contract["maximum_wait_seconds"]:
                raise TimeoutError("GPU wait budget exhausted")
            time.sleep(10)
        jobs = [
            (
                "smoke",
                [
                    "train_detection_long_range",
                    "--config",
                    str(plan_dir / "smoke300.json"),
                    "--output",
                    str(root / "smoke"),
                ],
                root / "smoke/STATUS.json",
            ),
            (
                "smoke_gate",
                [
                    "audit_detection_long_range_smoke",
                    "--plan-dir",
                    str(plan_dir),
                    "--run",
                    str(root / "smoke"),
                    "--reference",
                    str(
                        args.range_queue.parent.parent / "screen_range150_v1/evaluation/000000.json"
                    ),
                    "--output",
                    str(root / "SMOKE_GATE.json"),
                ],
                root / "SMOKE_GATE.json",
            ),
            (
                "screen",
                [
                    "train_detection_long_range",
                    "--config",
                    str(plan_dir / "range300.json"),
                    "--output",
                    str(root / "screen"),
                ],
                root / "screen/STATUS.json",
            ),
        ]
        for name, argv, receipt in jobs:
            for relative, expected in sources.items():
                if digest(ROOT / relative) != expected:
                    raise ValueError(f"Frozen source changed: {relative}")
            if digest(plan_dir / "PLAN.json") != contract["plan_sha256"]:
                raise ValueError("Plan changed")
            if name in state["completed"]:
                if digest(receipt) != state["completed"][name]["receipt_sha256"]:
                    raise ValueError("Completed receipt changed")
                continue
            if (root / "STOP").exists():
                raise RuntimeError("Stop requested before next stage")
            state.update(status="RUNNING", active_job=name)
            write_json(status_path, state)
            command = [sys.executable, str(ROOT / f"scripts/{argv[0]}.py"), *argv[1:]]
            with (root / f"{name}.log").open("a", encoding="utf-8") as log:
                child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                state["pid"] = child.pid
                write_json(status_path, state)
                code = child.wait()
            if code or not receipt.is_file():
                raise RuntimeError(f"{name} failed; inspect its preserved log")
            result = json.loads(receipt.read_text())
            expected_status = "PASS" if name == "smoke_gate" else "COMPLETE"
            if result.get("status") != expected_status:
                raise ValueError(f"{name} did not meet its completion gate")
            state["completed"][name] = {"receipt_sha256": digest(receipt), "returncode": code}
            write_json(status_path, state)
        state.update(status="COMPLETE", active_job=None, pid=None)
        write_json(status_path, state)
    except Exception as error:
        state.update(status="FAILED", error=str(error))
        write_json(status_path, state)
        raise
    finally:
        if child is None or child.poll() is not None:
            lock.unlink()


if __name__ == "__main__":
    main()
