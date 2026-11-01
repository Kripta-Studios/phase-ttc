"""Run a bounded external-initialization experiment after the existing GPU queues."""

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
from e_jepa_ttc.detection.pretrained_campaign import PretrainedConfig  # noqa: E402
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def main() -> None:
    """Verify every completed dependency; a failed learning gate never starts a screen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--after-queue", required=True, type=Path)
    parser.add_argument("--confirmation-queue", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_dir = args.plan_dir.resolve()
    plan_path = plan_dir / "PLAN.json"
    plan = json.loads(plan_path.read_text())
    for name in ("smoke", "screen"):
        path = plan_dir / f"{name}.json"
        if digest(path) != plan[f"{name}_config_sha256"]:
            raise ValueError("Registered configuration changed")
        PretrainedConfig.load(path)
    modules = (
        "pretrained_campaign",
        "pretrained_initialization",
        "long_range_campaign",
        "range_campaign",
        "benchmark_evaluation",
        "campaign_training",
        "compact_training_cache",
        "resident_input_cache",
        "eap_rgb_data",
        "indexed_rgb_data",
        "monodgp_eap_adapter",
        "geometry_contract",
        "development_metrics",
        "portable_attention",
        "evaluation_protocol",
        "velocity_extension",
    )
    scripts = (
        "run_detection_pretrained_queue",
        "train_detection_pretrained",
        "audit_detection_pretrained_smoke",
        "audit_detection_range_smoke",
        "evaluate_detection_extended_range",
        "smoke_monodgp_portable",
    )
    paths = [
        *[ROOT / f"src/e_jepa_ttc/detection/{n}.py" for n in modules],
        *[ROOT / f"scripts/{n}.py" for n in scripts],
    ]
    sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    for relative, expected in plan["sources"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Source differs from preregistration: {relative}")
    contract = {
        "schema": "phase_pretrained_queue_v1",
        "sources": sources,
        "plan_sha256": digest(plan_path),
        "after_queue_sha256": digest(args.after_queue),
        "confirmation_queue_sha256": digest(args.confirmation_queue),
        "maximum_wait_seconds": 14400,
        "new_cloud_spend": False,
        "promotion": False,
    }
    contract_path = root / "QUEUE.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Existing queue differs")
    write_json(contract_path, contract)
    for relative, expected in sources.items():
        archive = root / "code" / relative
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists() and digest(archive) != expected:
            raise ValueError("Archived source changed")
        if not archive.exists():
            archive.write_bytes((ROOT / relative).read_bytes())
    lock = root / "QUEUE.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    status_path = root / "QUEUE.status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    state: dict[str, Any] = {
        "status": "WAITING_FOR_RANGE300",
        "completed": previous.get("completed", {}),
        "queue_sha256": digest(contract_path),
    }
    child = None
    try:
        write_json(status_path, state)
        start = time.perf_counter()
        while True:
            if (root / "STOP").exists():
                raise RuntimeError("Stop requested before next stage")
            for path, key in (
                (args.after_queue, "after_queue_sha256"),
                (args.confirmation_queue, "confirmation_queue_sha256"),
            ):
                if digest(path) != contract[key]:
                    raise ValueError("GPU dependency contract changed")
            after = json.loads(args.after_queue.with_suffix(".status.json").read_text())
            confirmation = json.loads(
                args.confirmation_queue.with_suffix(".status.json").read_text()
            )
            if confirmation["status"] == "FAILED":
                raise RuntimeError("Prior confirmation failed")
            if after["status"] == "COMPLETE":
                if (
                    args.after_queue.with_suffix(".lock").exists()
                    or args.confirmation_queue.with_suffix(".lock").exists()
                ):
                    time.sleep(1)
                    continue
                if confirmation["status"] != "COMPLETE":
                    raise ValueError("GPU confirmation is not complete")
                full = args.confirmation_queue.parent / "full_selection/RESULTS.json"
                if digest(full) != confirmation["results_sha256"]:
                    raise ValueError("Prior full-selection results changed")
                for name, relative in (
                    ("smoke", "smoke/STATUS.json"),
                    ("smoke_gate", "SMOKE_GATE.json"),
                    ("screen", "screen/STATUS.json"),
                ):
                    receipt = args.after_queue.parent / relative
                    record = after["completed"][name]
                    if record["returncode"] or digest(receipt) != record["receipt_sha256"]:
                        raise ValueError("Range control receipt changed")
                break
            if after["status"] == "FAILED":
                intentional = (
                    after.get("error") == "Stop requested before next stage"
                    and not after.get("completed")
                    and confirmation["status"] in ("WAITING_FOR_QUERY_GPU", "RUNNING")
                )
                if not intentional:
                    raise RuntimeError("Prior 300m queue failed")
            if time.perf_counter() - start > contract["maximum_wait_seconds"]:
                raise TimeoutError("GPU dependency wait budget exhausted")
            time.sleep(10)
        jobs = [
            (
                "smoke",
                "train_detection_pretrained",
                ["--config", str(plan_dir / "smoke.json"), "--output", str(root / "smoke")],
                root / "smoke/STATUS.json",
                "COMPLETE",
            ),
            (
                "smoke_replay",
                "evaluate_detection_extended_range",
                [
                    "--checkpoint",
                    str(root / "smoke/checkpoints/last.pt"),
                    "--panel-only",
                    "--output",
                    str(root / "smoke_replay"),
                ],
                root / "smoke_replay/STATUS.json",
                "COMPLETE",
            ),
            (
                "smoke_gate",
                "audit_detection_pretrained_smoke",
                [
                    "--plan-dir",
                    str(plan_dir),
                    "--run",
                    str(root / "smoke"),
                    "--reference",
                    str(root / "smoke_replay"),
                    "--output",
                    str(root / "SMOKE_GATE.json"),
                ],
                root / "SMOKE_GATE.json",
                "PASS",
            ),
            (
                "screen",
                "train_detection_pretrained",
                ["--config", str(plan_dir / "screen.json"), "--output", str(root / "screen")],
                root / "screen/STATUS.json",
                "COMPLETE",
            ),
        ]
        for name, script, argv, receipt, expected_status in jobs:
            for relative, expected in sources.items():
                if digest(ROOT / relative) != expected:
                    raise ValueError(f"Frozen source changed: {relative}")
            if digest(plan_path) != contract["plan_sha256"]:
                raise ValueError("Plan changed")
            if name in state["completed"]:
                if digest(receipt) != state["completed"][name]["receipt_sha256"]:
                    raise ValueError("Completed receipt changed")
                continue
            if (root / "STOP").exists():
                raise RuntimeError("Stop requested before next stage")
            state.update(status="RUNNING", active_job=name)
            write_json(status_path, state)
            command = [sys.executable, str(ROOT / f"scripts/{script}.py"), *argv]
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
                raise RuntimeError(f"{name} failed; inspect preserved logs")
            if json.loads(receipt.read_text()).get("status") != expected_status:
                raise ValueError(f"{name} did not meet its gate")
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
