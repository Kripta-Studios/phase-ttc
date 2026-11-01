"""Confirm the selected 150m checkpoint before resuming the queued 300m experiment."""

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
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def main() -> None:
    """Serialize panel parity/full selection and preserve failure before any promotion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-panel", type=Path, required=True)
    parser.add_argument("--query-queue", type=Path, required=True)
    parser.add_argument("--range-queue", type=Path, required=True)
    parser.add_argument("--long-range-root", type=Path, required=True)
    parser.add_argument("--long-range-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    deferred = args.long_range_root.resolve()
    deferral = json.loads((deferred / "DEFERRED_FOR_CONFIRMATION.json").read_text())
    if digest(args.checkpoint) != deferral["expected_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint differs from deferral")
    names = [
        Path(__file__),
        ROOT / "scripts/evaluate_detection_range.py",
        ROOT / "src/e_jepa_ttc/detection/indexed_rgb_data.py",
    ]
    sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in names}
    contract = {
        "schema": "phase_range150_confirmation_queue_v1",
        "sources": sources,
        "checkpoint_sha256": digest(args.checkpoint),
        "reference_panel_sha256": digest(args.reference_panel),
        "query_queue_sha256": digest(args.query_queue),
        "deferred_queue_sha256": digest(deferred / "QUEUE.json"),
        "deferral_sha256": digest(deferred / "DEFERRED_FOR_CONFIRMATION.json"),
        "scope": "512-frame producer parity followed by fixed full public selection",
        "reason": "Registered 60/150 plan calls for confirming selected checkpoints; "
        "panel gain exceeds 0.005 over control and is positive over incumbent",
        "promotion": False,
        "new_cloud_spend": False,
        "maximum_wait_seconds": 14400,
    }
    contract_path = root / "QUEUE.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Existing confirmation queue differs")
    write_json(contract_path, contract)
    for relative, expected in sources.items():
        archive = root / "code" / relative
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists() and digest(archive) != expected:
            raise ValueError("Archived confirmation source changed")
        if not archive.exists():
            archive.write_bytes((ROOT / relative).read_bytes())
    lock = root / "QUEUE.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    status_path = root / "QUEUE.status.json"
    state: dict[str, Any] = {"status": "WAITING_FOR_QUERY_GPU", "promotion": False}
    child = None
    try:
        write_json(status_path, state)
        start = time.perf_counter()
        while True:
            if digest(args.query_queue) != contract["query_queue_sha256"]:
                raise ValueError("Query queue changed")
            queries = json.loads(args.query_queue.with_suffix(".status.json").read_text())
            long_status = json.loads((deferred / "QUEUE.status.json").read_text())
            completed = queries.get("completed", {}).get("train_inference")
            if completed:
                receipt = args.query_queue.parent / "train_inference/COMPLETE.json"
                if completed["returncode"] or digest(receipt) != completed["receipt_sha256"]:
                    raise ValueError("GPU inference completion differs")
                if (
                    long_status["status"] == "FAILED"
                    and long_status.get("error") == "Stop requested before next stage"
                    and not (deferred / "QUEUE.lock").exists()
                ):
                    break
            if queries["status"] == "FAILED" and not completed:
                raise RuntimeError("Preceding GPU inference failed")
            if time.perf_counter() - start > contract["maximum_wait_seconds"]:
                raise TimeoutError("Confirmation dependency wait budget exhausted")
            time.sleep(10)
        for stage, extra in (("panel", ["--panel-only"]), ("full_selection", [])):
            for relative, expected in sources.items():
                if digest(ROOT / relative) != expected:
                    raise ValueError(f"Confirmation source changed: {relative}")
            if digest(args.checkpoint) != contract["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed")
            state.update(status="RUNNING", active_job=stage)
            write_json(status_path, state)
            command = [
                sys.executable,
                str(ROOT / "scripts/evaluate_detection_range.py"),
                "--checkpoint",
                str(args.checkpoint),
                "--output",
                str(root / stage),
                *extra,
            ]
            with (root / f"{stage}.log").open("a", encoding="utf-8") as log:
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
            if code:
                raise RuntimeError(f"Confirmation {stage} failed; preserve logs and shards")
            finished = json.loads((root / stage / "STATUS.json").read_text())
            if finished["status"] != "COMPLETE":
                raise ValueError("Evaluation omitted completion")
            if stage == "panel":
                if digest(args.reference_panel) != contract["reference_panel_sha256"]:
                    raise ValueError("Reference panel changed")
                reference = json.loads(args.reference_panel.read_text())
                result = json.loads((root / stage / "RESULTS.json").read_text())
                predictions = {}
                for path in sorted((root / stage / "shards").glob("*/predictions.json")):
                    predictions.update(json.loads(path.read_text()))
                if (
                    predictions != reference["predictions"]
                    or result["metrics"] != reference["metrics"]
                ):
                    raise ValueError("Indexed evaluator differs from the training evaluator")
                write_json(
                    root / "PANEL_PARITY.json",
                    {
                        "status": "PASS",
                        "frames": len(predictions),
                        "exact_predictions": True,
                        "exact_metrics": True,
                        "reference_sha256": contract["reference_panel_sha256"],
                        "results_sha256": digest(root / stage / "RESULTS.json"),
                    },
                )
        if digest(deferred / "STOP") != deferral["stop_marker_sha256"]:
            raise ValueError("Deferral marker changed; do not override another stop request")
        if (deferred / "QUEUE.lock").exists():
            raise ValueError("Deferred queue writer still present")
        (deferred / "STOP").unlink()
        command = [
            sys.executable,
            str(ROOT / "scripts/run_detection_long_range_queue.py"),
            "--plan-dir",
            str(args.long_range_plan),
            "--query-queue",
            str(args.query_queue),
            "--range-queue",
            str(args.range_queue),
            "--output",
            str(deferred),
        ]
        with (root / "resume_long_range.log").open("a", encoding="utf-8") as log:
            resumed = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        state.update(
            status="COMPLETE",
            active_job=None,
            pid=None,
            resumed_long_range_pid=resumed.pid,
            results_sha256=digest(root / "full_selection/RESULTS.json"),
        )
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
