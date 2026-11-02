"""Single-writer sequential GPU queue; never promotes or extends a scientific budget."""

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
from e_jepa_ttc.detection.range_campaign import RangeConfig, digest, write_json  # noqa: E402


def main() -> None:
    """Run fixed-budget jobs or stop with a durable failure/budget receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.queue.read_text())
    if (
        set(plan) != {"schema", "admission", "admission_sha256", "jobs"}
        or plan["schema"] != "phase_range_queue_v1"
    ):
        raise ValueError("Invalid queue schema")
    admission_path = Path(plan["admission"])
    if digest(admission_path) != plan["admission_sha256"]:
        raise ValueError("Admission changed")
    admission = json.loads(admission_path.read_text())
    if admission["status"] != "PASS":
        raise ValueError("Panel not admitted")
    lock = args.queue.with_suffix(".lock")
    status_path = args.queue.with_suffix(".status.json")
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    state: dict[str, Any] = {
        "schema": "phase_range_queue_state_v1",
        "queue_sha256": digest(args.queue),
        "jobs": [],
        "status": "RUNNING",
    }
    process: subprocess.Popen | None = None
    try:
        for job in plan["jobs"]:
            if set(job) != {"config", "config_sha256", "output"}:
                raise ValueError("Unexpected job keys")
            config = Path(job["config"])
            output = Path(job["output"])
            if digest(config) != job["config_sha256"]:
                raise ValueError("Config changed")
            cfg = RangeConfig.load(config)
            if cfg.population_sha256 != admission["population_sha256"]:
                raise ValueError("Admission population differs")
            command = [
                sys.executable,
                str(ROOT / "scripts/train_detection_range.py"),
                "--config",
                str(config),
                "--output",
                str(output),
            ]
            if output.exists():
                candidates = [output, *sorted((output / "segments").glob("*"))]
                existing = candidates[-1]
                if (existing / "WRITER.lock").exists():
                    raise ValueError(
                        "Existing writer lock requires PID verification before recovery"
                    )
                complete = existing / "STATUS.json"
                saved_contract = existing / "CONTRACT.json"
                if (
                    not saved_contract.is_file()
                    or json.loads(saved_contract.read_text())["config_sha256"] != cfg.sha256
                ):
                    raise ValueError("Existing job contract differs")
                if (
                    complete.is_file()
                    and json.loads(complete.read_text()).get("status") == "COMPLETE"
                ):
                    state["jobs"].append({"output": str(existing), "status": "ALREADY_COMPLETE"})
                    continue
                checkpoint = existing / "checkpoints/last.pt"
                if not checkpoint.is_file():
                    raise ValueError("Incomplete job has no resumable checkpoint")
                next_output = output / "segments" / f"{len(candidates):04d}"
                command = [
                    sys.executable,
                    str(ROOT / "scripts/train_detection_range.py"),
                    "--config",
                    str(config),
                    "--output",
                    str(next_output),
                    "--resume",
                    str(checkpoint),
                ]
                output = next_output
            if args.queue.with_suffix(".stop").exists():
                state["status"] = "STOP_REQUESTED"
                break
            output.parent.mkdir(parents=True, exist_ok=True)
            log_path = output.with_suffix(".log")
            state["active_job"] = {
                "output": str(output),
                "config": str(config),
                "log": str(log_path),
            }
            write_json(status_path, state)
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                state["active_job"] = {**state["active_job"], "pid": process.pid}
                write_json(status_path, state)
                while process.poll() is None:
                    time.sleep(1)
                result = process.returncode
            status = (
                json.loads((output / "STATUS.json").read_text())
                if (output / "STATUS.json").is_file()
                else {}
            )
            state["jobs"].append({"output": str(output), "returncode": result, "result": status})
            if result != 0:
                state["status"] = "FAILED"
                break
            if status.get("status") != "COMPLETE":
                state["status"] = "BUDGET_OR_SEGMENT_STOP"
                break
        else:
            state["status"] = "COMPLETE"
        state.pop("active_job", None)
        write_json(status_path, state)
        if state["status"] != "COMPLETE":
            raise RuntimeError(f"Queue stopped: {state['status']}")
    except Exception as error:
        state["status"] = "FAILED"
        state["error"] = str(error)
        write_json(status_path, state)
        raise
    finally:
        if process is None or process.poll() is not None:
            lock.unlink()
        else:
            state["status"] = "OWNER_INTERRUPTED_CHILD_MAY_BE_RUNNING"
            write_json(status_path, state)


if __name__ == "__main__":
    main()
