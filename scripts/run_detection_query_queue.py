"""Run admitted query-quality stages after the paired GPU range campaign finishes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def main() -> None:
    """Serialize GPU work; stop on failed parity, changed sources or failed stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--range-queue", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    panel, plan = str(args.panel.resolve()), str(args.plan.resolve())
    parity, inference, heads = root / "parity", root / "train_inference", root / "heads"
    script_names = (
        "run_detection_query_queue",
        "infer_detection_query_panel",
        "infer_detection_release",
        "train_detection_query_quality",
        "evaluate_detection_query_quality",
        "evaluate_detection_benchmark_local",
    )
    sources = {
        p.relative_to(ROOT).as_posix(): digest(p)
        for p in [
            *[ROOT / f"scripts/{name}.py" for name in script_names],
            *[
                ROOT / f"src/e_jepa_ttc/detection/{name}.py"
                for name in (
                    "indexed_rgb_data",
                    "query_quality",
                    "quality_ranking",
                    "benchmark_evaluation",
                    "range_campaign",
                )
            ],
        ]
    }
    jobs = [
        {
            "name": "parity",
            "argv": [
                "infer_detection_query_panel",
                "--panel",
                panel,
                "--role",
                "parity",
                "--output",
                str(parity),
            ],
            "receipt": str(parity / "PARITY.json"),
        },
        {
            "name": "train_inference",
            "argv": [
                "infer_detection_query_panel",
                "--panel",
                panel,
                "--role",
                "train",
                "--parity-receipt",
                str(parity / "PARITY.json"),
                "--output",
                str(inference),
            ],
            "receipt": str(inference / "COMPLETE.json"),
        },
        {
            "name": "fit_heads",
            "argv": [
                "train_detection_query_quality",
                "--panel",
                panel,
                "--plan",
                plan,
                "--inference",
                str(inference),
                "--output",
                str(heads),
            ],
            "receipt": str(heads / "COMPLETE.json"),
        },
    ]
    for arm in ("linear_16_to_8", "mlp_16_to_64_gelu_to_8"):
        output = root / f"evaluation_{arm}"
        jobs.append(
            {
                "name": f"evaluate_{arm}",
                "argv": [
                    "evaluate_detection_query_quality",
                    "--shards",
                    str(args.shards.resolve()),
                    "--aligned",
                    str(args.aligned.resolve()),
                    "--plan",
                    plan,
                    "--head",
                    str(heads / arm / "head.pt"),
                    "--output",
                    str(output),
                ],
                "receipt": str(output / "RESULTS.json"),
            }
        )
    contract = {
        "schema": "phase_query_queue_v1",
        "sources": sources,
        "jobs": jobs,
        "panel_sha256": digest(args.panel),
        "plan_sha256": digest(args.plan),
        "range_queue_sha256": digest(args.range_queue),
        "maximum_wait_seconds": 14400,
    }
    contract_path = root / "QUEUE.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Existing queue contract differs")
    else:
        if any(root.iterdir()):
            raise ValueError("Unowned queue directory")
        write_json(contract_path, contract)
    status_path, lock = root / "QUEUE.status.json", root / "QUEUE.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    results = previous.get("completed", {})
    state = {
        "status": "WAITING_FOR_RANGE",
        "completed": results,
        "queue_sha256": digest(contract_path),
    }
    child = None
    try:
        write_json(status_path, state)
        started = time.perf_counter()
        while True:
            if digest(args.range_queue) != contract["range_queue_sha256"]:
                raise ValueError("Range queue changed")
            range_status = json.loads(args.range_queue.with_suffix(".status.json").read_text())
            if (
                range_status["status"] == "COMPLETE"
                and not args.range_queue.with_suffix(".lock").exists()
            ):
                break
            if range_status["status"] not in ("RUNNING", "COMPLETE"):
                raise RuntimeError(f"Range queue stopped: {range_status['status']}")
            if time.perf_counter() - started > contract["maximum_wait_seconds"]:
                raise TimeoutError("Range dependency wait budget exhausted")
            time.sleep(10)
        for job in jobs:
            for name, expected in sources.items():
                if digest(ROOT / name) != expected:
                    raise ValueError(f"Admitted source changed: {name}")
            if (
                digest(args.panel) != contract["panel_sha256"]
                or digest(args.plan) != contract["plan_sha256"]
            ):
                raise ValueError("Panel or plan changed")
            name, receipt = job["name"], Path(job["receipt"])
            if name in results:
                if digest(receipt) != results[name]["receipt_sha256"]:
                    raise ValueError("Completed stage receipt changed")
                continue
            argv = job["argv"]
            command = [sys.executable, str(ROOT / f"scripts/{argv[0]}.py"), *argv[1:]]
            state.update(status="RUNNING", active_job=name)
            write_json(status_path, state)
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
                result = child.wait()
            if result != 0:
                raise RuntimeError(f"Stage {name} failed with exit code {result}; preserve its log")
            if not receipt.exists():
                raise ValueError("Successful stage omitted its completion receipt")
            results[name] = {"receipt_sha256": digest(receipt), "returncode": result}
            state["completed"] = results
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
