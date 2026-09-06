"""Run one source checkout's CPU test suite with durable subprocess optimizer accounting."""

# ruff: noqa: E402 -- explicit repository and isolated QA utility bootstrap
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "scripts/qa_support"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SUPPORT))

import psutil
from stage70_optimizer_audit import initialize, snapshot

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.training.stage70_update_ledger import UpdateLedger


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *arguments], text=True, encoding="utf-8"
    ).strip()


def expansion_reservation(campaign: Path, expected_identity: str | None = None) -> tuple[int, str]:
    """Account for the live, fixed 27,307-row cache without reading source data."""
    progress = json.loads(
        (campaign / "counts_expansion27307/CACHE_PROGRESS.json").read_text(encoding="utf-8")
    )
    identity = progress["identity_sha256"]
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("concurrent expansion cache identity changed")
    completed = progress["completed_rows"]
    shards = progress["shards"]
    if not 0 <= completed <= 27307 or completed != sum(shard["rows"] for shard in shards):
        raise ValueError("concurrent expansion cache progress is inconsistent")
    # uint32 is the conservative whole-cache fallback; the present cache is uint16.
    itemsize = 4
    if shards:
        records = [
            value
            for shard in shards
            for key, value in shard["components"].items()
            if key.startswith("counts_")
        ]
        dtypes = {record["dtype"] for record in records}
        if dtypes not in ({"uint16"}, {"uint32"}) or any(
            record["shape"][1:] != [2, 8, 2, 128, 128] for record in records
        ):
            raise ValueError("concurrent expansion cache representation changed")
        itemsize = 2 if dtypes == {"uint16"} else 4
    return (27307 - completed) * (2 * 8 * 2 * 128 * 128 * itemsize + 4096) + 4 * 1024**2, identity


def main() -> None:
    """Reserve the complete QA allowance before any test process imports Torch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--audit-id", required=True)
    parser.add_argument("--update-allowance", type=int, required=True)
    parser.add_argument("--disk-reservation-bytes", type=int, required=True)
    parser.add_argument("--ram-reservation-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--track-expansion-cache", action="store_true")
    parser.add_argument("tests", nargs="*", default=["tests"])
    args = parser.parse_args()
    checkout = args.checkout.resolve(strict=True)
    if checkout not in (ROOT, ROOT.parent / "e-jepa-ttc-v10-stage70-baseline-qa"):
        raise ValueError("test audit must use the current or isolated pinned baseline checkout")
    if _git(checkout, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("test source must be committed and clean before execution")
    commit = _git(checkout, "rev-parse", "HEAD")
    if checkout != ROOT and commit != "57b39cb2b9a5ec8378755f8f350634c007aa822e":
        raise ValueError("baseline source commit differs from the mandatory source pin")
    if not args.audit_id or any(
        c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in args.audit_id
    ):
        raise ValueError("audit ID must be a simple lowercase identifier")
    for selected in args.tests:
        if not (checkout / selected).resolve(strict=True).is_relative_to(checkout / "tests"):
            raise ValueError("test selection must remain within the selected checkout tests")
    campaign = ROOT / "artifacts/stage70_76_architecture"
    output = campaign / "baseline_qa" / args.audit_id
    if output.exists():
        raise ValueError(
            "immutable test audit already exists; inspect its state instead of rerunning"
        )
    if args.disk_reservation_bytes < 3 * 1024**3 or args.ram_reservation_bytes < 4 * 1024**3:
        raise ValueError("suite requires at least 3 GiB disk and 4 GiB incremental RAM reservation")
    recipe = {
        "kind": "CPU_PYTEST_AUDIT",
        "checkout": str(checkout),
        "commit": commit,
        "tests": args.tests,
        "allowance": args.update_allowance,
        "files": [binding(path) for path in [Path(__file__), *sorted(SUPPORT.glob("*.py"))]],
        "runtime": {"python": sys.version, "cuda_visible_devices": "-1", "threads": 2},
    }
    identity = object_digest(recipe)
    concurrent_bytes, cache_identity = (
        expansion_reservation(campaign) if args.track_expansion_cache else (0, None)
    )
    with ResourceOwner(
        campaign / "training_writer",
        remaining_bytes=args.disk_reservation_bytes + concurrent_bytes,
        planned_ram_bytes=args.ram_reservation_bytes,
    ) as owner:
        ledger = UpdateLedger(campaign / "UPDATE_LEDGER.json", owner.root)
        ledger.reserve_technical_audit(args.audit_id, args.update_allowance, identity)
        output.mkdir(parents=True)
        atomic_json(output / "RECIPE.json", {**recipe, "sha256": identity})
        database = output / "optimizer_audit.sqlite"
        initialize(database, args.update_allowance, identity)
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTHONPATH": str(SUPPORT),
            "STAGE70_QA_OPTIMIZER_DB": str(database),
            "STAGE70_QA_IDENTITY": identity,
            "STAGE70_QA_OUTPUT": str(output / "capture"),
        }
        command = [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "-p",
            "stage70_capture",
            f"--junitxml={output / 'results.xml'}",
            *args.tests,
        ]
        interruption = None
        with (output / "stdout.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=checkout, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            try:
                while process.poll() is None:
                    if args.track_expansion_cache:
                        concurrent_bytes, _ = expansion_reservation(campaign, cache_identity)
                        owner.estimated_remaining_bytes = (
                            args.disk_reservation_bytes + concurrent_bytes
                        )
                    owner.check()
                    progress = {**snapshot(database), "pid": process.pid, "timestamp": time.time()}
                    atomic_json(output / "PROGRESS.json", progress)
                    print(
                        f"QA {args.audit_id}: {progress['confirmed']} confirmed updates; "
                        "pytest running",
                        flush=True,
                    )
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
            except BaseException as error:
                interruption = repr(error)
                if process.poll() is None:
                    parent = psutil.Process(process.pid)
                    children = parent.children(recursive=True)
                    for child in reversed(children):
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    process.terminate()
                    process.wait()
                    psutil.wait_procs(children, timeout=10)
        with sqlite3.connect(database) as connection:
            active = []
            for pid, created in connection.execute("SELECT pid, started FROM processes"):
                try:
                    if psutil.Process(pid).create_time() == created:
                        active.append(pid)
                except psutil.NoSuchProcess:
                    pass
        if active:
            raise RuntimeError(f"QA descendants remain active; reservation retained: {active}")
        receipt = {
            **snapshot(database),
            "all_processes_closed": True,
            "pytest_exit_code": process.returncode,
            "interruption": interruption,
            "database": binding(database),
            "stdout": binding(output / "stdout.log"),
            "scientific_endpoint": False,
        }
        receipt_path = output / "CLOSED_ACCOUNTING.json"
        atomic_json(receipt_path, {**receipt, "sha256": object_digest(receipt)})
        ledger.settle_technical_audit(args.audit_id, receipt_path)
        print(json.dumps(ledger.totals()), flush=True)
    if interruption:
        raise RuntimeError(interruption)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
