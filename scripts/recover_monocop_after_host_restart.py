"""Prepare an auditable resume predecessor after a verified host restart."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.range_campaign import digest, write_json


def finite_state(value: object) -> bool:
    """Check nested model and optimizer tensors without instantiating CUDA."""
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_state(v) for v in value)
    return True


def main() -> None:
    """Preserve the damaged original and reserve a conservative spent-time budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Require CPU-only execution and a fresh recovery directory")
    torch.set_num_threads(2)
    lock_path = args.plan.parent / "SCREEN.lock"
    owner = json.loads(lock_path.read_text())
    local_owner = json.loads((args.run / "WRITER.lock").read_text())
    if (
        psutil.pid_exists(owner["pid"])
        or owner != local_owner
        or owner["created_epoch"] >= psutil.boot_time()
    ):
        raise ValueError("Old writer absence and subsequent host restart are not proven")
    contract = json.loads((args.run / "CONTRACT.json").read_text())
    plan = json.loads(args.plan.read_text())
    if digest(args.plan) != contract["plan_sha256"] or contract["sources"] != plan["sources"]:
        raise ValueError("Plan identity changed")
    for name, expected in plan["sources"].items():
        if digest(ROOT / name) != expected or digest(args.run / "code" / name) != expected:
            raise ValueError(f"Frozen producer differs: {name}")
    status_bytes = (args.run / "STATUS.json").read_bytes()
    if not status_bytes or any(status_bytes):
        raise ValueError("This recovery handles only an all-zero damaged status file")
    last = torch.load(
        args.run / "checkpoints/last.pt", map_location="cpu", weights_only=True, mmap=True
    )
    selected = torch.load(
        args.run / "checkpoints/selected.pt", map_location="cpu", weights_only=True, mmap=True
    )
    for state in (last, selected):
        if state["contract"] != contract or not finite_state((state["model"], state["optimizer"])):
            raise ValueError("Checkpoint lineage or numeric state is invalid")
    history = [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines()]
    last_logged = history[-1]["update"]
    if (
        [row["update"] for row in history] != list(range(1, last_logged + 1))
        or not selected["update"] <= last["update"] <= last_logged
        or selected["best"] != last["best"]
    ):
        raise ValueError("Interrupted history or checkpoint ordering differs")
    # Count all time up to reboot plus 60 s, including unsaved updates; downtime is excluded.
    reserved = max(
        last["training_seconds"], math.ceil(psutil.boot_time() - owner["created_epoch"]) + 60
    )
    if reserved >= contract["config"]["maximum_seconds"]:
        raise ValueError("Original time budget is already exhausted")
    args.output.mkdir(parents=True)
    files = [
        "CONTRACT.json",
        "ENVIRONMENT.json",
        "metrics.jsonl",
        "checkpoints/last.pt",
        "checkpoints/selected.pt",
        *[p.relative_to(args.run).as_posix() for p in (args.run / "evaluation").glob("*.json")],
    ]
    hashes = {}
    for name in files:
        source, target = args.run / name, args.output / name
        expected = digest(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != expected or digest(source) != expected:
            raise ValueError(f"Recovery copy differs: {name}")
        hashes[name] = expected
    for path in (args.run / "STATUS.json", args.run / "WRITER.lock", lock_path):
        dest = (
            args.output
            / "interruption"
            / ("global_" + path.name if path == lock_path else path.name)
        )
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, dest)
    receipt = {
        "schema": "phase_monocop_host_restart_recovery_v1",
        "original_run": args.run.as_posix(),
        "old_writer": owner,
        "boot_epoch": psutil.boot_time(),
        "old_writer_absent": True,
        "original_status_sha256": digest(args.run / "STATUS.json"),
        "original_status_condition": "all bytes are zero; original preserved",
        "restored_update": last["update"],
        "selected_update": selected["update"],
        "last_logged_update": last_logged,
        "replayed_logged_updates": last_logged - last["update"],
        "replayed_logged_exposures": (last_logged - last["update"])
        * contract["technical_config"]["batch_size"],
        "reserved_consumed_seconds": reserved,
        "budget_policy": "ceil(reboot_epoch - original_process_creation) + 60; no budget extension",
        "original_maximum_seconds": contract["config"]["maximum_seconds"],
        "checkpoint_cpu_load_and_finite_verified": True,
        "files": hashes,
        "recovery_script_sha256": digest(Path(__file__)),
    }
    write_json(args.output / "RECOVERY.json", receipt)
    shutil.copyfile(Path(__file__), args.output / Path(__file__).name)
    write_json(
        args.output / "STATUS.json",
        {
            "status": "RECOVERY_PREDECESSOR_ONLY",
            "update": last["update"],
            "selected_update": selected["update"],
            "best_aligned_mAP": last["best"],
            "elapsed_seconds": reserved,
            "elapsed_semantics": "conservative budget reservation",
        },
    )
    if psutil.pid_exists(owner["pid"]) or json.loads(lock_path.read_text()) != owner:
        raise ValueError("Writer identity changed before retiring stale global lock")
    lock_path.rename(args.plan.parent / "SCREEN.lock.retired_host_restart_20261102")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
