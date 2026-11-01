"""Freeze a CPU-validated MonoCoP smoke plan without admitting or starting GPU work."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.benchmark_evaluation import PROFILE_SHA256
from e_jepa_ttc.detection.monocop_smoke import (
    INITIAL_SHA256,
    POPULATION_SHA256,
    PORT_SHA256,
    MonoCoPSmokeConfig,
    verify_inputs,
)
from e_jepa_ttc.detection.range_campaign import digest, write_json


def main() -> None:
    """Record immutable local inputs and the identities of the active predecessor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-run", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--population", required=True, type=Path)
    parser.add_argument("--predecessor-queue", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = MonoCoPSmokeConfig(
        schema="phase_monocop_gpu_smoke_v1",
        experiment_id="monocop-all-range300-seed7-gpu-smoke-v1",
        dataset_root=args.dataset_root.resolve().as_posix(),
        population=args.population.resolve().as_posix(),
        population_sha256=POPULATION_SHA256,
        portable_source=args.source.resolve().as_posix(),
        port_sha256=PORT_SHA256,
        initialized_checkpoint=(args.cpu_run.resolve() / "initialized_model.pt").as_posix(),
        initialized_checkpoint_sha256=INITIAL_SHA256,
        cpu_result=(args.cpu_run.resolve() / "RESULTS.json").as_posix(),
        cpu_result_sha256=digest(args.cpu_run / "RESULTS.json"),
        evaluation_profile_sha256=PROFILE_SHA256,
        seed=7,
        width=640,
        height=384,
        depth_max_m=300.0,
        batch_size=2,
        updates=32,
        evaluation_updates=[0, 16, 32],
        learning_rate=5e-5,
        backbone_lr_multiplier=0.1,
        weight_decay=1e-4,
        gradient_clip=1.0,
        maximum_seconds=600.0,
        maximum_cuda_bytes=9 * 2**30,
        cpu_threads=4,
    )
    verify_inputs(cfg)
    predecessor = args.predecessor_queue.resolve()
    if json.loads(predecessor.read_text())["schema"] != "phase_pretrained_queue_v1":
        raise ValueError("Expected the registered KITTI predecessor")
    processes = []
    for lock in (predecessor.with_suffix(".lock"), predecessor.parent / "screen/WRITER.lock"):
        pid = int(lock.read_text())
        process = psutil.Process(pid)
        processes.append(
            {"pid": pid, "created_epoch": process.create_time(), "lock_source": lock.as_posix()}
        )
    module_names = (
        "monocop_smoke",
        "monocop_initialization",
        "range_campaign",
        "benchmark_evaluation",
        "campaign_training",
        "eap_rgb_data",
        "indexed_rgb_data",
        "monodgp_eap_adapter",
        "geometry_contract",
        "development_metrics",
        "portable_attention",
        "evaluation_protocol",
    )
    files = [
        Path(__file__),
        ROOT / "scripts/run_monocop_gpu_smoke.py",
        ROOT / "scripts/prepare_monocop_portable.py",
        ROOT / "scripts/audit_monocop_cpu_port.py",
        *[ROOT / f"src/e_jepa_ttc/detection/{name}.py" for name in module_names],
    ]
    if torch.cuda.is_initialized():
        raise ValueError("Preparation must not initialize CUDA")
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = args.output / "smoke.json"
    write_json(config_path, asdict(cfg))
    doc = ROOT / "docs/detection_sota_20261029/MONOCOP_GPU_SMOKE_PLAN_20261101.md"
    (args.output / doc.name).write_bytes(doc.read_bytes())
    plan = {
        "schema": "phase_monocop_gpu_smoke_plan_v1",
        "status": "PREPARED_NOT_ADMITTED",
        "config_sha256": cfg.sha256,
        "config_file_sha256": digest(config_path),
        "sources": {path.relative_to(ROOT).as_posix(): digest(path) for path in files},
        "git_commit_at_preparation": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "execution_tree_note": "Any uncommitted producer bytes are archived with their hashes; "
        "HEAD alone does not identify them.",
        "plan_document_sha256": digest(doc),
        "predecessor_queue": predecessor.as_posix(),
        "predecessor_queue_sha256": digest(predecessor),
        "predecessor_processes": processes,
        "required_separate_admission": "After predecessor completion and scientific review, "
        "local GPU_TECHNICAL_SMOKE_ONLY receipt",
        "technical_gate": "Finite 32-update repeated-batch learning, last8 mean loss below first8, "
        "CPU/CUDA parity, exact checkpoint replay on16 frames, far targets admitted",
        "budget": {
            "seconds": 600,
            "maximum_cuda_bytes": cfg.maximum_cuda_bytes,
            "updates": 32,
            "additional_cloud_spend": False,
        },
        "quality_screen_admitted": False,
        "promotion": False,
        "cuda_initialized_during_preparation": False,
    }
    for name in plan["sources"]:
        path = args.output / "code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    write_json(args.output / "PLAN.json", plan)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "config_sha256": cfg.sha256,
                "sources": len(files),
                "cuda_initialized": False,
            }
        )
    )


if __name__ == "__main__":
    main()
