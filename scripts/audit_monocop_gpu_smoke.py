"""Recover the technical gate after a JSON representation-only replay failure."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_monocop_gpu_smoke import build_detector, evaluate, finite, seed_all

from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
from e_jepa_ttc.detection.monocop_smoke import (
    MonoCoPSmokeConfig,
    verify_inputs,
    verify_replay,
)
from e_jepa_ttc.detection.range_campaign import digest, write_json


def main() -> None:
    """Preserve the failed receipt; separately verify the exact final saved values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    if args.output.exists() or (args.run / "WRITER.lock").exists():
        raise ValueError("Require a completed writer and a new audit destination")
    contract = json.loads((args.run / "CONTRACT.json").read_text())
    cfg = MonoCoPSmokeConfig(**contract["config"])
    cfg.validate()
    population = verify_inputs(cfg)
    status = json.loads((args.run / "STATUS.json").read_text())
    if (
        status["status"] != "FAILED"
        or status["update"] != 32
        or status["error"] != "Saved-model predictions or metrics differ after restore"
    ):
        raise ValueError("Only the observed JSON replay failure is eligible")
    sources = {}
    for name, expected in contract["sources"].items():
        path = args.run / "code" / name
        if digest(path) != expected:
            raise ValueError(f"Executed source archive changed: {name}")
        sources[str(path)] = expected
    initial = json.loads((args.run / "evaluation/000032.json").read_text())
    previous_replay = json.loads((args.run / "REPLAY.json").read_text())
    verify_replay(previous_replay, initial)
    history = [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines()]
    if [row["update"] for row in history] != list(range(1, 33)):
        raise ValueError("Incomplete training history")
    early = sum(row["loss"] for row in history[:8]) / 8
    late = sum(row["loss"] for row in history[-8:]) / 8
    if late >= early:
        raise ValueError("Repeated-batch learning gate failed")
    for name, update in (("initial.pt", 0), ("update000016.pt", 16), ("last.pt", 32)):
        saved = torch.load(
            args.run / "checkpoints" / name, map_location="cpu", weights_only=True, mmap=True
        )
        if saved["contract"] != contract or saved["update"] != update:
            raise ValueError(f"Checkpoint lineage differs: {name}")
        finite(saved["model"])
    torch.set_num_threads(cfg.cpu_threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_all(cfg.seed)
    model, criterion, effective = build_detector(Path(cfg.portable_source))
    if effective != contract["effective_model"]:
        raise ValueError("Model configuration differs")
    model.load_state_dict(saved["model"], strict=True)
    probes = torch.tensor([60.0, 150.0, 249.153, 299.0])
    bins = criterion.ddn_loss.bin_depths(probes, target=True)
    if not ((bins >= 0) & (bins < 80)).all():
        raise ValueError("Valid depth mapped to background")
    args.output.mkdir(parents=True)
    for name in (
        "scripts/audit_monocop_gpu_smoke.py",
        "scripts/run_monocop_gpu_smoke.py",
        "src/e_jepa_ttc/detection/monocop_smoke.py",
    ):
        destination = args.output / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        sources[str(ROOT / name)] = digest(destination)
    torch.cuda.reset_peak_memory_stats()
    model.cuda()
    with EAPRGBDataset(cfg.dataset_root, population["roles"]["selection"]) as selection:
        replay = evaluate(model, selection, cfg, population["media"])
    write_json(args.output / "REPLAY.json", replay)
    verify_replay(replay, initial)
    elapsed = time.perf_counter() - start
    if elapsed > 60 or torch.cuda.max_memory_allocated() > cfg.maximum_cuda_bytes:
        raise RuntimeError("Recovery audit exceeded its 60s/9GiB technical budget")
    inputs = [
        "CONTRACT.json", "STATUS.json", "PARITY.json", "TIMINGS.json", "metrics.jsonl",
        "evaluation/000032.json", "REPLAY.json", "checkpoints/initial.pt",
        "checkpoints/update000016.pt", "checkpoints/last.pt",
    ]
    gate = {
        "schema": "phase_monocop_gpu_smoke_recovery_gate_v1",
        "status": "PASS_TECHNICAL_SMOKE_ONLY",
        "original_status": status,
        "explanation": "Original check compared tuple-bearing PROFILE against JSON lists. "
        "Both original serialized replays and this independent fresh-process replay are exact.",
        "updates": 32,
        "additional_training_updates": 0,
        "early_mean_loss": early,
        "late_mean_loss": late,
        "replay_exact_frames": len(replay["predictions"]),
        "depth_probe_bins": bins.tolist(),
        "checkpoint_sha256": digest(args.run / "checkpoints/last.pt"),
        "training_peak_bytes": None,
        "training_peak_note": "Original runner enforced 9GiB after every update, but did not "
        "persist the exact peak on its failed replay path.",
        "fresh_replay_peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": elapsed,
        "inputs": {str(args.run / name): digest(args.run / name) for name in inputs},
        "sources": sources,
        "replay_sha256": digest(args.output / "REPLAY.json"),
        "promotion": False,
        "quality_screen_admitted": False,
    }
    write_json(args.output / "GATE.json", gate)
    print(json.dumps({k: v for k, v in gate.items() if k not in ("inputs", "sources")}))


if __name__ == "__main__":
    main()
