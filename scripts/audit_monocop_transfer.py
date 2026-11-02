"""Verify the full MonoCoP pretrained transfer on CPU without optimizer steps."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_monocop_cpu_port import backward, equal, finite, fixture
from prepare_monocop_portable import build_detector, digest

from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
from e_jepa_ttc.detection.monocop_initialization import CLASS_ROWS, transfer_monocop_weights
from e_jepa_ttc.detection.monodgp_eap_adapter import decode_eap, prepare_sample

PURE_SHA256 = "344222be2ed6413294593986bba84fc3d4bf17548dd0688d6022c3535d7d7c06"
PORT_SHA256 = "9ba3647ece606feb68e9aea9332b6a6a5745db4788c5ec2f7271e37996f2deec"
FIXTURE_SHA256 = "8ffa487c6c81747fb11f3744ef80e06237d3973a32b80fcc4b1b25e8d5dad923"


def run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    """Check mapping, reproducibility, eight-class learning paths and real TRAIN geometry."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or torch.cuda.is_initialized():
        raise ValueError("Use a fresh CPU-only process with CUDA_VISIBLE_DEVICES=-1")
    if digest(args.source / "PORT.json") != PORT_SHA256:
        raise ValueError("Only the admitted MonoCoP port v2 is covered")
    if digest(ROOT / "scripts/audit_monocop_cpu_port.py") != FIXTURE_SHA256:
        raise ValueError("Previously verified test helper changed")
    pure = args.weights / "model_state.pt"
    conversion = json.loads((args.weights / "CONVERSION.json").read_text())
    if digest(pure) != PURE_SHA256 or conversion["model_state_sha256"] != PURE_SHA256:
        raise ValueError("Converted weights changed")
    if conversion["source_sha256"] != digest(args.weights / "executed_converter.py"):
        raise ValueError("Executed conversion source changed")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_rng_state(torch.Generator(device="cpu").manual_seed(7).get_state())
    started = time.perf_counter()

    def stage(name: str, result: Any, start: float) -> None:
        elapsed = time.perf_counter() - started
        report["stages"][name] = {"seconds": time.perf_counter() - start, "result": result}
        (args.output / "STATUS.json").write_text(
            json.dumps({"status": "RUNNING", "stage": name, "elapsed_seconds": elapsed}) + "\n"
        )
        print(json.dumps({"stage": name, "elapsed_seconds": elapsed}), flush=True)
        if elapsed > 600 or torch.cuda.is_initialized():
            raise RuntimeError("CPU transfer exceeded 600s or initialized CUDA")

    step = time.perf_counter()
    with patch(
        "torch.hub.load_state_dict_from_url", side_effect=RuntimeError("Unexpected download")
    ):
        model, criterion, config = build_detector(args.source)
    report["configuration"] = config
    stage("construction", {"parameters": sum(p.numel() for p in model.parameters())}, step)
    step = time.perf_counter()
    source = torch.load(pure, map_location="cpu", weights_only=True, mmap=True)
    original_bins = model.state_dict()["depth_predictor.depth_bin_values"].clone()
    rng = torch.get_rng_state().clone()
    transfer = transfer_monocop_weights(model, source, seed=7)
    equal(rng, torch.get_rng_state())
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    for index in range(3):
        for part in ("weight", "bias"):
            for old, new in CLASS_ROWS:
                equal(
                    initial[f"class_embed.{index}.{part}"][new],
                    source[f"class_embed.{index}.{part}"][old],
                )
    equal(initial["depth_predictor.depth_bin_values"], original_bins)
    equal(
        initial["depth_predictor.depth_pos_embed.weight"][:61],
        source["depth_predictor.depth_pos_embed.weight"],
    )
    equal(
        initial["depth_predictor.depth_pos_embed.weight"][61:],
        source["depth_predictor.depth_pos_embed.weight"][60].expand(240, 256),
    )
    for old, new in (*CLASS_ROWS, (3, 8)):
        equal(initial["label_enc.weight"][new], source["label_enc.weight"][old])
    torch.rand(39)
    rng = torch.get_rng_state().clone()
    transfer_monocop_weights(model, source, seed=7)
    equal(initial, model.state_dict())
    equal(rng, torch.get_rng_state())
    stage(
        "transfer_contract",
        {**transfer, "global_cpu_rng_preserved": True, "repeat_exact": True},
        step,
    )
    # A separate initialized checkpoint is saved before any forward/backward.
    checkpoint = args.output / "initialized_model.pt"
    temporary = args.output / ".initialized_model.tmp"
    torch.save({"model": initial, "configuration": config, "transfer": transfer}, temporary)
    os.replace(temporary, checkpoint)
    step = time.perf_counter()
    inputs = fixture()
    stage("eight_class_backward", backward(model, criterion, inputs), step)
    images, calibration, targets, sizes = inputs
    empty = [
        {
            name: torch.zeros_like(value) if name == "obj_region" else value[:0]
            for name, value in targets[0].items()
        }
    ]
    step = time.perf_counter()
    stage(
        "empty_target_backward",
        backward(model, criterion, (images, calibration, empty, sizes)),
        step,
    )
    step = time.perf_counter()
    model.eval()
    with torch.no_grad():
        before = model(images, calibration, None, sizes)
        finite(before)
        model.class_embed[-1].weight.add_(1)
        restored = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(restored["model"], strict=True)
        equal(before, model(images, calibration, None, sizes))
    stage("checkpoint_replay", {"sha256": digest(checkpoint), "outputs_exact": True}, step)
    step = time.perf_counter()
    plan = json.loads((args.range_plan / "PLAN.json").read_text())
    cfg_path = args.range_plan / "smoke300.json"
    if digest(cfg_path) != plan["smoke_config_sha256"]:
        raise ValueError("Registered smoke configuration changed")
    cfg = json.loads(cfg_path.read_text())
    population_path = Path(cfg["population"])
    if digest(population_path) != cfg["population_sha256"]:
        raise ValueError("Registered TRAIN population changed")
    population = json.loads(population_path.read_text())
    token = plan["smoke_far_target"]["token"]
    if token not in population["roles"]["train"]:
        raise ValueError("Expected a registered TRAIN frame")
    with EAPRGBDataset(cfg["dataset_root"], [token]) as data:
        raw = data[0]
        if raw.member_sha256 != population["media"][token]["member_sha256"]:
            raise ValueError("Registered TRAIN image changed")
        prepared = prepare_sample(raw, width=640, height=384, max_depth=300)
    maximum = float(prepared.target["depth"].max())
    if maximum <= 200:
        raise ValueError("Required far target is missing")
    actual = (
        prepared.image[None],
        prepared.calibration[None],
        [prepared.target],
        prepared.image_size[None],
    )
    real = backward(model, criterion, actual)
    real.update(
        token=token,
        member_sha256=raw.member_sha256,
        admitted_targets=len(prepared.target["labels"]),
        maximum_target_depth_m=maximum,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(actual[0], actual[1], None, actual[3])
        finite(outputs)
        boxes, counts = decode_eap(
            outputs, 0, prepared.calibration, raw.event_from_ego, prepared.image_size
        )
        real.update(decoded_boxes=len(boxes), decode_counts=counts)
    equal(initial, model.state_dict())
    stage("real_train_frame_backward", real, step)
    report.update(
        status="PASS_CPU_TRANSFER_ONLY",
        elapsed_seconds=time.perf_counter() - started,
        cuda_initialized=False,
        optimizer_steps=0,
        model_parameters_and_buffers_unchanged_by_checks=True,
        quality_evaluated=False,
        gpu_training_admitted=False,
    )


def main() -> None:
    """Archive the executed producer and preserve technical failures in a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--range-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = [
        Path(__file__),
        ROOT / "scripts/prepare_monocop_portable.py",
        ROOT / "scripts/audit_monocop_cpu_port.py",
        ROOT / "src/e_jepa_ttc/detection/monocop_initialization.py",
        ROOT / "src/e_jepa_ttc/detection/monodgp_eap_adapter.py",
        ROOT / "src/e_jepa_ttc/detection/eap_rgb_data.py",
        ROOT / "src/e_jepa_ttc/detection/geometry_contract.py",
        ROOT / "src/e_jepa_ttc/detection/portable_attention.py",
    ]
    for path in files:
        shutil.copyfile(path, args.output / ("executed_" + path.name))
    report: dict[str, Any] = {
        "schema": "phase_monocop_cpu_transfer_admission_v1",
        "started_utc": datetime.now(UTC).isoformat(),
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "cpu_threads": 2,
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in files},
        "port_receipt_sha256": digest(args.source / "PORT.json"),
        "conversion_sha256": digest(args.weights / "CONVERSION.json"),
        "plan_sha256": digest(
            ROOT / "docs/detection_sota_20261029/MONOCOP_TRANSFER_PLAN_20261101.md"
        ),
        "stages": {},
    }
    try:
        run(args, report)
    except Exception:
        report.update(
            status="FAILED",
            traceback=traceback.format_exc(),
            cuda_initialized=torch.cuda.is_initialized(),
        )
        raise
    finally:
        (args.output / "RESULTS.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        (args.output / "STATUS.json").write_text(
            json.dumps({"status": report["status"]}) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
