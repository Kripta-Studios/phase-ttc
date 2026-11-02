"""Check the full random-initialized MonoCoP port in CPU, without optimizer steps."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import importlib
import json
import os
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
from prepare_monocop_portable import build_detector, digest

from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
from e_jepa_ttc.detection.monodgp_eap_adapter import decode_eap, prepare_sample
from e_jepa_ttc.detection.portable_attention import deformable_attention_pytorch


def finite(value: Any) -> None:
    """Reject nonfinite outputs, losses or gradients recursively."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite tensor in CPU admission")
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            finite(item)


def equal(first: Any, second: Any) -> None:
    """Check exact replay of every nested model output."""
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        if first.keys() != second.keys():
            raise ValueError("Replay output keys differ")
        for name in first:
            equal(first[name], second[name])
    elif isinstance(first, (tuple, list)):
        if len(first) != len(second):
            raise ValueError("Replay output lengths differ")
        for left, right in zip(first, second, strict=True):
            equal(left, right)


def fixture() -> tuple[torch.Tensor, torch.Tensor, list[dict], torch.Tensor]:
    """Create eight-class software inputs with focal lengths deliberately unequal."""
    height, width = 96, 160
    images = torch.rand(1, 3, height, width)
    depth = torch.tensor([8.0, 12.0, 20.0, 40.0, 60.0, 100.0, 180.0, 249.0])
    xy = torch.tensor([[0.15 + 0.23 * (i % 4), 0.4 + 0.2 * (i // 4)] for i in range(8)])
    hwl = torch.tensor([[1.5 + i * 0.1, 1.8, 4.0] for i in range(8)])
    wh = torch.stack((120.0 * hwl[:, 1] / depth / width, 190.0 * hwl[:, 0] / depth / height), 1)
    extents = torch.stack((wh[:, 0] / 2, wh[:, 0] / 2, wh[:, 1] / 2, wh[:, 1] / 2), 1)
    region = torch.zeros(height, width)
    for center, size in zip(xy, wh, strict=True):
        lo = torch.floor((center - size / 2) * torch.tensor([width, height])).long()
        hi = torch.ceil((center + size / 2) * torch.tensor([width, height])).long()
        region[max(0, lo[1]) : min(height, hi[1]), max(0, lo[0]) : min(width, hi[0])] = 1
    targets = [
        {
            "labels": torch.arange(8),
            "boxes": torch.cat((xy, wh), 1),
            "boxes_3d": torch.cat((xy, extents), 1),
            "depth": depth[:, None],
            "size_3d": hwl,
            "heading_bin": torch.arange(8)[:, None],
            "heading_res": torch.zeros(8, 1),
            "obj_region": region,
        }
    ]
    calibration = torch.tensor([[[120.0, 0, 80, 0], [0, 190.0, 48, 0], [0, 0, 1, 0]]])
    return images, calibration, targets, torch.tensor([[width, height]], dtype=torch.float32)


def attention_check() -> dict[str, Any]:
    """Compare independent accumulation orders and derivatives against upstream PyTorch."""
    module = importlib.import_module("lib.models.monocop.ops.functions.ms_deform_attn_func")
    reference = module.ms_deform_attn_core_pytorch
    shapes = torch.tensor([[2, 3], [1, 2]], dtype=torch.long)
    value = torch.randn(1, 8, 2, 3, dtype=torch.float64, requires_grad=True)
    locations = (torch.rand(1, 4, 2, 2, 3, 2, dtype=torch.float64) * 1.4 - 0.2).requires_grad_()
    weights = torch.randn(1, 4, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    ours = deformable_attention_pytorch(value, shapes, locations, weights)
    theirs = reference(value, shapes, locations, weights)
    torch.testing.assert_close(ours, theirs, atol=1e-12, rtol=1e-12)
    probe = torch.randn_like(ours)
    grad_ours = torch.autograd.grad((ours * probe).sum(), (value, locations, weights))
    grad_theirs = torch.autograd.grad((theirs * probe).sum(), (value, locations, weights))
    differences = []
    for left, right in zip(grad_ours, grad_theirs, strict=True):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
        differences.append(float((left - right).abs().max()))
    return {
        "reference": "MonoCoP ms_deform_attn_core_pytorch",
        "dtype": "float64",
        "forward_max_abs": float((ours - theirs).abs().max().detach()),
        "gradient_max_abs_value_locations_weights": differences,
        "compiled_cuda_extension_tested": False,
    }


def backward(model: Any, criterion: Any, inputs: tuple) -> dict[str, Any]:
    """Exercise supervised losses while passing no annotations into model.forward."""
    images, calibration, targets, sizes = inputs
    model.train()
    criterion.train()
    model.zero_grad(set_to_none=True)
    outputs = model(images, calibration, None, sizes)
    finite(outputs)
    losses = criterion(outputs, targets)
    finite(losses)
    total = sum(
        value * criterion.weight_dict[name]
        for name, value in losses.items()
        if name in criterion.weight_dict
    )
    if not isinstance(total, torch.Tensor):
        raise ValueError("No weighted tensor losses")
    total.backward()
    gradient_parameters = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            finite(parameter.grad)
            gradient_parameters += 1
    if gradient_parameters == 0:
        raise ValueError("No model gradients")
    result = {
        "loss": float(total.detach()),
        "loss_terms": {key: float(value.detach()) for key, value in losses.items()},
        "gradient_parameters": gradient_parameters,
        "logit_shape": list(outputs["pred_logits"].shape),
    }
    if result["logit_shape"] != [1, 550, 8]:
        raise ValueError("Full 11-group/eight-class training path was not exercised")
    model.zero_grad(set_to_none=True)
    return result


def run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    """Run bounded technical stages; accuracy and GPU training remain unmeasured."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or torch.cuda.is_initialized():
        raise ValueError("Run in a fresh process with CUDA_VISIBLE_DEVICES=-1")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(7)
    started = time.perf_counter()

    def stage(name: str, result: Any, start: float) -> None:
        elapsed = time.perf_counter() - started
        report["stages"][name] = {"seconds": time.perf_counter() - start, "result": result}
        (args.output / "STATUS.json").write_text(
            json.dumps({"status": "RUNNING", "stage": name, "elapsed_seconds": elapsed}) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"stage": name, "elapsed_seconds": elapsed}), flush=True)
        if elapsed > 600 or torch.cuda.is_initialized():
            raise RuntimeError("CPU admission exceeded time budget or initialized CUDA")

    step = time.perf_counter()
    with patch(
        "torch.hub.load_state_dict_from_url", side_effect=RuntimeError("Unexpected download")
    ):
        model, criterion, config = build_detector(args.source)
    report["configuration"] = config
    report["parameters"] = sum(p.numel() for p in model.parameters())
    report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stage("construction", {"depth_max": model.depth_predictor.depth_max}, step)
    step = time.perf_counter()
    stage("attention_reference", attention_check(), step)
    step = time.perf_counter()
    depths = torch.tensor([0.0, 0.001, 60, 150, 249.153, 299, 300, 301, float("nan"), float("inf")])
    bins = criterion.ddn_loss.bin_depths(depths, target=True)
    if bins.tolist() != [80, 0, 35, 56, 72, 79, 80, 80, 80, 80]:
        raise ValueError(f"Depth discretization differs: {bins.tolist()}")
    stage("depth_bins", {"bins": bins.tolist()}, step)
    inputs = fixture()
    step = time.perf_counter()
    stage("eight_class_backward", backward(model, criterion, inputs), step)
    images, calibration, targets, sizes = inputs
    empty = [
        {
            key: torch.zeros_like(value) if key == "obj_region" else value[:0]
            for key, value in targets[0].items()
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
        original = model(images, calibration, None, sizes)
        finite(original)
        fx = calibration.clone()
        fx[:, 0, 0] *= 1.5
        equal(original, model(images, fx, None, sizes))
        fy = calibration.clone()
        fy[:, 1, 1] *= 1.5
        changed = model(images, fy, None, sizes)
        height = (
            (original["pred_boxes"][:, :, 4] + original["pred_boxes"][:, :, 5]) * sizes[:, 1:2]
        ).clamp(min=1)
        expected = (
            original["pred_3d_dim"][:, :, 0]
            / height
            * (fy[:, 1, 1] - calibration[:, 1, 1])[:, None]
        )
        observed = changed["pred_depth"][:, :, 0] - original["pred_depth"][:, :, 0]
        torch.testing.assert_close(observed, expected, atol=1e-4, rtol=1e-5)
        for key in ("pred_logits", "pred_boxes", "pred_3d_dim", "pred_angle"):
            equal(original[key], changed[key])
        if original["pred_logits"].shape != (1, 50, 8):
            raise ValueError("Inference query/class shape differs")
        stage(
            "focal_contract",
            {
                "fy_max_abs_error": float((observed - expected).abs().max()),
                "fx_outputs_exact": True,
            },
            step,
        )
        step = time.perf_counter()
        checkpoint = args.output / "random_fixture_model.pt"
        torch.save({"model": model.state_dict(), "config": config}, checkpoint)
        model.class_embed[-1].weight.add_(1)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(saved["model"], strict=True)
        equal(original, model(images, calibration, None, sizes))
        stage("checkpoint_replay", {"sha256": digest(checkpoint), "exact": True}, step)
    step = time.perf_counter()
    plan = json.loads((args.range_plan / "PLAN.json").read_text())
    cfg_path = args.range_plan / "smoke300.json"
    if digest(cfg_path) != plan["smoke_config_sha256"]:
        raise ValueError("Registered real-frame config changed")
    cfg = json.loads(cfg_path.read_text())
    population_path = Path(cfg["population"])
    if digest(population_path) != cfg["population_sha256"]:
        raise ValueError("Registered real-frame population changed")
    population = json.loads(population_path.read_text())
    token = plan["smoke_far_target"]["token"]
    if token not in population["roles"]["train"]:
        raise ValueError("Real frame is not registered TRAIN")
    with EAPRGBDataset(cfg["dataset_root"], [token]) as data:
        raw = data[0]
        if raw.member_sha256 != population["media"][token]["member_sha256"]:
            raise ValueError("Real image changed")
        prepared = prepare_sample(raw, width=640, height=384, max_depth=300)
    maximum = float(prepared.target["depth"].max())
    if maximum <= 200:
        raise ValueError("Required far target not admitted")
    actual_inputs = (
        prepared.image[None],
        prepared.calibration[None],
        [prepared.target],
        prepared.image_size[None],
    )
    real = backward(model, criterion, actual_inputs)
    real.update(
        token=token,
        member_sha256=raw.member_sha256,
        maximum_target_depth_m=maximum,
        admitted_targets=len(prepared.target["labels"]),
    )
    model.eval()
    with torch.no_grad():
        outputs = model(
            prepared.image[None], prepared.calibration[None], None, prepared.image_size[None]
        )
        finite(outputs)
        boxes, counts = decode_eap(
            outputs, 0, prepared.calibration, raw.event_from_ego, prepared.image_size
        )
        real["decode_counts"] = counts
        real["decoded_boxes"] = len(boxes)
    stage("real_train_frame_backward", real, step)
    report.update(
        status="PASS_CPU_PORT_ONLY",
        elapsed_seconds=time.perf_counter() - started,
        cuda_initialized=torch.cuda.is_initialized(),
        optimizer_steps=0,
        eap_quality_measured=False,
        gpu_training_admitted=False,
    )


def main() -> None:
    """Preserve completed stages and failures in a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--range-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": "phase_ttc_monocop_cpu_port_admission_v1",
        "started_utc": datetime.now(UTC).isoformat(),
        "torch": torch.__version__,
        "runner_sha256": digest(Path(__file__)),
        "source_receipt_sha256": digest(args.source / "PORT.json"),
        "stages": {},
        "scope": "random_initialization_software_only",
        "cpu_threads": 2,
    }
    (args.output / "executed_runner.py").write_bytes(Path(__file__).read_bytes())
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
