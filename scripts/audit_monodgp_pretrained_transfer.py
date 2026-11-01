"""Exercise the pinned KITTI transfer on one real far eAP frame using CPU only."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smoke_monodgp_portable import load_detector_config, require_finite  # noqa: E402

from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset  # noqa: E402
from e_jepa_ttc.detection.long_range_campaign import LongRangeConfig  # noqa: E402
from e_jepa_ttc.detection.monodgp_eap_adapter import prepare_sample  # noqa: E402
from e_jepa_ttc.detection.pretrained_initialization import transfer_kitti_weights  # noqa: E402
from e_jepa_ttc.detection.range_campaign import derive_source, digest, write_json  # noqa: E402


def main() -> None:
    """Verify loading/finite loss/backward without competing for the active GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--conversion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Set CUDA_VISIBLE_DEVICES=-1 for this CPU-only audit")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = LongRangeConfig.load(args.plan_dir / "smoke300.json")
    plan = json.loads((args.plan_dir / "PLAN.json").read_text())
    receipt = json.loads(args.conversion.read_text())
    weight_path = args.conversion.parent / "model_state.pt"
    if receipt["status"] != "PASS" or digest(weight_path) != receipt["model_state_sha256"]:
        raise ValueError("Unverified pretrained weights")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    source = args.output / "portable_source"
    source_receipt = derive_source(cfg, source)
    relative = "lib/models/monodgp/depth_predictor/ddn_loss/ddn_loss.py"
    ddn = source / relative
    original = ddn.read_text()
    anchor = "self.device = torch.cuda.current_device()"
    if original.count(anchor) != 1 or original.count("self.device") != 1:
        raise ValueError("Expected exactly one unused CUDA device attribute")
    before = digest(ddn)
    ddn.write_text(original.replace(anchor, 'self.device = torch.device("cpu")'))
    source_receipt["cpu_only_patch"] = {
        "file": relative,
        "before_sha256": before,
        "after_sha256": digest(ddn),
        "change": "Avoid initializing CUDA for the otherwise unused self.device attribute",
    }
    source_receipt["files"][relative] = digest(ddn)
    sys.path.insert(0, str(source.resolve()))
    build = importlib.import_module("lib.models.monodgp").build

    model_config = load_detector_config(source, 11, predict_velocity=True, velocity_loss_coef=0.1)
    model_config["device"] = "cpu"
    model, criterion = build(model_config)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    policy = transfer_kitti_weights(model, state, seed=7)
    del state
    population = json.loads(Path(cfg.population).read_text())
    token = plan["smoke_far_target"]["token"]
    with EAPRGBDataset(cfg.dataset_root, [token], max_open_shards=1) as dataset:
        raw = dataset[0]
        if raw.member_sha256 != population["media"][token]["member_sha256"]:
            raise ValueError("Input differs from TRAIN media admission")
        sample = prepare_sample(raw, max_depth=300, include_velocity=True)
        label_free = prepare_sample(replace(raw, annotations=[]), max_depth=300)
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        outputs = model(
            label_free.image[None],
            label_free.calibration[None],
            None,
            label_free.image_size[None],
        )
        require_finite(outputs)
    inference_seconds = time.perf_counter() - start
    inference_shapes = {
        key: list(value.shape) for key, value in outputs.items() if isinstance(value, torch.Tensor)
    }
    del outputs
    model.train()
    criterion.train()
    start = time.perf_counter()
    outputs = model(
        sample.image[None], sample.calibration[None], [sample.target], sample.image_size[None]
    )
    require_finite(outputs)
    losses = criterion(outputs, [sample.target])
    require_finite(losses)
    terms = [
        value * criterion.weight_dict[key]
        for key, value in losses.items()
        if key in criterion.weight_dict
    ]
    if not terms:
        raise ValueError("No weighted losses")
    loss = torch.stack(terms).sum()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    if torch.cuda.is_initialized() or any(p.is_cuda for p in model.parameters()):
        raise ValueError("CPU-only audit unexpectedly initialized CUDA")
    write_json(
        args.output / "TRANSFER_AUDIT.json",
        {
            "schema": "phase_monodgp_kitti_transfer_cpu_audit_v1",
            "status": "PASS_CPU_ONLY",
            "conversion_sha256": digest(args.conversion),
            "transfer_policy": policy,
            "source_derivation": source_receipt,
            "effective_model": model_config,
            "frame": token,
            "frame_sha256": raw.member_sha256,
            "train_target_count": len(sample.native_ground_truth),
            "maximum_target_depth_m": float(sample.target["depth"].max()),
            "label_free_inference_shapes": inference_shapes,
            "label_free_inference_seconds": inference_seconds,
            "training_forward_backward_seconds": time.perf_counter() - start,
            "weighted_loss": float(loss.detach()),
            "gradient_norm": float(norm),
            "cuda_initialized": False,
            "mAP_measured": False,
            "next_gate": "GPU smoke and a registered, budget-matched learning comparison",
            "source_sha256": {
                "audit": digest(Path(__file__)),
                "transfer_policy": digest(
                    ROOT / "src/e_jepa_ttc/detection/pretrained_initialization.py"
                ),
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS_CPU_ONLY",
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "cuda_initialized": torch.cuda.is_initialized(),
            }
        )
    )


if __name__ == "__main__":
    main()
