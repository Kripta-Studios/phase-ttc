"""Audit pinned event-only Garl using generated inputs and zero optimizer updates."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding
from e_jepa_ttc.data.garl_official_preprocessing import official_timevolume_roi_np
from e_jepa_ttc.models.garl_ttc_replica import GarlTTCConfig, GarlTTCReplica

PIN = "256661242b8a7f5e56aa3c1c02348b30f6e89de6"


def main() -> None:
    """Check source identity and random-weight event-only adapter parity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release = args.release_root.resolve(strict=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=release, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=release, text=True
    ).strip()
    if commit != PIN or dirty:
        raise ValueError("Garl source must match clean pinned commit")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(release))
    config = yaml.safe_load((release / "configs/ablation/event_lhr.yaml").read_text())
    config["model"].pop("pretrained_ckpt_rgb", None)
    config["model"].pop("pretrained_ckpt_event", None)
    model_class = importlib.import_module("garl_ttc.models.ttc_network").TTCNetwork
    # This pure NumPy source does not need the dataset package's optional cv2
    # import. Load the already pinned file directly without constructing a loader.
    spec = importlib.util.spec_from_file_location(
        "stage70_official_event_representation",
        release / "garl_ttc/datasets/event_representation.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load pinned source event representation")
    representation_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(representation_module)
    source_representation = representation_module.get_timevolume_roi_np
    torch.set_num_threads(4)
    torch.manual_seed(7)
    official = model_class(config, is_train=True).eval()
    local = GarlTTCReplica(GarlTTCConfig(event_channels=40, foreground_supervision=False)).eval()
    mapped = {}
    for key, value in official.state_dict().items():
        if key.startswith("backbone."):
            target = "event_encoder." + key.removeprefix("backbone.")
        elif key.startswith("final_layer."):
            target = "height_head.height_regressor." + key.removeprefix("final_layer.")
        else:
            target = key
        mapped[target] = value
    local.load_state_dict(mapped, strict=True)
    data = torch.randn(2, 40, 128, 128)
    elapsed = torch.full((2,), 0.1)
    with torch.inference_mode():
        official_heights, _ = official(data)
        local_output = local(data, elapsed)
    assert local_output.predicted_heights is not None
    torch.testing.assert_close(official_heights, local_output.predicted_heights, rtol=0, atol=5e-5)
    raw_error = float((official_heights - local_output.predicted_heights).abs().max())
    source_ttc = elapsed / (1 - official_heights[:, 0] / official_heights[:, 1])
    torch.testing.assert_close(source_ttc, local_output.ttc_seconds, rtol=1e-5, atol=5e-5)
    preprocessing = []
    rng = np.random.default_rng(7)
    for offset in (0, 2**40):
        x = rng.integers(-3, 39, 1000)
        y = rng.integers(-2, 33, 1000)
        timestamps = np.sort(rng.integers(0, 110000, 1000, dtype=np.int64)) + offset
        source, source_counts = source_representation((0, 0, 37, 31), x, y, timestamps)
        own, own_counts = official_timevolume_roi_np((0, 0, 37, 31), x, y, timestamps)
        np.testing.assert_array_equal(source, own)
        np.testing.assert_array_equal(source_counts, own_counts)
        preprocessing.append({"absolute_clock_offset_us": offset, "max_abs_error": 0.0})
    names = (
        "configs/ablation/event_lhr.yaml",
        "configs/garl_ttc_eventdecoder.yaml",
        "garl_ttc/models/ttc_network.py",
        "garl_ttc/models/resnet.py",
        "garl_ttc/engine/trainer.py",
        "garl_ttc/datasets/event_representation.py",
        "garl_ttc/datasets/ttc_dataset.py",
        "garl_ttc_benchmark/build_garlttc_dataset.py",
    )
    result = {
        "status": "PASS_SYNTHETIC_SOURCE_PARITY_ONLY",
        "release_commit": commit,
        "source_files": [binding(release / name) for name in names],
        "optimizer_updates": 0,
        "pretrained_weights_loaded": False,
        "real_data_loaded": False,
        "generated_input_shape": list(data.shape),
        "raw_height_max_abs_error": raw_error,
        "ttc_max_abs_error": float((source_ttc - local_output.ttc_seconds).abs().max()),
        "official_parameters": sum(p.numel() for p in official.parameters()),
        "local_parameters": sum(p.numel() for p in local.parameters()),
        "preprocessing_parity": preprocessing,
        "config": config,
        "limitations": [
            "No decoder/foreground parity claimed by event_lhr configuration.",
            "No trained-checkpoint forward replay or real TRAIN preprocessing parity in this test.",
            "No scientific fit, GPU smoke, exact training resume, or S76 isolation proof.",
            "Audit interpreter is separate from the yet-to-be-validated scientific environment.",
        ],
    }
    atomic_json(args.output, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"config", "source_files"}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
