"""Explicit KITTI-to-eAP model-only transfer; no optimizer or negative-only class reuse."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

POLICY = "kitti_car_to_eap_car_reset_other_classes_depth_velocity_v1"


def transfer_kitti_weights(
    model: torch.nn.Module, source: Mapping[str, torch.Tensor], *, seed: int
) -> dict[str, object]:
    """Transfer compatible geometry/features and Car row 1 to eAP car row 0.

    The published recipe is Car-only despite its three output rows. Pedestrian
    and Cyclist rows therefore do not seed eAP classes. New classes, velocity
    and the changed depth-bin classifier are initialized independently.
    """
    target = model.state_dict()
    source_keys = set(source)
    target_without_velocity = {k for k in target if not k.startswith("velocity_embed.")}
    if source_keys != target_without_velocity:
        raise ValueError("Unexpected pretrained/model keys")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Nonnegative integer seed required")
    classifiers = model.get_submodule("class_embed")
    if not isinstance(classifiers, torch.nn.ModuleList) or len(classifiers) != 4:
        raise ValueError("Expected four classification heads")
    for index, layer in enumerate(classifiers):
        if not isinstance(layer, torch.nn.Linear) or layer.out_features != 8:
            raise ValueError("Eight-class linear heads required")
        if source[f"class_embed.{index}.weight"].shape != (3, layer.in_features):
            raise ValueError("Expected three-row KITTI classifier")
        if source[f"class_embed.{index}.bias"].shape != (3,):
            raise ValueError("Expected three-row KITTI classifier bias")
    reset_keys = []
    copied = []
    with torch.no_grad():
        for key, value in target.items():
            if key.startswith(
                ("class_embed.", "velocity_embed.", "depth_predictor.depth_classifier.")
            ):
                reset_keys.append(key)
                continue
            if key == "depth_predictor.depth_bin_values":
                reset_keys.append(key)
                continue
            if key == "depth_predictor.depth_pos_embed.weight":
                old = source[key]
                if len(old) != 61 or value.shape[1:] != old.shape[1:] or len(value) < 61:
                    raise ValueError("Unexpected physical depth embedding layout")
                value[:61].copy_(old)
                if len(value) > 61:
                    value[61:].copy_(old[60].expand_as(value[61:]))
            else:
                if source[key].shape != value.shape:
                    raise ValueError(f"Pretrained shape differs: {key}")
                value.copy_(source[key])
            copied.append(key)
        model.load_state_dict(target, strict=True)
        devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 91001)
            for index, layer in enumerate(classifiers):
                assert isinstance(layer, torch.nn.Linear)
                layer.reset_parameters()
                layer.bias.fill_(-math.log(99))
                layer.weight[0].copy_(source[f"class_embed.{index}.weight"][1])
                layer.bias[0].copy_(source[f"class_embed.{index}.bias"][1])
            torch.manual_seed(seed + 73001)
            depth = model.get_submodule("depth_predictor.depth_classifier")
            if not isinstance(depth, torch.nn.Conv2d):
                raise ValueError("Expected depth classifier convolution")
            depth.reset_parameters()
            torch.manual_seed(seed + 92001)
            for layer in model.get_submodule("velocity_embed").modules():
                if isinstance(layer, torch.nn.Linear):
                    layer.reset_parameters()
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("Nonfinite transferred model")
    return {
        "policy": POLICY,
        "copied_tensor_keys": copied,
        "reset_tensor_keys": reset_keys,
        "class_mapping": {"KITTI_Car_row_1": "eAP_car_row_0"},
        "new_class_rows": list(range(1, 8)),
        "position_embedding": "Preserve physical rows 0..60; repeat row60 above60",
        "depth_bins": "Keep destination physical bins; reset their classifier",
        "optimizer_scheduler_sampler_rng": "No upstream training state transferred",
        "semantic_limits": [
            "Published KITTI configuration used; checkpoint lacks its original config hash.",
            "KITTI native 2D boxes differ from eAP projected 3D boxes.",
            "KITTI alpha uses a 2D-box-center ray; eAP uses the projected 3D-center ray.",
            "Shared tensor shape alone does not establish performance on eAP.",
        ],
    }
