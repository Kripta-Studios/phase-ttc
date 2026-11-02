"""Explicit model-only transfer from three KITTI classes to eight eAP classes."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

POLICY = "monocop_all_kitti_to_eap_eight_300m_v1"
CLASS_ROWS = ((0, 3), (1, 0), (2, 1))
NEW_ROWS = (2, 4, 5, 6, 7)


def transfer_monocop_weights(
    model: nn.Module, source: Mapping[str, torch.Tensor], *, seed: int
) -> dict[str, object]:
    """Transfer verified source tensors on CPU with explicit semantic exceptions.

    The caller must verify checkpoint provenance. Cyclist-to-bike is a transfer
    hypothesis between dataset taxonomies, not an assertion of identical labels.
    No optimizer state is accepted. Initialization preserves the caller's RNG.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("A nonnegative integer seed is required")
    target = model.state_dict()
    if len(target) != 694 or target.keys() != source.keys():
        raise ValueError("Unexpected MonoCoP checkpoint/model keys")
    if any(v.device.type != "cpu" for v in (*target.values(), *source.values())):
        raise ValueError("Transfer on CPU before moving the admitted model to a device")
    classifiers = model.get_submodule("class_embed")
    if not isinstance(classifiers, nn.ModuleList) or len(classifiers) != 3:
        raise ValueError("Expected three decoder classification heads")
    if any(not isinstance(layer, nn.Linear) or layer.out_features != 8 for layer in classifiers):
        raise ValueError("Expected eight-class linear heads")
    aliases: dict[tuple, list[str]] = {}
    for key, value in target.items():
        identity = (value.data_ptr(), tuple(value.shape), tuple(value.stride()))
        aliases.setdefault(identity, []).append(key)
    alias_groups = [names for names in aliases.values() if len(names) > 1]
    for names in alias_groups:
        if not all(torch.equal(source[names[0]], source[name]) for name in names[1:]):
            raise ValueError(f"Inconsistent source values for shared model tensors: {names}")
    class_keys = {
        f"class_embed.{index}.{part}" for index in range(3) for part in ("weight", "bias")
    }
    bins = "depth_predictor.depth_bin_values"
    position = "depth_predictor.depth_pos_embed.weight"
    label = "label_enc.weight"
    depth_keys = {f"depth_predictor.depth_classifier.{part}" for part in ("weight", "bias")}
    special = class_keys | depth_keys | {bins, position, label}
    for name, value in source.items():
        if type(value) is not torch.Tensor or not torch.isfinite(value).all():
            raise ValueError(f"Source is not a finite plain tensor: {name}")
        if value.dtype != target[name].dtype:
            raise ValueError(f"Tensor dtype differs: {name}")
        if name not in {position, label, *class_keys} and value.shape != target[name].shape:
            raise ValueError(f"Unexpected compatible tensor shape: {name}")
    if source[position].shape != (61, 256) or target[position].shape != (301, 256):
        raise ValueError("Expected physical depth embedding rows 0..60 and 0..300")
    if source[label].shape != (4, 255) or target[label].shape != (9, 255):
        raise ValueError("Expected three/eight class embeddings plus the background row")
    if source[bins].shape != (81,) or source[bins][-1] != 60 or target[bins][-1] != 300:
        raise ValueError("Unexpected physical depth bins")
    if not (torch.diff(source[bins]) > 0).all() or not (torch.diff(target[bins]) > 0).all():
        raise ValueError("Physical depth bins must increase")
    for index in range(3):
        if source[f"class_embed.{index}.weight"].shape != (3, 256):
            raise ValueError("Expected three-row source class weights")
        if source[f"class_embed.{index}.bias"].shape != (3,):
            raise ValueError("Expected three-row source class biases")
    # Build a separate state first; shape/provenance failures cannot partially load the model.
    state = {key: value.detach().clone() for key, value in target.items()}
    copied = sorted(set(state) - special)
    for name in copied:
        state[name].copy_(source[name])
    state[position][:61].copy_(source[position])
    state[position][61:].copy_(source[position][60].expand_as(state[position][61:]))
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.random.default_generator.manual_seed(seed + 91001)
        for index in range(3):
            layer = nn.Linear(256, 8, device="cpu")
            layer.bias.fill_(-math.log(99))
            for old, new in CLASS_ROWS:
                layer.weight[new].copy_(source[f"class_embed.{index}.weight"][old])
                layer.bias[new].copy_(source[f"class_embed.{index}.bias"][old])
            state[f"class_embed.{index}.weight"].copy_(layer.weight)
            state[f"class_embed.{index}.bias"].copy_(layer.bias)
        torch.random.default_generator.manual_seed(seed + 93001)
        label_layer = nn.Embedding(9, 255, device="cpu")
        for old, new in (*CLASS_ROWS, (3, 8)):
            label_layer.weight[new].copy_(source[label][old])
        state[label].copy_(label_layer.weight)
        torch.random.default_generator.manual_seed(seed + 73001)
        depth_layer = nn.Conv2d(256, 81, kernel_size=1, device="cpu")
        state["depth_predictor.depth_classifier.weight"].copy_(depth_layer.weight)
        state["depth_predictor.depth_classifier.bias"].copy_(depth_layer.bias)
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Nonfinite destination tensor")
    model.load_state_dict(state, strict=True)
    loaded = model.state_dict()
    if any(not torch.equal(v, loaded[key]) for key, v in state.items()):
        raise ValueError("Model state differs after strict load")
    return {
        "policy": POLICY,
        "seed": seed,
        "source_class_order": ["Pedestrian", "Car", "Cyclist"],
        "source_to_destination_rows": [list(pair) for pair in CLASS_ROWS],
        "reset_class_rows": list(NEW_ROWS),
        "copied_tensor_keys": copied,
        "adapted_tensor_keys": sorted(special),
        "shared_tensor_aliases_verified": alias_groups,
        "depth_position": "Copy physical rows 0..60; repeat row 60 above 60m",
        "depth_classifier": "Reset for changed LID bins; keep destination bin values",
        "label_embedding": "Map classes and background; unused with use_dn=False",
        "upstream_optimizer_or_rng_transferred": False,
        "velocity_head": False,
        "quality_established": False,
    }
