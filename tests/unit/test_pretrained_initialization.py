"""Catch class-order, range-bin and unsupported-shape transfer errors."""

import pytest
import torch
from torch import nn

from e_jepa_ttc.detection.pretrained_initialization import transfer_kitti_weights


def models():
    def make(classes, rows, velocity):
        model = nn.Module()
        model.class_embed = nn.ModuleList([nn.Linear(4, classes) for _ in range(4)])
        model.features = nn.Linear(4, 4)
        model.depth_predictor = nn.Module()
        model.depth_predictor.depth_pos_embed = nn.Embedding(rows, 4)
        model.depth_predictor.depth_classifier = nn.Conv2d(4, 81, 1)
        model.depth_predictor.register_buffer("depth_bin_values", torch.linspace(0, rows - 1, 80))
        if velocity:
            model.velocity_embed = nn.ModuleList([nn.Sequential(nn.Linear(4, 2)) for _ in range(3)])
        return model

    return make(3, 61, False), make(8, 301, True)


def test_transfer_preserves_physics_and_copies_only_the_trained_car_row():
    source, target = models()
    bins = target.depth_predictor.depth_bin_values.clone()
    source_state = source.state_dict()
    report = transfer_kitti_weights(target, source_state, seed=7)
    assert report["class_mapping"] == {"KITTI_Car_row_1": "eAP_car_row_0"}
    assert torch.equal(target.features.weight, source.features.weight)
    assert torch.equal(target.depth_predictor.depth_bin_values, bins)
    assert torch.equal(
        target.depth_predictor.depth_pos_embed.weight[:61],
        source.depth_predictor.depth_pos_embed.weight,
    )
    assert torch.equal(
        target.depth_predictor.depth_pos_embed.weight[300],
        source.depth_predictor.depth_pos_embed.weight[60],
    )
    for a, b in zip(source.class_embed, target.class_embed, strict=True):
        assert torch.equal(a.weight[1], b.weight[0])
        assert torch.equal(a.bias[1], b.bias[0])
        assert not torch.equal(a.weight[0], b.weight[1])
    _, second = models()
    transfer_kitti_weights(second, source_state, seed=7)
    assert all(
        torch.equal(value, second.state_dict()[key]) for key, value in target.state_dict().items()
    )


def test_transfer_rejects_unknown_keys_and_geometry_shape_changes():
    source, target = models()
    state = source.state_dict()
    with pytest.raises(ValueError, match="keys"):
        transfer_kitti_weights(target, state | {"extra": torch.zeros(1)}, seed=7)
    with pytest.raises(ValueError, match="shape"):
        transfer_kitti_weights(target, state | {"features.weight": torch.zeros(3, 4)}, seed=7)
