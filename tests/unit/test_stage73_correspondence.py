"""Known-transform mathematics and label-free encoder gradients."""

import math

import torch
from torch import nn

from e_jepa_ttc.training.stage73_correspondence import (
    TexturePair,
    bilinear_labels,
    correspondence_loss,
    texture_pair,
)


def test_four_neighbor_labels_preserve_subpixel_probability():
    indices, mass, valid = bilinear_labels(torch.tensor([[0.0, 0.0, 1.0]]), 4, 4)
    assert indices.tolist() == [[5, 6, 9, 10]]
    torch.testing.assert_close(mass, torch.full((1, 4), 0.25))
    assert valid.tolist() == [True]
    _, _, valid = bilinear_labels(torch.tensor([[1.0, 0.0, 1.0]]), 4, 4)
    assert not valid.any()


def test_texture_augmentation_replays_generator_state_without_labels():
    texture = torch.ones(8, 128, 128)
    mask = torch.ones(128, 128, dtype=torch.bool)
    generator = torch.Generator().manual_seed(7)
    state = generator.get_state()
    pair = texture_pair(texture, mask, generator)
    generator.set_state(state)
    resumed = texture_pair(texture, mask, generator)
    for key in pair.__dict__:
        assert torch.equal(getattr(pair, key), getattr(resumed, key))
    assert not torch.equal(pair.source, pair.target)
    scale = float(torch.linalg.det(pair.source_to_target[:2, :2]).sqrt())
    assert math.exp(-0.15) <= scale <= math.exp(0.15)
    assert bool((pair.source_to_target[:2, 2].abs() <= 0.25).all())


def identity_pair(texture, mask=None):
    if mask is None:
        mask = torch.ones(128, 128, dtype=torch.bool)
    return TexturePair(texture, texture.clone(), mask, mask.clone(), torch.eye(3))


def test_known_identity_correspondence_beats_shuffled_and_has_encoder_gradients():
    torch.manual_seed(7)
    encoder = nn.Conv2d(8, 16, 4, stride=4, bias=False)
    result = correspondence_loss(
        encoder, identity_pair(torch.randn(8, 128, 128)), torch.Generator().manual_seed(7)
    )
    assert result is not None
    assert result.query_counts == (128, 128)
    assert result.matching_error_pixels == 0
    assert float(result.loss.detach()) < result.shuffled_correspondence_ce
    assert result.descriptor_variance > 0
    result.loss.backward()
    assert encoder.weight.grad is not None
    assert torch.isfinite(encoder.weight.grad).all() and encoder.weight.grad.abs().sum() > 0


def test_constant_descriptor_and_invalid_pair_are_explicit():
    encoder = nn.Conv2d(8, 16, 4, stride=4, bias=False)
    generator = torch.Generator().manual_seed(7)
    result = correspondence_loss(encoder, identity_pair(torch.ones(8, 128, 128)), generator)
    assert result is not None
    assert abs(float(result.loss.detach()) - math.log(1024)) < 1e-5
    assert abs(result.descriptor_variance) < 1e-10
    invalid = identity_pair(torch.ones(8, 128, 128), torch.zeros(128, 128, dtype=torch.bool))
    assert correspondence_loss(encoder, invalid, generator) is None
