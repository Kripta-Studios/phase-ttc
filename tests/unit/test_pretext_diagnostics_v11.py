"""Observation cannot perturb training randomness, modes or encoder tensors."""

import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from e_jepa_ttc.evaluation.pretext_diagnostics_v11 import (
    readonly_diagnostic,
    summarize_texture_trace,
)


@pytest.mark.parametrize("raises", [False, True])
def test_diagnostic_restores_randomness_and_individual_module_modes(raises):
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
    model.train()
    model[1].eval()
    generator = torch.Generator().manual_seed(17)
    before = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        generator.get_state(),
    )
    expected_weights = {key: value.clone() for key, value in model.state_dict().items()}

    def observe():
        with readonly_diagnostic(model, generator):
            assert not model.training and not model[0].training and not model[1].training
            random.random()
            np.random.rand(3)
            torch.rand(3)
            torch.rand(3, generator=generator)
            assert not model(torch.ones(1, 2)).requires_grad
            if raises:
                raise RuntimeError("injected observer interruption")

    if raises:
        with pytest.raises(RuntimeError, match="observer interruption"):
            observe()
    else:
        observe()
    assert random.getstate() == before[0]
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == before[1][0] and actual_numpy[2:] == before[1][2:]
    np.testing.assert_array_equal(actual_numpy[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
    assert torch.equal(generator.get_state(), before[3])
    assert model.training and model[0].training and not model[1].training
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in expected_weights.items()
    )


def test_observer_cannot_silently_modify_an_encoder():
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="mutated"):
        with readonly_diagnostic(model, torch.Generator()):
            model.weight.add_(1)


def test_texture_summary_keeps_train_and_held_controls_and_invalid_draw_counts(tmp_path):
    rows = []
    for held in (False, True):
        rows.append({"kind": "invalid_transform", "diagnostic": held})
        for i in range(2):
            rows.append(
                {
                    "kind": "valid_correspondence",
                    "diagnostic": held,
                    "token": f"r{i}",
                    "cross_entropy": 1.0 + i,
                    "matching_error_pixels": 2.0,
                    "constant_descriptor_ce": 7.0,
                    "shuffled_correspondence_ce": 8.0,
                    "descriptor_variance": 0.1,
                    "masked_fraction": 0.2,
                    "query_counts": [128, 64],
                }
            )
    path = tmp_path / "TRACE.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    result = summarize_texture_trace(path, expected_images=2)
    for scope in ("train_texture", "held_texture"):
        assert result[scope]["cross_entropy"] == 1.5
        assert result[scope]["invalid_transform_attempts"] == 1
        assert result[scope]["minimum_direction_queries"] == 64
    with pytest.raises(ValueError, match="omitted"):
        summarize_texture_trace(path, expected_images=3)
