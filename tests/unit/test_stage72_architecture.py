"""Production supervised transforms, matched sampling and encoder-only task initialization."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from e_jepa_ttc.data.event_context128 import COUNT_SHAPE, RateNormalization, four_contexts
from e_jepa_ttc.data.event_context128_reader import CountObservation, TrainingView
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.models.phase_distribution_v11 import phase_edges
from e_jepa_ttc.training.stage72_architecture import SupervisedBatches, initialize_architecture
from e_jepa_ttc.training.stage72_augmentation import (
    DERANGEMENTS,
    AugmentationParameters,
    augment_observation,
    augmentation_generator,
    spatial_transform,
    time_derangement,
)
from e_jepa_ttc.training.stage72_supervision import Supervision


@pytest.fixture
def row():
    counts = np.zeros(COUNT_SHAPE, dtype=np.uint16)
    counts[..., 50:70, 40:60] = 17
    return CountObservation(
        counts,
        np.full((2, 8), 0.01),
        np.ones((128, 128), dtype=bool),
        np.array([-350000, -250000, -150000, -50000]),
    )


def norm(identity="a" * 64):
    return RateNormalization((0.2, 0.3), (1.0, 2.0), identity, 100)


def supervision(tokens):
    return Supervision(
        tokens,
        torch.full((len(tokens),), 0.02),
        torch.full((len(tokens),), 1 / len(tokens), dtype=torch.float64),
        torch.full((108,), 1 / 108, dtype=torch.float64),
        0.0,
        "b" * 64,
    )


def test_spatial_transform_has_no_wrap_and_preserves_clock_and_context_identity(row):
    parameters = AugmentationParameters(True, 4, -3, 1.0, (1.0, 1.0))
    baseline = four_contexts(
        row.counts, row.durations_s, row.support, row.midpoint_twice_relative_us, norm()
    )
    actual, _ = augment_observation(
        row, norm(), torch.Generator().manual_seed(3), parameters=parameters
    )
    torch.testing.assert_close(
        actual.contexts, spatial_transform(baseline.contexts, parameters), rtol=0, atol=0
    )
    assert torch.equal(actual.times, baseline.times)
    assert not actual.sensor_support[-3:].any()
    assert not actual.sensor_support[:, :4].any()
    assert not actual.contexts[..., -3:, :].any()
    assert np.array_equal(row.counts[..., 50:70, 40:60], np.full((2, 8, 2, 20, 20), 17))


def test_thinning_gain_resume_is_exact_and_never_changes_time(row):
    generator = augmentation_generator(7, 4, 9, "token")
    state = generator.get_state()
    first, p1 = augment_observation(row, norm(), generator)
    generator.set_state(state)
    second, p2 = augment_observation(row, norm(), generator)
    assert p1 == p2
    assert torch.equal(first.contexts, second.contexts)
    assert torch.equal(first.times, second.times)
    assert torch.equal(first.sensor_support, second.sensor_support)
    parameters = AugmentationParameters(False, 0, 0, 0.8, (0.9, 1.1))
    output, _ = augment_observation(row, norm(), generator, parameters=parameters)
    values = output.contexts.reshape(COUNT_SHAPE).numpy()
    means = np.array(norm().mean, dtype=np.float32)[None, None, :, None, None]
    stds = np.array(norm().std, dtype=np.float32)[None, None, :, None, None]
    gains = np.array(parameters.polarity_gain)[None, None, :, None, None]
    recovered = np.expm1(values * stds + means) * 0.01 / gains
    assert recovered.max() <= 17.00002
    assert abs(recovered[..., 50:70, 40:60].mean() / 17 - 0.8) < 0.02


def test_timeperm_is_fixed_complete_and_independent_of_sampling_order():
    assert len(DERANGEMENTS) == 9
    tokens = [f"row{i}" for i in range(100)]
    first = {token: time_derangement(token, 7) for token in tokens}
    second = {token: time_derangement(token, 7) for token in reversed(tokens)}
    assert first == second
    assert set(first.values()) == set(DERANGEMENTS)
    assert first != {token: time_derangement(token, 13) for token in tokens}
    assert all(
        sorted(perm) == [0, 1, 2, 3] and all(i != v for i, v in enumerate(perm))
        for perm in first.values()
    )


def test_pretext_transfer_resets_every_task_head_from_registered_seed():
    torch.set_num_threads(2)
    prior = supervision(("a",))
    scratch = initialize_architecture(7, prior)
    pretrained = {key: value + 0.25 for key, value in scratch.encoder.state_dict().items()}
    transferred = initialize_architecture(7, prior, encoder_state=pretrained)
    for key, value in transferred.state_dict().items():
        if key.startswith("encoder."):
            assert torch.equal(value, pretrained[key[len("encoder.") :]])
        else:
            assert torch.equal(value, scratch.state_dict()[key])
    with pytest.raises(ValueError, match="exactly"):
        initialize_architecture(7, prior, encoder_state=scratch.state_dict())


class InputRecorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(108))
        self.edges = phase_edges()
        self.seen = []

    def forward(self, contexts, times, *, arm, permutation):
        self.seen.extend((c.clone(), t.clone()) for c, t in zip(contexts, times, strict=True))
        return {
            "logits": self.parameter[None].expand(len(contexts), -1),
            "ttc": torch.ones(len(contexts)),
        }


def test_sampler_and_augmentation_are_microbatch_invariant_and_target_isolated(row, tmp_path):
    class Reader:
        identity = "c" * 64
        metadata = pd.DataFrame(dict(sample_token=["a", "b", "c"], sequence_id=["train"] * 3))

        def observation(self, token):
            return row

    roles = TrainRoles((("train", "train"),), ("train",), ("train",), (), (), (), "fixture")
    view = TrainingView(
        Reader(), ("a", "b", "c"), roles, excluded_sequences=(), include_expansion=False
    )
    models, records = [], []
    for batch in (1, 2):
        model = InputRecorder()
        labels = supervision(view.tokens)
        if batch == 2:
            labels = replace(labels, target_phase=labels.target_phase * -5)
        path = tmp_path / f"batch{batch}.jsonl"
        callback = SupervisedBatches(model, view, norm(view.identity), labels, "DIRECT", 7, path)
        generator = torch.Generator().manual_seed(7)
        for micro in range(16 // batch):
            assert torch.isfinite(callback(1, micro, batch, generator))
        records.append(
            [item for line in path.read_text().splitlines() for item in json.loads(line)["rows"]]
        )
        models.append(model)
    assert records[0] == records[1]
    for first, second in zip(models[0].seen, models[1].seen, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
