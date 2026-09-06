"""Target-free partitioning and bounded redraw/replenishment lifecycle."""

import json
from types import SimpleNamespace

import pytest
import torch

from e_jepa_ttc.data.event_context128 import RateNormalization
from e_jepa_ttc.training import correspondence_pretext_v11 as pretext


def test_held_token_membership_is_hash_fixed_and_disjoint():
    tokens = tuple(str(i) for i in range(100))
    train, held = pretext.texture_split(tokens)
    assert len(train) == 90 and len(held) == 10 and not set(train) & set(held)
    reverse_train, reverse_held = pretext.texture_split(tokens[::-1])
    assert set(train) == set(reverse_train) and set(held) == set(reverse_held)
    with pytest.raises(ValueError, match="unique"):
        pretext.texture_split(("duplicate",) * 20)


def test_three_redraws_then_replenishment_does_not_create_fake_update(tmp_path, monkeypatch):
    read_tokens = []

    def inputs(token, normalization):
        del normalization
        read_tokens.append(token)
        return SimpleNamespace(
            contexts=torch.zeros(4, 8, 128, 128),
            sensor_support=torch.ones(128, 128, dtype=torch.bool),
        )

    view = SimpleNamespace(
        identity="a" * 64, tokens=tuple(str(i) for i in range(20)), inputs=inputs
    )
    normalization = RateNormalization((0.0, 0.0), (1.0, 1.0), view.identity, 1)
    pool = pretext.PretextTextures(view, normalization, tmp_path / "trace.jsonl")
    calls = 0

    def objective(*args):
        nonlocal calls
        calls += 1
        if calls <= 4:
            return None
        return SimpleNamespace(
            loss=torch.tensor(2.0, requires_grad=True),
            matching_error_pixels=0.0,
            constant_descriptor_ce=7.0,
            shuffled_correspondence_ce=7.0,
            descriptor_variance=0.1,
            masked_fraction=0.0,
            query_counts=(128, 128),
        )

    monkeypatch.setattr(pretext, "texture_pair", lambda *args: None)
    monkeypatch.setattr(pretext, "correspondence_loss", objective)
    loss = pool.objective(
        torch.nn.Linear(1, 1),
        update=1,
        microbatch_index=0,
        batch=1,
        generator=torch.Generator().manual_seed(7),
    )
    assert float(loss.detach()) == 2 and calls == 5 and len(read_tokens) == 2
    assert not set(read_tokens) & set(pool.held_tokens)
    events = [json.loads(line) for line in pool.journal.read_text().splitlines()]
    invalid = [event for event in events if event["kind"] == "invalid_transform"]
    assert [event["redraw"] for event in invalid] == [0, 1, 2, 3]
    assert all(not event["optimizer_update_consumed"] for event in invalid)


def test_pretext_rejects_foreign_normalization_before_texture_access(tmp_path):
    view = SimpleNamespace(identity="a" * 64)
    normalization = RateNormalization((0.0, 0.0), (1.0, 1.0), "b" * 64, 1)
    with pytest.raises(ValueError, match="ancestor"):
        pretext.PretextTextures(view, normalization, tmp_path / "trace.jsonl")
