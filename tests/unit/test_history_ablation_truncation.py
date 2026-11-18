"""Synthetic tests for the causal slot truncation of the history ablation."""

from __future__ import annotations

import numpy as np
import pytest

from operational.history_ablation.folds import (
    BLOCKS,
    chain_blocks,
    fusion_train_blocks,
    refiner_block,
)
from operational.history_ablation.truncation import (
    GAP_FIELD,
    truncate_history,
    truncate_mapping,
)


def _histories(seed: int = 0, n: int = 64, length: int = 8):
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, length + 1, size=n)
    valid = np.arange(length)[None, :] >= (length - counts)[:, None]
    features = rng.normal(size=(n, length, 17)).astype(np.float32)
    timing = rng.uniform(0.01, 0.5, size=(n, length, 4)).astype(np.float32)
    features[~valid] = 0
    timing[~valid] = 0
    first = valid.argmax(1)
    timing[np.arange(n), first, GAP_FIELD] = 0
    return features, timing, valid


@pytest.mark.parametrize("k", [1, 2, 4])
def test_keeps_only_k_most_recent_valid_slots(k):
    features, timing, valid = _histories()
    out_f, out_t, out_v = truncate_history(features, timing, valid, k)
    assert np.array_equal(out_v.sum(1), np.minimum(valid.sum(1), k))
    assert out_v[:, -1].all()
    assert not np.any(out_v[:, :-1] & ~out_v[:, 1:])  # contiguous suffix preserved
    assert np.all(out_f[~out_v] == 0) and np.all(out_t[~out_v] == 0)
    kept = out_v
    assert np.array_equal(out_f[kept], features[kept])
    first = out_v.argmax(1)
    rows = np.arange(len(out_v))
    assert np.all(out_t[rows, first, GAP_FIELD] == 0)
    mask = kept.copy()
    mask[rows, first] = False
    assert np.array_equal(out_t[mask], timing[mask])
    assert np.array_equal(out_t[rows, first, :GAP_FIELD], timing[rows, first, :GAP_FIELD])


def test_k_at_least_length_is_identity():
    features, timing, valid = _histories(seed=1)
    for k in (8, 9):
        out_f, out_t, out_v = truncate_history(features, timing, valid, k)
        assert np.array_equal(out_f, features)
        assert np.array_equal(out_t, timing)
        assert np.array_equal(out_v, valid)


def test_output_does_not_depend_on_dropped_older_slots():
    features, timing, valid = _histories(seed=2)
    k = 2
    reference = truncate_history(features, timing, valid, k)
    perturbed_f, perturbed_t = features.copy(), timing.copy()
    older = valid & (np.arange(8)[None, :] < 8 - k)
    perturbed_f[older] += 100.0
    perturbed_t[older] += 7.0
    other = truncate_history(perturbed_f, perturbed_t, valid, k)
    for left, right in zip(reference, other, strict=True):
        assert np.array_equal(left, right)


def test_each_row_is_truncated_independently():
    features, timing, valid = _histories(seed=3)
    batch = truncate_history(features, timing, valid, 4)
    for row in (0, 5, 17):
        single = truncate_history(
            features[row : row + 1], timing[row : row + 1], valid[row : row + 1], 4
        )
        for whole, part in zip(batch, single, strict=True):
            assert np.array_equal(whole[row : row + 1], part)


def test_identical_operator_for_training_cache_and_generation_fragment():
    features, timing, valid = _histories(seed=4)
    experts = np.ones((len(valid), 3), np.float32)
    cache = {
        "features": features,
        "timing": timing,
        "valid": valid,
        "expert_phase": experts,
        "target_phase": np.zeros(len(valid), np.float32),
        "group_id": np.array(["g"] * len(valid)),
    }
    fragment = {
        "features": features.copy(),
        "timing": timing.copy(),
        "valid": valid.copy(),
        "expert_phase": experts,
        "sample_token": np.array(["t"] * len(valid)),
    }
    a, b = truncate_mapping(cache, 2), truncate_mapping(fragment, 2)
    for key in ("features", "timing", "valid"):
        assert np.array_equal(a[key], b[key])
    assert np.array_equal(a["expert_phase"], experts)  # current-slot experts untouched
    assert set(a) == set(cache) and set(b) == set(fragment)


def test_rejects_noncontiguous_or_missing_current_slot():
    features, timing, valid = _histories(seed=5, n=4)
    bad = valid.copy()
    bad[0, -1] = False
    with pytest.raises(ValueError):
        truncate_history(features, timing, bad, 2)
    with pytest.raises(ValueError):
        truncate_history(features, timing, valid, 0)


def test_masked_refiner_equals_refiner_on_short_history():
    """The released GRU refiner treats masked slots exactly as absent slots."""
    try:
        import torch

        from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
    except Exception as error:  # broken torch builds raise non-ImportError exceptions
        pytest.skip(f"torch unavailable: {error}")
    torch.manual_seed(0)
    model = TemporalRefiner(
        TemporalConfig(feature_count=17, hidden=160, backbone="gru", output_mode="residual")
    ).eval()
    for module in (model.location, model.width, model.cost):
        torch.nn.init.normal_(module.weight, std=0.1)
    features, timing, valid = _histories(seed=6, n=16)
    valid[:] = True  # full histories, so the short form is a plain slice
    k = 2
    tf, tt, tv = truncate_history(features, timing, valid, k)
    short_t = timing[:, -k:].copy()
    short_t[:, 0, GAP_FIELD] = 0
    experts = torch.full((16, 3), 0.02)
    with torch.no_grad():
        masked = model(torch.from_numpy(tf), torch.from_numpy(tt), torch.from_numpy(tv), experts)
        short = model(
            torch.from_numpy(features[:, -k:].copy()),
            torch.from_numpy(short_t),
            torch.ones((16, k), dtype=torch.bool),
            experts,
        )
    for key in ("raw_location", "q10", "q90"):
        assert torch.allclose(masked[key], short[key], atol=1e-6)


def test_fold_rules_keep_each_chain_off_its_own_block():
    for block in range(BLOCKS):
        train = fusion_train_blocks(block)
        assert block not in train and (block - 1) % BLOCKS not in train
        assert len(set(train)) == 3
        assert block not in chain_blocks(block)
        # no fusion-training block's context refiner was trained on the test block
        assert all(refiner_block(b) != block for b in train)
