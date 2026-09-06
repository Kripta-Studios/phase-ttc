"""Analytic TRAIN masses and label identity checks for old and expanded pools."""

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from e_jepa_ttc.training.stage72_supervision import train_supervision


def inputs():
    labels = pd.DataFrame(
        {
            "sample_token": ["a", "b", "c", "d", "e"],
            "sequence_id": ["one"] * 4 + ["two"],
            "target_ttc": [1.0, 4.0, 8.0, -1.0, 2.0],
            "bucket": ["crucial", "small", "large", "negative", "crucial"],
        }
    )
    view = SimpleNamespace(
        tokens=tuple(labels.sample_token),
        identity="a" * 64,
        reader=SimpleNamespace(metadata=labels[["sample_token", "sequence_id"]]),
    )
    return view, labels


def test_present_bucket_normalization_preserves_sequence_macro_mass():
    view, labels = inputs()
    supervision = train_supervision(view, labels, expanded=True)
    torch.testing.assert_close(
        supervision.mass, torch.tensor([0.25, 0.15, 0.05, 0.05, 0.5], dtype=torch.float64)
    )
    assert abs(float(supervision.prior_bin_mass.sum()) - 1) < 1e-12
    assert supervision.outside_support_mass == 0
    with pytest.raises(ValueError, match="all four buckets"):
        train_supervision(view, labels, expanded=False)


def test_label_sequence_and_bucket_substitution_rejected():
    view, labels = inputs()
    changed = labels.copy()
    changed.loc[0, "sequence_id"] = "other"
    with pytest.raises(ValueError, match="lineage"):
        train_supervision(view, changed, expanded=True)
    changed = labels.copy()
    changed.loc[0, "bucket"] = "negative"
    with pytest.raises(ValueError, match="disagrees"):
        train_supervision(view, changed, expanded=True)


def test_outside_phase_support_target_kept_and_reported():
    view, labels = inputs()
    labels.loc[3, "target_ttc"] = -0.01
    supervision = train_supervision(view, labels, expanded=True)
    assert len(supervision.target_phase) == 5
    assert abs(supervision.outside_support_mass - 0.05) < 1e-12
    assert abs(float(supervision.prior_bin_mass.sum()) - 1) < 1e-12
