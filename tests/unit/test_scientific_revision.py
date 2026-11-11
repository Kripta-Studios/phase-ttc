"""Matched comparisons must preserve identities, coverage and recording units."""

import numpy as np
import pytest

from operational.scientific_revision import (
    align_records,
    geometry_arms,
    paired_mae,
    prediction_summary,
    simple_fusion,
)


def test_join_uses_query_identity_and_rejects_duplicates() -> None:
    rows = [{"token": "b", "value": 2}, {"token": "a", "value": 1}]
    assert [r["value"] for r in align_records(rows, ["a", "b"])] == [1, 2]
    with pytest.raises(ValueError, match="duplicate"):
        align_records(rows + [rows[0]], ["a", "b"])
    with pytest.raises(ValueError, match="identity"):
        align_records(rows, ["a", "c"])


def test_missing_geometry_uses_exact_original_without_changing_coverage() -> None:
    original = np.array([2.0, 3.0, -4.0])
    geometry = np.array([1.0, np.nan, 4.0])
    arms = geometry_arms(original, geometry)
    np.testing.assert_array_equal(arms["geometry_with_original_fallback"], [1.0, 3.0, 4.0])
    np.testing.assert_allclose(arms["fixed_half_inverse_rate"], [4 / 3, 3, 60])
    assert np.isnan(arms["geometry_only"][1])


def test_summary_reports_denominator_and_common_support_explicitly() -> None:
    target = np.array([1.0, 2.0, 3.0])
    summary = prediction_summary(target, np.array([2.0, np.nan, 4.0]))
    assert summary["population"] == 3
    assert summary["scored"] == 2
    assert summary["coverage"] == 2 / 3
    assert summary["MAE_s"] == 1
    with pytest.raises(ValueError, match="target"):
        prediction_summary(np.array([0.0]), np.array([1.0]))


def test_paired_bootstrap_resamples_recordings_not_rows() -> None:
    target = np.array([1.0, 1.0, 1.0, 1.0])
    candidate = np.array([2.0, 2.0, 2.0, 1.0])
    reference = target.copy()
    groups = np.array(["A", "A", "A", "B"])
    result = paired_mae(target, candidate, reference, groups, draws=1000, seed=7)
    assert result["delta_MAE_s"] == 0.75
    assert result["ci95"] == [0.0, 1.0]
    assert result["recordings"] == 2
    assert result["resampling_unit"] == "recording"


def test_paired_comparison_rejects_missing_rows_and_one_recording_ci() -> None:
    with pytest.raises(ValueError, match="common"):
        paired_mae(np.ones(2), np.array([1, np.nan]), np.ones(2), np.array(["A", "B"]))
    result = paired_mae(np.ones(2), np.ones(2), np.ones(2), np.array(["A", "A"]))
    assert result["ci95"] is None


def test_fixed_fusion_distinguishes_coordinates_and_validates_domain() -> None:
    rgb, event = np.array([2.0]), np.array([4.0])
    assert simple_fusion(rgb, event, "ttc")[0] == 3
    assert simple_fusion(rgb, event, "inverse_rate")[0] == pytest.approx(8 / 3)
    phase = -np.log1p(-0.1 / np.array([2.0, 4.0])).mean()
    assert simple_fusion(rgb, event, "phase")[0] == pytest.approx(0.1 / -np.expm1(-phase))
    with pytest.raises(ValueError, match="domain"):
        simple_fusion(np.array([0.05]), event, "phase")


def test_head_controls_match_capacity_and_remove_event_information() -> None:
    import torch

    from operational.scientific_revision_heads import MatchedHead

    features = torch.zeros(4, 340)
    features[:, 160:165] = 0.04
    features[:, 330:335] = 0.08
    widths = torch.zeros(4, 2)
    for mode in ("phase_true", "phase_zero_event", "inverse_residual"):
        model = MatchedHead(mode)
        assert sum(p.numel() for p in model.parameters()) == 1955
        assert torch.allclose(model(features, widths)["raw_location"], features[:, 160], atol=1e-7)
    model = MatchedHead("phase_zero_event")
    torch.nn.init.normal_(model.output.weight)
    altered = features.clone()
    altered[:, 170:] = torch.randn_like(altered[:, 170:])
    assert torch.equal(model(features, widths)["ttc"], model(altered, widths)["ttc"])


def test_head_training_selection_excludes_all_evttc_rows() -> None:
    from operational.scientific_revision_heads import source_training_mask

    with pytest.raises(ValueError, match="88744"):
        source_training_mask({"domain": np.array(["eap", "evttc"])})
