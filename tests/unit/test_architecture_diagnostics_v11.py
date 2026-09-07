"""Exact metric integration and row-preserving failure semantics on synthetic predictions."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from e_jepa_ttc.evaluation.architecture_diagnostics_v11 import evaluate_rows, posterior_diagnostics
from e_jepa_ttc.evaluation.stage63_65 import scientific_mid_per_row, strict_macro_mass
from e_jepa_ttc.models.phase_distribution_v11 import phase_edges, phase_to_finite_ttc, quantile


@pytest.fixture
def fixture():
    torch.set_num_threads(2)
    targets = np.tile([2.0, 4.0, 8.0, -2.0], 2)
    metadata = pd.DataFrame(
        dict(
            sample_token=[f"r{i}" for i in range(8)],
            sequence_id=["s0"] * 4 + ["s1"] * 4,
            track_id=["t0", "t1"] * 4,
            outer_fold=[0] * 4 + [1] * 4,
            target_ttc_s=targets,
        )
    )
    edges = phase_edges(dtype=torch.float64)
    logits = torch.zeros((8, 108), dtype=torch.float64)
    median = quantile(logits, edges)
    output = dict(
        logits=logits.numpy(),
        phase=median.numpy(),
        ttc=phase_to_finite_ttc(median).numpy(),
        direct_logits=logits.numpy().copy(),
        score_correction=logits.numpy().copy(),
        evidence=np.zeros((8, 4, 108)),
        observability=np.zeros((8, 4, 108), dtype=bool),
    )
    old = pd.DataFrame(
        dict(A5=targets * 1.1, C2F=targets * 0.9, PAIR=targets * 1.2),
        index=pd.Index(metadata.sample_token),
    )
    return metadata, output, edges.numpy(), old


def test_float64_posterior_and_full_metric_match_historical_implementation(fixture):
    metadata, output, edges, old = fixture
    result = evaluate_rows(metadata, output, edges, old)
    mass = strict_macro_mass(metadata.target_ttc_s.to_numpy(), metadata.sequence_id.to_numpy())
    loss = scientific_mid_per_row(metadata.target_ttc_s.to_numpy(), output["ttc"])
    assert result.summary["score"] == pytest.approx(mass @ loss, abs=1e-12)
    np.testing.assert_allclose(result.arrays["posterior108"], 1 / 108, atol=1e-15)
    np.testing.assert_allclose(result.rows.posterior_entropy_normalized, 1, atol=1e-15)
    assert not result.rows.failure.any()
    assert not result.rows.geometric_validity_fraction.any()
    assert set(result.rows.calibration_status) == {"not_calibrated"}
    assert result.summary["old_hull_oracle"] <= result.summary["old_selector_oracle"]
    assert result.summary["old_plus_new_oracle"] <= result.summary["old_selector_oracle"]
    assert len(result.summary["group_statistics"]["sequence_id"]) == 2
    assert len(result.summary["group_statistics"]["sequence_id/track_id"]) == 4
    json.dumps(result.summary, allow_nan=False)


def test_nonfinite_rows_are_retained_and_never_receive_partial_score(fixture):
    metadata, output, edges, old = fixture
    output["logits"][1, 0] = np.nan
    output["ttc"][3] = np.inf
    result = evaluate_rows(metadata, output, edges, old)
    assert len(result.rows) == 8
    assert result.rows.failure.tolist() == [False, True, False, True, False, False, False, False]
    assert result.summary["score"] is None
    assert result.summary["old_plus_new_oracle"] is None
    assert np.isnan(result.arrays["posterior108"][[1, 3]]).all()
    assert result.rows.sample_token.tolist() == metadata.sample_token.tolist()
    json.dumps(result.summary, allow_nan=False)


@pytest.mark.parametrize("failed", [False, True])
def test_expanded_diagnostics_normalize_present_buckets_without_dropping_failures(fixture, failed):
    from e_jepa_ttc.evaluation.capacity_metrics import present_bucket_mass

    metadata, output, edges, old = fixture
    positions = [0, 1, 2, 4]
    metadata = metadata.iloc[positions].copy()
    old = old.iloc[positions]
    output = {key: value[positions].copy() for key, value in output.items()}
    if failed:
        output["ttc"][0] = np.inf
    result = evaluate_rows(metadata, output, edges, old, expanded_cohort=True)
    expected = present_bucket_mass(
        metadata.target_ttc_s.to_numpy(), metadata.sequence_id.to_numpy()
    )
    np.testing.assert_allclose(result.rows.global_mass, expected)
    assert result.rows.groupby("sequence_id").global_mass.sum().tolist() == pytest.approx(
        [0.5, 0.5]
    )
    assert result.rows.sample_token.tolist() == metadata.sample_token.tolist()
    assert len(result.arrays["posterior108"]) == 4
    assert result.summary["metric"] == "present_bucket_normalized_sequence_macro_signed_MiD"
    if failed:
        assert result.summary["score"] is None
        assert result.rows.failure.sum() == 1
    else:
        assert result.summary["score"] == pytest.approx(expected @ result.rows.mid_loss.to_numpy())
    json.dumps(result.summary, allow_nan=False)


def test_exact_alignment_and_boolean_geometry_are_required(fixture):
    metadata, output, edges, old = fixture
    with pytest.raises(ValueError, match="token order"):
        evaluate_rows(metadata, output, edges, old.iloc[::-1])
    duplicate = metadata.copy()
    duplicate.loc[1, "sample_token"] = duplicate.loc[0, "sample_token"]
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_rows(duplicate, output, edges, old)
    output["observability"] = output["observability"].astype(np.float64)
    with pytest.raises(ValueError, match="boolean"):
        posterior_diagnostics(output, edges)


def test_finite_clip_phase_is_separate_from_near_zero_posterior_median(fixture):
    metadata, output, edges, old = fixture
    logits = torch.full((8, 108), -1000.0, dtype=torch.float64)
    boundary = int(np.argmin(abs(edges)))
    logits[:, boundary - 1 : boundary + 1] = 0
    median = quantile(logits, torch.from_numpy(edges))
    output.update(
        logits=logits.numpy(), phase=median.numpy(), ttc=phase_to_finite_ttc(median).numpy()
    )
    result = evaluate_rows(metadata, output, edges, old)
    assert np.abs(result.rows.prediction_ttc_s).eq(60).all()
    assert (abs(result.rows.finite_clip_phase_delta) > 0.001).all()
    assert (result.rows.phase_q10 < 0).all() and (result.rows.phase_q90 > 0).all()
    assert result.summary["phase_nll_infinite_rows"] == 8
    assert result.summary["phase_nll_mean"] is None
    assert np.isfinite(result.rows.phase_crps).all()
    json.dumps(result.summary, allow_nan=False)
