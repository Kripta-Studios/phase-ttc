from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.models.temporal_uncertainty import (
    propagate_temporal_uncertainty,
    replace_uncertainty_after_head_regeneration,
    revise_causal_scale_output,
)
from operational.uncertainty_revision.replay import main as replay_main


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ratio = torch.tensor([[0.08, 0.08]], dtype=torch.float64)
    log_variance = torch.full_like(ratio, -6.0)
    known = torch.ones_like(ratio, dtype=torch.bool)
    delta_t = torch.full_like(ratio, 0.1)
    return ratio, log_variance, known, delta_t


def test_independent_delta_variance_matches_autograd_jacobian() -> None:
    ratio, log_variance, known, delta_t = _inputs()
    weight = 0.75
    revised = propagate_temporal_uncertainty(
        ratio,
        log_variance,
        known,
        delta_t,
        current_pair_weight=weight,
        covariance_assumption="independent",
    )

    differentiable_ratio = ratio[0].clone().requires_grad_()
    pair_inverse = torch.expm1(differentiable_ratio) / delta_t[0]
    transported = pair_inverse[0] / (1.0 - delta_t[0, -1] * pair_inverse[0])
    mean = weight * pair_inverse[1] + (1.0 - weight) * transported
    jacobian = torch.autograd.grad(mean, differentiable_ratio)[0]
    expected = (jacobian.square() * log_variance[0].exp()).sum()

    torch.testing.assert_close(revised.inverse_variance[0], expected)
    assert revised.temporal_blend_used.tolist() == [True]
    assert revised.covariance_assumption == "independent_log_ratio_errors"


def test_current_only_reproduces_historical_current_pair_variance() -> None:
    ratio, log_variance, known, delta_t = _inputs()
    revised = propagate_temporal_uncertainty(
        ratio,
        log_variance,
        known,
        delta_t,
        current_pair_weight=1.0,
    )
    expected_mean = torch.expm1(ratio[:, -1]) / delta_t[:, -1]
    expected_variance = (ratio[:, -1].exp() / delta_t[:, -1]).square() * log_variance[:, -1].exp()
    torch.testing.assert_close(revised.inverse_ttc_mean, expected_mean)
    torch.testing.assert_close(revised.inverse_variance, expected_variance)
    assert revised.temporal_blend_used.tolist() == [False]


def test_single_pair_has_no_temporal_blend() -> None:
    ratio, log_variance, known, delta_t = _inputs()
    revised = propagate_temporal_uncertainty(
        ratio[:, -1:],
        log_variance[:, -1:],
        known[:, -1:],
        delta_t[:, -1:],
        current_pair_weight=0.25,
    )
    assert revised.temporal_blend_used.tolist() == [False]
    torch.testing.assert_close(
        revised.inverse_ttc_mean,
        torch.expm1(ratio[:, -1]) / delta_t[:, -1],
    )


def test_invalid_transport_falls_back_to_current_distribution() -> None:
    ratio, log_variance, known, delta_t = _inputs()
    ratio[:, 0] = math.log(2.0)  # previous inverse=10, denominator=0 for dt=0.1
    revised = propagate_temporal_uncertainty(
        ratio,
        log_variance,
        known,
        delta_t,
        current_pair_weight=0.75,
    )
    current_mean = torch.expm1(ratio[:, -1]) / delta_t[:, -1]
    current_variance = (ratio[:, -1].exp() / delta_t[:, -1]).square() * log_variance[:, -1].exp()
    assert revised.temporal_blend_used.tolist() == [False]
    torch.testing.assert_close(revised.inverse_ttc_mean, current_mean)
    torch.testing.assert_close(revised.inverse_variance, current_variance)


def test_conservative_covariance_is_upper_bound_and_matches_perfect_correlation() -> None:
    ratio, log_variance, known, delta_t = _inputs()
    independent = propagate_temporal_uncertainty(
        ratio,
        log_variance,
        known,
        delta_t,
        current_pair_weight=0.75,
        covariance_assumption="independent",
    )
    conservative = propagate_temporal_uncertainty(
        ratio,
        log_variance,
        known,
        delta_t,
        current_pair_weight=0.75,
        covariance_assumption="conservative_upper_bound",
    )
    ratio_variance = log_variance.exp()
    perfect_covariance = (ratio_variance[:, -2] * ratio_variance[:, -1]).sqrt()
    provided = propagate_temporal_uncertainty(
        ratio,
        log_variance,
        known,
        delta_t,
        current_pair_weight=0.75,
        ratio_covariance=perfect_covariance,
    )
    assert bool((conservative.inverse_variance >= independent.inverse_variance).all())
    torch.testing.assert_close(conservative.inverse_variance, provided.inverse_variance)
    assert conservative.covariance_assumption == "cauchy_schwarz_upper_bound"


def test_risk_logits_use_corrected_inverse_variance_and_unknowns_are_neutral() -> None:
    ratio, log_variance, known, delta_t = _inputs()
    known[0, -1] = False
    revised = propagate_temporal_uncertainty(
        ratio,
        log_variance,
        known,
        delta_t,
        current_pair_weight=0.75,
    )
    assert revised.known_mask.tolist() == [False]
    torch.testing.assert_close(revised.collision_logits, torch.zeros_like(revised.collision_logits))
    assert bool(torch.isfinite(revised.inverse_ttc_log_variance).all())
    assert bool(torch.isfinite(revised.ttc_log_variance).all())


def test_saturated_ratio_has_finite_floor_variance() -> None:
    ratio, log_variance, known, delta_t = _inputs()
    ratio[:, -1] = 20.0
    revised = propagate_temporal_uncertainty(
        ratio[:, -1:],
        log_variance[:, -1:],
        known[:, -1:],
        delta_t[:, -1:],
        current_pair_weight=1.0,
    )
    torch.testing.assert_close(
        revised.inverse_variance,
        torch.full_like(revised.inverse_variance, 1.0e-16),
    )
    assert bool(torch.isfinite(revised.inverse_ttc_log_variance).all())
    assert revised.uncertainty_valid_mask.tolist() == [False]
    assert revised.ratio_saturated_mask.tolist() == [[True]]
    torch.testing.assert_close(revised.collision_logits, torch.zeros_like(revised.collision_logits))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ratio", float("nan"), "pair_log_ratio must be finite"),
        ("log_variance", 1000.0, "overflows"),
    ],
)
def test_invalid_uncertainty_numerics_are_rejected(
    field: str,
    value: float,
    message: str,
) -> None:
    ratio, log_variance, known, delta_t = _inputs()
    target = ratio if field == "ratio" else log_variance
    target[0, 0] = value
    with pytest.raises(ValueError, match=message):
        propagate_temporal_uncertainty(
            ratio,
            log_variance,
            known,
            delta_t,
            current_pair_weight=0.75,
        )


def test_covariance_outside_psd_bound_is_rejected_at_small_variance() -> None:
    ratio, log_variance, known, delta_t = _inputs()
    variance = log_variance.exp()
    covariance_limit = (variance[:, -2] * variance[:, -1]).sqrt()
    with pytest.raises(ValueError, match="Cauchy-Schwarz"):
        propagate_temporal_uncertainty(
            ratio,
            log_variance,
            known,
            delta_t,
            current_pair_weight=0.75,
            ratio_covariance=1.01 * covariance_limit,
        )


def test_pair_known_requires_boolean_dtype() -> None:
    ratio, log_variance, known, delta_t = _inputs()
    with pytest.raises(ValueError, match="pair_known must be a boolean"):
        propagate_temporal_uncertainty(
            ratio,
            log_variance,
            known.float(),
            delta_t,
            current_pair_weight=0.75,
        )


def test_half_precision_is_rejected_before_variance_floor_underflows() -> None:
    ratio, log_variance, known, delta_t = _inputs()
    with pytest.raises(ValueError, match="float32 or float64"):
        propagate_temporal_uncertainty(
            ratio.half(),
            log_variance.half(),
            known,
            delta_t.half(),
            current_pair_weight=0.75,
        )


def test_ttc_variance_overflow_is_rejected() -> None:
    ratio = torch.tensor([[1.0e-20]], dtype=torch.float32)
    log_variance = torch.tensor([[80.0]], dtype=torch.float32)
    known = torch.ones_like(ratio, dtype=torch.bool)
    delta_t = torch.ones_like(ratio)
    with pytest.raises(ValueError, match="TTC variance is non-finite"):
        propagate_temporal_uncertainty(
            ratio,
            log_variance,
            known,
            delta_t,
            current_pair_weight=1.0,
            ttc_clip_seconds=1.0e30,
        )


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf")])
def test_transport_denominator_guard_must_be_positive_and_finite(value: float) -> None:
    ratio, log_variance, known, delta_t = _inputs()
    with pytest.raises(ValueError, match="transport_min_denominator"):
        propagate_temporal_uncertainty(
            ratio,
            log_variance,
            known,
            delta_t,
            current_pair_weight=0.75,
            transport_min_denominator=value,
        )


def test_legacy_model_schema_and_strict_state_dict_remain_unchanged() -> None:
    config = CausalScaleTTCConfig(hidden_dim=8, geometry_dim=8, residual_depth=1)
    source = CausalScaleTTC(config)
    state = source.state_dict()
    target = CausalScaleTTC(config)
    result = target.load_state_dict(state, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
    assert source.checkpoint_config() == target.checkpoint_config()


def test_legacy_calibrated_head_reuse_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="regenerate dependent features"):
        replace_uncertainty_after_head_regeneration(  # type: ignore[arg-type]
            SimpleNamespace(),
            SimpleNamespace(),  # type: ignore[arg-type]
        )


def test_revision_from_unrelated_batch_cannot_replace_output() -> None:
    output = SimpleNamespace(
        inverse_ttc_mean=torch.tensor([1.0]),
        ttc_mean_seconds=torch.tensor([1.0]),
        known_mask=torch.tensor([True]),
    )
    revision = SimpleNamespace(
        version="temporal_ttc_uncertainty_delta_v1",
        inverse_ttc_mean=torch.tensor([2.0]),
        ttc_mean_seconds=torch.tensor([0.5]),
        known_mask=torch.tensor([True]),
    )
    with pytest.raises(ValueError, match="point estimates do not match"):
        replace_uncertainty_after_head_regeneration(  # type: ignore[arg-type]
            output,
            revision,  # type: ignore[arg-type]
            downstream_artifacts_regenerated=True,
        )


def test_causal_output_adapter_rejects_wrong_explicit_ttc_clip() -> None:
    ratio = torch.tensor([[0.001]], dtype=torch.float64)
    log_variance = torch.full_like(ratio, -6.0)
    known = torch.ones_like(ratio, dtype=torch.bool)
    delta_t = torch.full_like(ratio, 0.1)
    historical = propagate_temporal_uncertainty(
        ratio,
        log_variance,
        known,
        delta_t,
        current_pair_weight=1.0,
        ttc_clip_seconds=60.0,
    )
    output = SimpleNamespace(
        pair_log_height_ratio=ratio,
        log_ratio_log_variance=log_variance,
        diagnostics={"pair_known": known},
        inverse_ttc_mean=historical.inverse_ttc_mean,
        ttc_mean_seconds=historical.ttc_mean_seconds,
    )
    with pytest.raises(ValueError, match="ttc_clip_seconds"):
        revise_causal_scale_output(  # type: ignore[arg-type]
            output,
            delta_t,
            current_pair_weight=1.0,
            risk_thresholds_s=(0.5, 1.0),
            ttc_clip_seconds=10.0,
        )


def test_operational_replay_writes_versioned_standalone_result(tmp_path: Path) -> None:
    ratio, log_variance, known, delta_t = _inputs()
    source = tmp_path / "pairs.npz"
    destination = tmp_path / "revision.npz"
    np.savez(
        source,
        pair_log_ratio=ratio.numpy(),
        pair_log_variance=log_variance.numpy(),
        pair_known=known.numpy(),
        pair_delta_t_s=delta_t.numpy(),
    )
    assert (
        replay_main(
            [
                "--input",
                str(source),
                "--output",
                str(destination),
                "--current-pair-weight",
                "0.75",
            ]
        )
        == 0
    )
    with np.load(destination, allow_pickle=False) as result:
        assert result["inverse_ttc_mean"].shape == (1,)
        assert result["uncertainty_valid_mask"].tolist() == [True]
    provenance = json.loads(
        destination.with_suffix(".npz.provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["schema"] == "temporal_ttc_uncertainty_delta_v1"
    assert provenance["compatible_with_legacy_calibrated_heads"] is False


def test_operational_replay_rejects_output_without_npz_suffix(tmp_path: Path) -> None:
    ratio, log_variance, known, delta_t = _inputs()
    source = tmp_path / "pairs.npz"
    destination = tmp_path / "revision"
    np.savez(
        source,
        pair_log_ratio=ratio.numpy(),
        pair_log_variance=log_variance.numpy(),
        pair_known=known.numpy(),
        pair_delta_t_s=delta_t.numpy(),
    )
    with pytest.raises(ValueError, match="must end in .npz"):
        replay_main(
            [
                "--input",
                str(source),
                "--output",
                str(destination),
                "--current-pair-weight",
                "0.75",
            ]
        )
    assert not destination.exists()
    assert not destination.with_suffix(".npz").exists()
