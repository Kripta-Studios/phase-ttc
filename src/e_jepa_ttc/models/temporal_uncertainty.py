"""Versioned, inference-only uncertainty propagation for temporal TTC blends.

The historical causal-scale checkpoints learned a variance for each pairwise
log-height ratio.  When their point estimate blends the current inverse TTC
with a transported previous estimate, that pair variance is not itself the
variance of the blend.  This module corrects the propagation after inference;
it deliberately adds no parameters and does not alter checkpoint loading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCOutput

TEMPORAL_UNCERTAINTY_REVISION = "temporal_ttc_uncertainty_delta_v1"

CovarianceAssumption = Literal["independent", "conservative_upper_bound"]


@dataclass(frozen=True)
class TemporalUncertaintyRevision:
    """Corrected distribution fields plus replay and provenance masks."""

    inverse_ttc_mean: torch.Tensor
    inverse_ttc_log_variance: torch.Tensor
    ttc_mean_seconds: torch.Tensor
    ttc_log_variance: torch.Tensor
    collision_logits: torch.Tensor
    known_mask: torch.Tensor
    temporal_blend_used: torch.Tensor
    uncertainty_valid_mask: torch.Tensor
    ttc_uncertainty_valid_mask: torch.Tensor
    ratio_saturated_mask: torch.Tensor
    inverse_variance: torch.Tensor
    ttc_variance: torch.Tensor
    version: str
    covariance_assumption: str
    inference_only: bool = True
    compatible_with_legacy_calibrated_heads: bool = False


def _validate_inputs(
    pair_log_ratio: torch.Tensor,
    pair_log_variance: torch.Tensor,
    pair_known: torch.Tensor,
    pair_delta_t_s: torch.Tensor,
) -> None:
    if pair_log_ratio.ndim != 2:
        raise ValueError("pair tensors must have shape [B,P]")
    if not (
        pair_log_variance.shape == pair_known.shape == pair_delta_t_s.shape == pair_log_ratio.shape
    ):
        raise ValueError("pair ratio, log variance, known mask, and delta_t must share shape")
    if pair_log_ratio.shape[1] < 1:
        raise ValueError("at least one temporal pair is required")
    if pair_known.dtype is not torch.bool:
        raise ValueError("pair_known must be a boolean tensor")
    if not bool(torch.isfinite(pair_delta_t_s).all()) or bool((pair_delta_t_s <= 0.0).any()):
        raise ValueError("pair_delta_t_s must be finite and strictly positive")
    if not bool(torch.isfinite(pair_log_variance).all()):
        raise ValueError("pair_log_variance must be finite")
    if not bool(torch.isfinite(pair_log_ratio).all()):
        raise ValueError("pair_log_ratio must be finite")
    if pair_log_ratio.dtype not in {torch.float32, torch.float64}:
        raise ValueError("pair ratios must use float32 or float64")
    if pair_log_variance.dtype != pair_log_ratio.dtype:
        raise ValueError("pair log variances must match the pair-ratio dtype")
    if pair_delta_t_s.dtype != pair_log_ratio.dtype:
        raise ValueError("pair delta_t must match the pair-ratio dtype")
    if not (
        pair_log_variance.device
        == pair_known.device
        == pair_delta_t_s.device
        == pair_log_ratio.device
    ):
        raise ValueError("all pair tensors must be on the same device")


def propagate_temporal_uncertainty(
    pair_log_ratio: torch.Tensor,
    pair_log_variance: torch.Tensor,
    pair_known: torch.Tensor,
    pair_delta_t_s: torch.Tensor,
    *,
    current_pair_weight: float,
    risk_thresholds_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    ttc_clip_seconds: float = 60.0,
    covariance_assumption: CovarianceAssumption = "conservative_upper_bound",
    ratio_covariance: torch.Tensor | None = None,
    ratio_clip: float = 12.0,
    transport_min_denominator: float = 1.0e-4,
    variance_floor: float = 1.0e-16,
) -> TemporalUncertaintyRevision:
    """Propagate pair log-ratio uncertainty through transport and blending.

    ``ratio_covariance`` is covariance between the previous and current
    *log-ratio* errors.  If it is unavailable, the default is the explicit
    Cauchy--Schwarz upper bound.  ``independent`` is available for controlled
    replay but must not be mistaken for a measured covariance.
    """

    _validate_inputs(pair_log_ratio, pair_log_variance, pair_known, pair_delta_t_s)
    if not 0.0 <= current_pair_weight <= 1.0:
        raise ValueError("current_pair_weight must lie in [0,1]")
    if covariance_assumption not in {"independent", "conservative_upper_bound"}:
        raise ValueError("unsupported covariance assumption")
    if not math.isfinite(ttc_clip_seconds) or ttc_clip_seconds <= 0.0:
        raise ValueError("ttc_clip_seconds must be finite and positive")
    if not math.isfinite(ratio_clip) or ratio_clip <= 0.0:
        raise ValueError("ratio_clip must be finite and positive")
    if not math.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError("variance_floor must be finite and positive")
    if not math.isfinite(transport_min_denominator) or transport_min_denominator <= 0.0:
        raise ValueError("transport_min_denominator must be finite and positive")
    if not risk_thresholds_s or any(
        not math.isfinite(value) or value <= 0.0 for value in risk_thresholds_s
    ):
        raise ValueError("risk thresholds must be finite and positive")

    ratio = pair_log_ratio.clamp(-ratio_clip, ratio_clip)
    pair_inverse = torch.expm1(ratio) / pair_delta_t_s
    pair_ratio_variance = pair_log_variance.exp()
    if not bool(torch.isfinite(pair_ratio_variance).all()):
        raise ValueError("pair log variance overflows in its tensor dtype")
    # This is the exact derivative of expm1(clamp(r))/dt away from the two
    # non-differentiable boundaries.  Saturated ratios carry no local delta
    # sensitivity; the finite variance floor records that clipping explicitly.
    ratio_active = (pair_log_ratio >= -ratio_clip) & (pair_log_ratio <= ratio_clip)
    pair_jacobian = torch.where(
        ratio_active,
        ratio.exp() / pair_delta_t_s,
        torch.zeros_like(pair_log_ratio),
    )

    current = pair_inverse[:, -1]
    current_variance = pair_jacobian[:, -1].square() * pair_ratio_variance[:, -1]
    blend_used = torch.zeros_like(current, dtype=torch.bool)
    inverse_mean = current
    inverse_variance = current_variance
    covariance_label = "not_applicable_current_only"
    uncertainty_valid = ratio_active[:, -1]

    can_blend = pair_inverse.shape[1] >= 2 and current_pair_weight < 1.0
    if can_blend:
        previous = pair_inverse[:, -2]
        current_dt = pair_delta_t_s[:, -1]
        denominator = 1.0 - current_dt * previous
        transported = previous / denominator.clamp_min(transport_min_denominator)
        blended = current_pair_weight * current + (1.0 - current_pair_weight) * transported
        ratio_argument = 1.0 + current_dt * blended
        blend_used = (
            pair_known[:, -2]
            & torch.isfinite(denominator)
            & (denominator > transport_min_denominator)
            & torch.isfinite(blended)
            & torch.isfinite(ratio_argument)
            & (ratio_argument > 0.0)
        )
        inverse_mean = torch.where(blend_used, blended, current)

        current_coefficient = current_pair_weight * pair_jacobian[:, -1]
        transport_jacobian = denominator.reciprocal().square()
        previous_coefficient = (
            (1.0 - current_pair_weight) * transport_jacobian * pair_jacobian[:, -2]
        )
        current_ratio_variance = pair_ratio_variance[:, -1]
        previous_ratio_variance = pair_ratio_variance[:, -2]
        if ratio_covariance is not None:
            if ratio_covariance.shape != current.shape:
                raise ValueError("ratio_covariance must have shape [B]")
            if not bool(torch.isfinite(ratio_covariance).all()):
                raise ValueError("ratio_covariance must be finite")
            covariance_limit = (current_ratio_variance * previous_ratio_variance).sqrt()
            finfo = torch.finfo(covariance_limit.dtype)
            tolerance = 16.0 * finfo.eps * covariance_limit + finfo.tiny
            if bool((ratio_covariance.abs() > covariance_limit + tolerance).any()):
                raise ValueError("ratio_covariance violates the Cauchy-Schwarz bound")
            blended_variance = (
                current_coefficient.square() * current_ratio_variance
                + previous_coefficient.square() * previous_ratio_variance
                + 2.0 * current_coefficient * previous_coefficient * ratio_covariance
            )
            covariance_label = "provided_log_ratio_covariance"
        elif covariance_assumption == "independent":
            blended_variance = (
                current_coefficient.square() * current_ratio_variance
                + previous_coefficient.square() * previous_ratio_variance
            )
            covariance_label = "independent_log_ratio_errors"
        else:
            blended_variance = (
                current_coefficient * current_ratio_variance.sqrt()
                + previous_coefficient.abs() * previous_ratio_variance.sqrt()
            ).square()
            covariance_label = "cauchy_schwarz_upper_bound"
        inverse_variance = torch.where(blend_used, blended_variance, current_variance)
        blended_uncertainty_valid = torch.ones_like(blend_used)
        if current_pair_weight > 0.0:
            blended_uncertainty_valid = blended_uncertainty_valid & ratio_active[:, -1]
        if current_pair_weight < 1.0:
            blended_uncertainty_valid = blended_uncertainty_valid & ratio_active[:, -2]
        uncertainty_valid = torch.where(
            blend_used,
            blended_uncertainty_valid,
            ratio_active[:, -1],
        )

    if not bool(torch.isfinite(inverse_variance).all()):
        raise ValueError("propagated inverse TTC variance is non-finite")
    if bool((inverse_variance < -torch.finfo(inverse_variance.dtype).eps).any()):
        raise ValueError("provided covariance produces a negative variance")
    inverse_variance = inverse_variance.clamp_min(variance_floor)
    inverse_log_variance = inverse_variance.log()

    minimum_inverse = 1.0 / ttc_clip_seconds
    sign = torch.where(
        inverse_mean < 0.0,
        -torch.ones_like(inverse_mean),
        torch.ones_like(inverse_mean),
    )
    safe_inverse = sign * inverse_mean.abs().clamp_min(minimum_inverse)
    ttc_mean = safe_inverse.reciprocal().clamp(-ttc_clip_seconds, ttc_clip_seconds)
    ttc_jacobian = torch.where(
        inverse_mean.abs() > minimum_inverse,
        -inverse_mean.reciprocal().square(),
        torch.zeros_like(inverse_mean),
    )
    ttc_variance = (ttc_jacobian.square() * inverse_variance).clamp_min(variance_floor)
    if not bool(torch.isfinite(ttc_variance).all()):
        raise ValueError("propagated TTC variance is non-finite")
    ttc_log_variance = ttc_variance.log()
    ttc_uncertainty_valid = uncertainty_valid & (inverse_mean.abs() > minimum_inverse)

    known = pair_known[:, -1] & torch.isfinite(inverse_mean)
    inverse_std = inverse_variance.sqrt().clamp_min(math.sqrt(variance_floor))
    thresholds = inverse_mean.new_tensor(risk_thresholds_s)
    standardized = (inverse_mean[:, None] - thresholds.reciprocal()[None, :]) / inverse_std[:, None]
    probability = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
    collision_logits = torch.logit(probability.clamp(1.0e-6, 1.0 - 1.0e-6))
    risk_valid = known & uncertainty_valid
    collision_logits = torch.where(
        risk_valid[:, None], collision_logits, torch.zeros_like(collision_logits)
    )

    return TemporalUncertaintyRevision(
        inverse_ttc_mean=inverse_mean,
        inverse_ttc_log_variance=inverse_log_variance,
        ttc_mean_seconds=ttc_mean,
        ttc_log_variance=ttc_log_variance,
        collision_logits=collision_logits,
        known_mask=known,
        temporal_blend_used=blend_used,
        uncertainty_valid_mask=uncertainty_valid,
        ttc_uncertainty_valid_mask=ttc_uncertainty_valid,
        ratio_saturated_mask=~ratio_active,
        inverse_variance=inverse_variance,
        ttc_variance=ttc_variance,
        version=TEMPORAL_UNCERTAINTY_REVISION,
        covariance_assumption=covariance_label,
    )


def revise_causal_scale_output(
    output: CausalScaleTTCOutput,
    pair_delta_t_s: torch.Tensor,
    *,
    current_pair_weight: float,
    risk_thresholds_s: tuple[float, ...],
    ttc_clip_seconds: float,
    covariance_assumption: CovarianceAssumption = "conservative_upper_bound",
    ratio_covariance: torch.Tensor | None = None,
) -> TemporalUncertaintyRevision:
    """Replay corrected uncertainty from a frozen historical model output."""

    revision = propagate_temporal_uncertainty(
        output.pair_log_height_ratio,
        output.log_ratio_log_variance,
        output.diagnostics["pair_known"].bool(),
        pair_delta_t_s,
        current_pair_weight=current_pair_weight,
        risk_thresholds_s=risk_thresholds_s,
        ttc_clip_seconds=ttc_clip_seconds,
        covariance_assumption=covariance_assumption,
        ratio_covariance=ratio_covariance,
    )
    if not torch.allclose(revision.inverse_ttc_mean, output.inverse_ttc_mean, rtol=1e-5, atol=1e-7):
        raise ValueError("replay controls do not reproduce the historical point estimate")
    if not torch.allclose(
        revision.ttc_mean_seconds,
        output.ttc_mean_seconds,
        rtol=1.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError("ttc_clip_seconds does not reproduce the historical TTC point estimate")
    return revision


def replace_uncertainty_after_head_regeneration(
    output: CausalScaleTTCOutput,
    revision: TemporalUncertaintyRevision,
    *,
    downstream_artifacts_regenerated: bool = False,
) -> CausalScaleTTCOutput:
    """Replace distribution fields only after dependent calibrated heads are rebuilt.

    Standalone analysis should consume :class:`TemporalUncertaintyRevision`
    directly.  This guard prevents corrected PHASE17/risk features from being
    silently fed to historical normalizers or calibrated heads.
    """

    if not downstream_artifacts_regenerated:
        raise RuntimeError(
            "corrected uncertainty is incompatible with legacy calibrated heads; "
            "regenerate dependent features, normalizers, and heads first"
        )
    if revision.version != TEMPORAL_UNCERTAINTY_REVISION:
        raise ValueError("unknown temporal uncertainty revision version")
    if revision.inverse_ttc_mean.shape != output.inverse_ttc_mean.shape:
        raise ValueError("revision and model output batch shapes differ")
    if not torch.allclose(
        revision.inverse_ttc_mean,
        output.inverse_ttc_mean,
        rtol=1.0e-5,
        atol=1.0e-7,
    ) or not torch.allclose(
        revision.ttc_mean_seconds,
        output.ttc_mean_seconds,
        rtol=1.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError("revision point estimates do not match this model output")
    if not torch.equal(revision.known_mask, output.known_mask):
        raise ValueError("revision known mask does not match this model output")
    diagnostics = {
        **output.diagnostics,
        "uncertainty_revision_valid": revision.uncertainty_valid_mask,
        "uncertainty_revision_ttc_valid": revision.ttc_uncertainty_valid_mask,
        "uncertainty_revision_ratio_saturated": revision.ratio_saturated_mask,
    }
    return replace(
        output,
        inverse_ttc_log_variance=revision.inverse_ttc_log_variance,
        ttc_log_variance=revision.ttc_log_variance,
        collision_logits=revision.collision_logits,
        diagnostics=diagnostics,
    )


__all__ = [
    "TEMPORAL_UNCERTAINTY_REVISION",
    "TemporalUncertaintyRevision",
    "propagate_temporal_uncertainty",
    "replace_uncertainty_after_head_regeneration",
    "revise_causal_scale_output",
]
