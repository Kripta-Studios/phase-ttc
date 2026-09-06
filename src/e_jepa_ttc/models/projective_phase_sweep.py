"""Differentiable scale/translation evidence, reference implementation.

A single translation is shared across source pixels for each hypothesis.
Allowing each pixel to pick an arbitrary translation BEFORE spatial aggregation
would erase the scale constraint. The resulting scores are features, not TTC.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from .projective_geometry_v11 import pixel_centers


def sample_correlations(correlation: Tensor, grid: Tensor) -> Tensor:
    """Bilinear zero-padded sampling with a deterministic CUDA input backward.

    Equivalent to grid_sample for single-channel correlation images with
    align_corners=False. CUDA grid_sample backward rejects strict deterministic
    mode. Gather backward supports it, retaining connected feature gradients.
    This routine is also differentiable in grid coordinates away from cell edges.
    """
    if correlation.ndim != 4 or correlation.shape[1] != 1:
        raise ValueError("correlation images must be N1HW")
    if grid.ndim != 4 or grid.shape[0] != correlation.shape[0] or grid.shape[-1] != 2:
        raise ValueError("sampling grid must be NXY2")
    n, _, height, width = correlation.shape
    coordinates = grid.reshape(n, -1, 2)
    x = ((coordinates[..., 0] + 1) * width - 1) / 2
    y = ((coordinates[..., 1] + 1) * height - 1) / 2
    left, top = x.floor(), y.floor()
    dx, dy = x - left, y - top
    result = torch.zeros_like(x)
    flat = correlation.flatten(1)
    for ix, iy, mass in (
        (left, top, (1 - dx) * (1 - dy)),
        (left + 1, top, dx * (1 - dy)),
        (left, top + 1, (1 - dx) * dy),
        (left + 1, top + 1, dx * dy),
    ):
        inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        index = iy.clamp(0, height - 1).long() * width + ix.clamp(0, width - 1).long()
        result = result + flat.gather(1, index) * (mass * inside)
    return result.reshape(n, 1, grid.shape[1], grid.shape[2])


def translations(
    *,
    radius: int = 6,
    step: float = 0.0625,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if radius < 0 or step <= 0:
        raise ValueError("nonnegative radius and positive step required")
    v = torch.arange(-radius, radius + 1, device=device, dtype=dtype) * step
    y, x = torch.meshgrid(v, v, indexing="ij")
    return torch.stack([x.flatten(), y.flatten()], -1)


def sweep_scores(
    source: Tensor,
    target: Tensor,
    scales: Tensor,
    displacement: Tensor,
    weights: Tensor | None = None,
    *,
    chunk: int = 8,
    min_overlap: float = 0.60,
    checkpoint_chunks: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return [B,K,D] cosine scores and weighted overlap.

    Feature maps [B,C,H,W], scales [B,K], displacement [D,2] in the SAME normalized
    common-ROI coordinates. Source weights are independent of the scale hypothesis.
    The source-to-target correlation matrix is computed once; interpolation samples
    one target-correlation image per source pixel. All gradients remain connected.

    Production code may optimize this kernel only after value/gradient equivalence.
    This function leaves invalid-scale masking to the projective caller.
    """
    if source.ndim != 4 or source.shape != target.shape:
        raise ValueError("source/target must have identical BCHW shape")
    b, c, h, w = source.shape
    if scales.ndim != 2 or scales.shape[0] != b or not bool((scales > 0).all()):
        raise ValueError("positive [B,K] scales required")
    if displacement.ndim != 2 or displacement.shape[-1] != 2 or len(displacement) == 0:
        raise ValueError("displacements must be nonempty [D,2]")
    if not 0 < min_overlap <= 1 or chunk < 1:
        raise ValueError("invalid overlap/chunk")
    for value in (source, target, scales, displacement):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite sweep input")
    n = h * w
    if weights is None:
        mass = source.new_full((b, n), 1 / n)
    else:
        if (
            weights.shape != (b, h, w)
            or not bool(torch.isfinite(weights).all())
            or bool((weights < 0).any())
        ):
            raise ValueError("weights must be nonnegative finite BHW")
        mass = weights.flatten(1)
        if bool((mass.sum(-1) <= 0).any()):
            raise ValueError("each source needs positive mass")
        mass = mass / mass.sum(-1, keepdim=True)
    f = functional.normalize(source.flatten(2), dim=1, eps=1e-6)
    g = functional.normalize(target.flatten(2), dim=1, eps=1e-6)
    corr = torch.einsum("bcp,bcq->bpq", f, g).reshape(b * n, 1, h, w)
    centers = pixel_centers(h, w, device=source.device, dtype=source.dtype).reshape(n, 2)
    k, d = scales.shape[1], displacement.shape[0]

    def one_chunk(
        correlation: Tensor, partial_scales: Tensor, source_mass: Tensor
    ) -> tuple[Tensor, Tensor]:
        count = partial_scales.shape[1]
        grid = (
            partial_scales[:, :, None, None, None] * centers[None, None, None, :, :]
            + displacement[None, None, :, None, :]
        )  # B,k,D,P,2
        valid = (grid[..., 0].abs() <= (1 - 1 / w + 1e-6)) & (
            grid[..., 1].abs() <= (1 - 1 / h + 1e-6)
        )
        sampling = grid.permute(0, 3, 1, 2, 4).reshape(b * n, count * d, 1, 2)
        values = sample_correlations(correlation, sampling)
        values = values.reshape(b, n, count, d).permute(0, 2, 3, 1)
        support = source_mass[:, None, None, :] * valid.to(source_mass.dtype)
        overlap = support.sum(-1)
        scores = (values * support).sum(-1) / overlap.clamp_min(1e-8)
        # Finite sentinel; the caller must consume the validity mask too.
        scores = torch.where(overlap >= min_overlap, scores, torch.full_like(scores, -2.0))
        return scores, overlap

    outputs, overlaps = [], []
    for first in range(0, k, chunk):
        partial = scales[:, first : first + chunk]
        if checkpoint_chunks and torch.is_grad_enabled():
            scores, overlap = cast(
                tuple[Tensor, Tensor],
                checkpoint(
                    one_chunk, corr, partial, mass, use_reentrant=False, preserve_rng_state=False
                ),
            )
        else:
            scores, overlap = one_chunk(corr, partial, mass)
        outputs.append(scores)
        overlaps.append(overlap)
    return torch.cat(outputs, 1), torch.cat(overlaps, 1)


def marginalize_translation(
    scores: Tensor, overlap: Tensor, temperature: float = 0.10, min_overlap: float = 0.60
) -> tuple[Tensor, Tensor]:
    """Uniform PRIOR over valid translation cells; remove count-of-valid-cell bias."""
    if scores.shape != overlap.shape or scores.ndim != 3 or temperature <= 0:
        raise ValueError("invalid score/overlap/temperature")
    valid = overlap >= min_overlap
    count = valid.sum(-1)
    logits = (scores / temperature).masked_fill(~valid, -1e4)
    evidence = temperature * (torch.logsumexp(logits, -1) - count.clamp_min(1).log())
    return torch.where(count > 0, evidence, torch.zeros_like(evidence)), count > 0
