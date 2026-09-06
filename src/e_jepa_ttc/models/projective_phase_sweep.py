"""Differentiable scale/translation evidence, reference implementation.

A single translation is shared across source pixels for each hypothesis.
Allowing each pixel to pick an arbitrary translation BEFORE spatial aggregation
would erase the scale constraint. The resulting scores are features, not TTC.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional

from .projective_geometry_v11 import pixel_centers


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
    outputs, overlaps = [], []
    for first in range(0, k, chunk):
        count = min(chunk, k - first)
        grid = (
            scales[:, first : first + count, None, None, None] * centers[None, None, None, :, :]
            + displacement[None, None, :, None, :]
        )  # B,k,D,P,2
        valid = (grid[..., 0].abs() <= (1 - 1 / w + 1e-6)) & (
            grid[..., 1].abs() <= (1 - 1 / h + 1e-6)
        )
        sampling = grid.permute(0, 3, 1, 2, 4).reshape(b * n, count * d, 1, 2)
        values = functional.grid_sample(
            corr, sampling, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        values = values.reshape(b, n, count, d).permute(0, 2, 3, 1)
        support = mass[:, None, None, :] * valid.to(mass.dtype)
        overlap = support.sum(-1)
        scores = (values * support).sum(-1) / overlap.clamp_min(1e-8)
        # Finite sentinel; the caller must consume the validity mask too.
        scores = torch.where(overlap >= min_overlap, scores, torch.full_like(scores, -2.0))
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
