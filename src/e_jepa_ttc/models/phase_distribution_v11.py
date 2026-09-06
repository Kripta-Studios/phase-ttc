"""Reference signed benchmark-phase distribution. No repository dependencies.

The distribution is piecewise uniform between ordered bin edges. Its point
prediction is a median in PHASE space, not an average of signed TTC values.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def phase_edges(
    *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Fixed broad support, with fine resolution over the development regime.

    Central spacing is 0.00375. Out-of-support target mass MUST be reported; no
    rows may be discarded because of the head's representational support.
    """
    left = torch.tensor([-math.log(2) + 1e-6, -0.50, -0.30, -0.20], device=device, dtype=dtype)
    middle = torch.linspace(-0.12, 0.24, 97, device=device, dtype=dtype)
    right = torch.tensor([0.32, 0.50, 0.80, 1.2, 2.0, 3.0, 4.0, 8.0], device=device, dtype=dtype)
    return torch.cat([left, middle, right])


def validate_edges(edges: Tensor) -> None:
    if edges.ndim != 1 or edges.numel() < 2 or not bool(torch.isfinite(edges).all()):
        raise ValueError("edges must be a finite vector with at least two entries")
    if not bool((edges[1:] > edges[:-1]).all()):
        raise ValueError("edges must be strictly increasing")


def ttc_to_phase(ttc: Tensor, delta: float = 0.1) -> Tensor:
    if delta <= 0 or not bool(torch.isfinite(ttc).all()):
        raise ValueError("finite TTC and a positive metric delta are required")
    if not bool(((ttc < 0) | (ttc > delta)).all()):
        raise ValueError("signed TTC domain is T<0 or T>metric_delta")
    return -torch.log1p(-delta / ttc)


def phase_to_inverse_ttc(phase: Tensor, delta: float = 0.1) -> Tensor:
    if delta <= 0 or not bool(torch.isfinite(phase).all()):
        raise ValueError("finite phase and positive metric delta required")
    return -torch.expm1(-phase) / delta


def phase_to_finite_ttc(phase: Tensor, delta: float = 0.1, clip: float = 60.0) -> Tensor:
    if clip <= delta:
        raise ValueError("clip must exceed metric delta")
    q = phase_to_inverse_ttc(phase, delta)
    # The positive tie at exactly q=0 matches the existing finite-TTC convention.
    sign = torch.where(q < 0, -torch.ones_like(q), torch.ones_like(q))
    magnitude = 1.0 / q.abs().clamp_min(1.0 / clip)
    return sign * torch.where(q.abs() <= 1.0 / clip, torch.full_like(q, clip), magnitude)


def _probabilities(logits: Tensor, edges: Tensor) -> Tensor:
    validate_edges(edges)
    if logits.ndim != 2 or logits.shape[-1] != edges.numel() - 1:
        raise ValueError("logits must be [B,len(edges)-1]")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("nonfinite distribution logits")
    return logits.softmax(-1)


def quantile(logits: Tensor, edges: Tensor, level: float = 0.5) -> Tensor:
    """Left-most piecewise-uniform distribution quantile, including within-bin interpolation."""
    if not 0 < level < 1:
        raise ValueError("quantile level must lie in (0,1)")
    p = _probabilities(logits, edges)
    cdf = p.cumsum(-1)
    index = (cdf < level).sum(-1).clamp_max(p.shape[-1] - 1)
    before = cdf - p
    row = torch.arange(len(p), device=p.device)
    fraction = (
        (level - before[row, index]) / p[row, index].clamp_min(torch.finfo(p.dtype).tiny)
    ).clamp(0, 1)
    return edges[index] + fraction * (edges[index + 1] - edges[index])


def crps(logits: Tensor, target_phase: Tensor, edges: Tensor) -> Tensor:
    """EXACT CRPS of a piecewise uniform CDF; returns a loss per row.

    Inside a bin, F(u)=a+p*u. Split the squared-CDF integral at the target,
    including its location within the bin rather than snapping it to a center.
    Targets beyond the finite support add the exact outside-support distance.
    """
    p = _probabilities(logits, edges)
    if target_phase.shape != (len(p),) or not bool(torch.isfinite(target_phase).all()):
        raise ValueError("targets must be a finite [B] vector")
    a = p.cumsum(-1) - p
    width = edges[1:] - edges[:-1]
    v = ((target_phase[:, None] - edges[:-1]) / width).clamp(0, 1)
    integral = (a - 1).square() + (a - 1) * p + p.square() / 3
    integral = integral + (2 * a - 1) * v + p * v.square()
    outside = (edges[0] - target_phase).clamp_min(0) + (target_phase - edges[-1]).clamp_min(0)
    return (integral * width).sum(-1) + outside


def benchmark_mass(sequence: Tensor, bucket: Tensor) -> Tensor:
    """Strict sequence macro with historical 0.5/0.3/0.1/0.1 bucket masses.

    Bucket IDs: crucial=0, small=1, large=2, negative=3. Missing buckets in any
    sequence are an error for THIS strict historical function. A new-data estimand
    must define and label its missing-bucket treatment separately.
    """
    if sequence.ndim != 1 or bucket.shape != sequence.shape or len(sequence) == 0:
        raise ValueError("nonempty row-aligned sequence and bucket vectors required")
    if not bool(((bucket >= 0) & (bucket <= 3)).all()):
        raise ValueError("unknown bucket ID")
    seqs = sequence.unique()
    weights = torch.zeros(len(sequence), device=sequence.device, dtype=torch.float64)
    for seq in seqs:
        for k, mass in enumerate((0.5, 0.3, 0.1, 0.1)):
            keep = (sequence == seq) & (bucket == k)
            if not bool(keep.any()):
                raise ValueError("strict historical metric requires all four buckets per sequence")
            weights[keep] = mass / len(seqs) / int(keep.sum())
    return weights


def weighted_training_loss(
    logits: Tensor, target_phase: Tensor, edges: Tensor, global_mass: Tensor, universe_rows: int
) -> Tensor:
    """Unbiased uniform-row minibatch estimate of the registered macro objective.

    L1 uses the phase of the ACTUALLY emitted, finite-clipped TTC. CRPS regularizes
    its untruncated phase distribution. Do not renormalize sampled weights per batch.
    """
    if universe_rows < len(logits) or global_mass.shape != target_phase.shape:
        raise ValueError("invalid population size or row mass")
    if bool((global_mass < 0).any()) or not bool(torch.isfinite(global_mass).all()):
        raise ValueError("invalid global mass")
    point = quantile(logits, edges)
    emitted = ttc_to_phase(phase_to_finite_ttc(point))
    per_row = ((emitted - target_phase).abs() + 0.2 * crps(logits, target_phase, edges)) / 0.03
    return (universe_rows * global_mass * per_row).mean()
