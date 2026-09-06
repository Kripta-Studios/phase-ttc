"""Exact image-scale transport for constant relative depth velocity.

This is an inductive model, not a universal identity for every real event patch.
It is NOT constant image divergence over the entire observation interval.
"""

from __future__ import annotations

import torch
from torch import Tensor


def relative_seconds(timestamps_us: Tensor, anchor_us: Tensor) -> Tensor:
    """Subtract in int64 FIRST. Never cast absolute microseconds to float32."""
    if timestamps_us.dtype != torch.int64 or anchor_us.dtype != torch.int64:
        raise TypeError("timestamp and anchor must be int64")
    return (timestamps_us - anchor_us).to(torch.float64) * 1e-6


def scale_between(q: Tensor, t_a: Tensor, t_b: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    """h(b)/h(a)=(1-q*t_a)/(1-q*t_b); q is inverse TTC at the SAME anchor.

    Broadcast inputs. Invalid hypotheses have a harmless scale of one and an
    explicit false mask. Invalid values must not enter geometric logits as evidence.
    """
    if any(not bool(torch.isfinite(x).all()) for x in (q, t_a, t_b)):
        raise ValueError("nonfinite projective arguments")
    da, db = 1 - q * t_a, 1 - q * t_b
    valid = (da > eps) & (db > eps)
    safe = da / torch.where(valid, db, torch.ones_like(db))
    return torch.where(valid, safe, torch.ones_like(safe)), valid


def reanchor(q: Tensor, elapsed: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    """q(t+elapsed)=q(t)/(1-q(t)*elapsed), before the modeled collision."""
    den = 1 - q * elapsed
    valid = den > eps
    return torch.where(valid, q / den.clamp_min(eps), torch.zeros_like(q)), valid


def pixel_centers(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """[H,W,2] x,y centers for grid_sample(align_corners=False)."""
    if min(height, width) <= 1:
        raise ValueError("spatial dimensions must exceed one")
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * 2 / width - 1
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * 2 / height - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], -1)
