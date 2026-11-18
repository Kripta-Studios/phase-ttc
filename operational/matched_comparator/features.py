"""Garl-TTC per-query features (R) and the stacked-head designs of the matched arms (numpy only).

Garl-TTC emits two visible heights ``(h0, h1)`` and ``T = 0.1 / (1 - h0/h1)``; its benchmark
phase is therefore exactly ``phi = -log(1 - 0.1/T) = -log(h0/h1)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

LEARNED_PHASE_CLIP = 0.5  # same clip as the geometry vector G (phase_geometry.geometry)
PHASE_SCALE = 0.03
HEIGHT_REFERENCE = 128.0  # Garl ROI side in pixels
LOG_HEIGHT_CLIP = 3.0
R_FIELDS = ("phi_garl", "log_height", "garl_valid")
ARM_INPUTS = {"R": "r", "GEO_R": "g+r"}


def garl_phase(heights: NDArray[Any]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Exact benchmark phase ``-log(h0/h1)`` and its validity (finite, both heights > 0)."""
    h = np.asarray(heights, dtype=np.float64).reshape(-1, 2)
    valid = np.isfinite(h).all(axis=1) & (h[:, 0] > 0) & (h[:, 1] > 0)
    phase = np.zeros(len(h))
    phase[valid] = -np.log(h[valid, 0] / h[valid, 1])
    return phase, valid


def garl_inputs(heights: NDArray[Any]) -> NDArray[np.float32]:
    """Pre-registered R vector: clipped phase / 0.03, clipped log(h1/128), validity flag."""
    h = np.asarray(heights, dtype=np.float64).reshape(-1, 2)
    phase, valid = garl_phase(h)
    log_height = np.zeros(len(h))
    log_height[valid] = np.clip(
        np.log(h[valid, 1] / HEIGHT_REFERENCE), -LOG_HEIGHT_CLIP, LOG_HEIGHT_CLIP
    )
    columns = [
        np.where(valid, np.clip(phase, -LEARNED_PHASE_CLIP, LEARNED_PHASE_CLIP), 0.0) / PHASE_SCALE,
        log_height,
        valid.astype(np.float64),
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def garl_anchor(heights: NDArray[Any], phi_area: NDArray[Any]) -> NDArray[np.float32]:
    """Clipped Garl phase where valid, otherwise the clipped geometric area phase."""
    phase, valid = garl_phase(heights)
    fallback = np.clip(
        np.asarray(phi_area, dtype=np.float64), -LEARNED_PHASE_CLIP, LEARNED_PHASE_CLIP
    )
    clipped = np.clip(phase, -LEARNED_PHASE_CLIP, LEARNED_PHASE_CLIP)
    return np.where(valid, clipped, fallback).astype(np.float32)


def design(
    arm: str, geometry_g: NDArray[Any], heights: NDArray[Any], phi_area: NDArray[Any]
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """(inputs, anchor phase, zero width base) of one Garl arm (``R`` or ``GEO_R``)."""
    r = garl_inputs(heights)
    if arm == "R":
        x = r
    elif arm == "GEO_R":
        x = np.concatenate((np.asarray(geometry_g, dtype=np.float32), r), axis=1)
    else:
        raise ValueError(f"Unknown Garl arm {arm}")
    anchor = garl_anchor(heights, phi_area)
    return x.astype(np.float32), anchor, np.zeros((len(x), 2), dtype=np.float32)


def release_ttc(heights: NDArray[Any], delta: float = 0.1) -> NDArray[np.float64]:
    """Release conversion ``dT / (1 - h0/h1)`` without clipping (non-finite kept)."""
    h = np.asarray(heights, dtype=np.float64).reshape(-1, 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return delta / (1.0 - h[:, 0] / h[:, 1])


__all__ = [
    "ARM_INPUTS",
    "R_FIELDS",
    "design",
    "garl_anchor",
    "garl_inputs",
    "garl_phase",
    "release_ttc",
    "s8_rows",
]


def s8_rows(
    sequence_id: NDArray[Any],
    track_id: NDArray[Any],
    timestamp_us: NDArray[Any],
    sample_token: NDArray[Any],
    step: int = 8,
) -> NDArray[np.int64]:
    """Amendment-1 subsample: rank within (sequence, track) by (time, token) congruent 0 mod 8."""
    order = np.lexsort(
        (
            np.asarray(sample_token).astype(str),
            np.asarray(timestamp_us, dtype=np.int64),
            np.asarray(track_id).astype(str),
            np.asarray(sequence_id).astype(str),
        )
    )
    sequences = np.char.add(np.asarray(sequence_id).astype(str)[order], "|")
    keys = np.char.add(sequences, np.asarray(track_id).astype(str)[order])
    new_group = np.r_[True, keys[1:] != keys[:-1]]
    starts = np.maximum.accumulate(np.where(new_group, np.arange(len(order)), 0))
    rank = np.arange(len(order)) - starts
    return np.sort(order[rank % step == 0]).astype(np.int64)
