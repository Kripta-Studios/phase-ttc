"""Exact candidate accumulators for causal detector event voxels.

This module is an isolated S4 performance experiment.  It does not replace the
frozen implementation in :mod:`e_jepa_ttc.detection.event_data`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from e_jepa_ttc.detection.event_data import (
    SOURCE_SIZE,
    CausalEventConfig,
    resize_event_coordinates_pixel_center,
)
from e_jepa_ttc.representations.voxel_grid import robust_normalize

AccumulatorKind = Literal["two_bincount_reference", "single_bincount_candidate"]
FLOAT64_EXACT_INTEGER_MAX = 2**53
INT64_MAX = np.iinfo(np.int64).max


@dataclass(frozen=True)
class EventAccumulationResult:
    """Raw exact numerators and the resulting normalized FP32 voxel."""

    raw_numerator: np.ndarray
    voxel: np.ndarray
    event_count: int
    nonempty_chunk_count: int
    maximum_chunk_events_observed: int
    raw_numerator_sha256_int64_le: str
    voxel_sha256_float32_le: str
    accumulator: AccumulatorKind


def _array_sha256(value: np.ndarray, dtype: str) -> str:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def validate_exact_accumulation_bounds(*, event_count: int, window_us: int) -> None:
    """Fail closed unless weighted bincount and final int64 sums are exact."""
    if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 0:
        raise ValueError("event_count must be a nonnegative integer")
    if isinstance(window_us, bool) or not isinstance(window_us, int) or window_us <= 0:
        raise ValueError("window_us must be a positive integer")
    worst_sum = event_count * window_us
    if worst_sum > FLOAT64_EXACT_INTEGER_MAX:
        raise OverflowError("Per-chunk weighted sum can exceed exact float64 integer range")
    if worst_sum > INT64_MAX:
        raise OverflowError("Per-chunk weighted sum can exceed int64 range")


def _checked_add(accumulator: np.ndarray, counts_float64: np.ndarray) -> None:
    if accumulator.dtype != np.int64 or accumulator.ndim != 1:
        raise ValueError("accumulator must be a flat int64 array")
    if counts_float64.shape != accumulator.shape or counts_float64.dtype != np.float64:
        raise ValueError("bincount output must be aligned float64")
    if not np.all(np.isfinite(counts_float64)) or np.any(counts_float64 < 0):
        raise OverflowError("bincount produced invalid weighted counts")
    counts = counts_float64.astype(np.int64)
    if not np.array_equal(counts.astype(np.float64), counts_float64):
        raise OverflowError("bincount weighted counts are not exact integers")
    if np.any(accumulator > INT64_MAX - counts):
        raise OverflowError("Cross-chunk event accumulation would overflow int64")
    accumulator += counts


def accumulate_linear_chunk(
    accumulator: np.ndarray,
    *,
    lower_flat: np.ndarray,
    upper_flat: np.ndarray,
    upper_remainder: np.ndarray,
    window_us: int,
    kind: AccumulatorKind,
) -> None:
    """Add one temporally interpolated chunk with explicit exactness bounds."""
    lower = np.asarray(lower_flat)
    upper = np.asarray(upper_flat)
    remainder = np.asarray(upper_remainder)
    if lower.ndim != 1 or lower.shape != upper.shape or lower.shape != remainder.shape:
        raise ValueError("Chunk indices and temporal remainders must be aligned vectors")
    if not all(np.issubdtype(value.dtype, np.integer) for value in (lower, upper, remainder)):
        raise ValueError("Chunk indices and temporal remainders must be integers")
    if np.any(lower < 0) or np.any(upper < 0):
        raise ValueError("Flattened voxel indices must be nonnegative")
    if np.any(lower >= accumulator.size) or np.any(upper >= accumulator.size):
        raise ValueError("Flattened voxel index exceeds the accumulator")
    if np.any(remainder < 0) or np.any(remainder >= window_us):
        raise ValueError("Temporal remainder must be in [0, window_us)")
    validate_exact_accumulation_bounds(event_count=len(lower), window_us=window_us)
    lower_i64 = lower.astype(np.int64, copy=False)
    upper_i64 = upper.astype(np.int64, copy=False)
    remainder_i64 = remainder.astype(np.int64, copy=False)
    if kind == "two_bincount_reference":
        lower_counts = np.bincount(
            lower_i64,
            weights=window_us - remainder_i64,
            minlength=accumulator.size,
        )
        _checked_add(accumulator, lower_counts)
        upper_counts = np.bincount(
            upper_i64,
            weights=remainder_i64,
            minlength=accumulator.size,
        )
        _checked_add(accumulator, upper_counts)
        return
    if kind != "single_bincount_candidate":
        raise ValueError(f"Unknown accumulator kind: {kind}")
    combined_indices = np.concatenate((lower_i64, upper_i64))
    combined_weights = np.concatenate((window_us - remainder_i64, remainder_i64))
    counts = np.bincount(
        combined_indices,
        weights=combined_weights,
        minlength=accumulator.size,
    )
    _checked_add(accumulator, counts)


def encode_causal_event_chunks_experimental(
    chunks: Iterable[Mapping[str, np.ndarray]],
    *,
    cutoff_us: int,
    config: CausalEventConfig,
    kind: AccumulatorKind = "single_bincount_candidate",
) -> EventAccumulationResult:
    """Encode the frozen S4 representation using an isolated accumulator."""
    if cutoff_us < config.window_ms * 1000:
        raise ValueError("cutoff_us cannot provide the configured causal history")
    start_us = cutoff_us - config.window_ms * 1000
    window_us = config.window_ms * 1000
    channels = config.bins_per_polarity * 2
    raw = np.zeros((channels, config.height, config.width), dtype=np.int64)
    raw_flat = raw.reshape(-1)
    pixels = config.height * config.width
    event_count = 0
    chunk_count = 0
    maximum_chunk_events = 0
    previous_timestamp: int | None = None
    for chunk in chunks:
        if set(chunk) != {"x", "y", "t", "p"}:
            raise ValueError("Event chunks must contain exactly x/y/t/p")
        x = np.asarray(chunk["x"])
        y = np.asarray(chunk["y"])
        timestamps = np.asarray(chunk["t"])
        polarity = np.asarray(chunk["p"])
        if not (x.ndim == y.ndim == timestamps.ndim == polarity.ndim == 1):
            raise ValueError("Event chunk arrays must be one-dimensional")
        if not (x.shape == y.shape == timestamps.shape == polarity.shape):
            raise ValueError("Event chunk arrays must be aligned")
        if len(x) > config.chunk_events:
            raise ValueError("Reader exceeded the configured chunk event bound")
        if len(x) == 0:
            continue
        if not np.issubdtype(timestamps.dtype, np.integer):
            raise ValueError("Event timestamps must be integer microseconds")
        timestamps = timestamps.astype(np.int64)
        if np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError("Event timestamps rolled back within a chunk")
        if previous_timestamp is not None and int(timestamps[0]) < previous_timestamp:
            raise ValueError("Event timestamps rolled back across chunks")
        if int(timestamps[0]) < start_us or int(timestamps[-1]) >= cutoff_us:
            raise ValueError("Reader returned an event outside the causal half-open window")
        if not np.issubdtype(polarity.dtype, np.integer) or not np.all(np.isin(polarity, (0, 1))):
            raise ValueError("Pinned eAP polarity must contain only raw values 0 and 1")
        resized_x, resized_y = resize_event_coordinates_pixel_center(
            x,
            y,
            source_size=SOURCE_SIZE,
            output_size=(config.width, config.height),
        )
        temporal_numerator = (timestamps - start_us) * (config.bins_per_polarity - 1)
        lower = temporal_numerator // window_us
        remainder = temporal_numerator % window_us
        upper = np.minimum(lower + 1, config.bins_per_polarity - 1)
        polarity_offset = np.where(polarity > 0, 0, config.bins_per_polarity)
        spatial = resized_y.astype(np.int64) * config.width + resized_x.astype(np.int64)
        lower_flat = (lower + polarity_offset) * pixels + spatial
        upper_flat = (upper + polarity_offset) * pixels + spatial
        accumulate_linear_chunk(
            raw_flat,
            lower_flat=lower_flat,
            upper_flat=upper_flat,
            upper_remainder=remainder,
            window_us=window_us,
            kind=kind,
        )
        event_count += len(x)
        chunk_count += 1
        maximum_chunk_events = max(maximum_chunk_events, len(x))
        previous_timestamp = int(timestamps[-1])
    raw_float = raw.astype(np.float64) / window_us
    voxel = robust_normalize(raw_float) if config.normalize else raw_float.astype(np.float32)
    return EventAccumulationResult(
        raw_numerator=raw,
        voxel=voxel,
        event_count=event_count,
        nonempty_chunk_count=chunk_count,
        maximum_chunk_events_observed=maximum_chunk_events,
        raw_numerator_sha256_int64_le=_array_sha256(raw, "<i8"),
        voxel_sha256_float32_le=_array_sha256(voxel, "<f4"),
        accumulator=kind,
    )
