"""E128-T8 integer counts, physical support and TRAIN-only four-context views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from e_jepa_ttc.data.raw_temporal_cache import _signed_polarity

COUNT_SHAPE = (2, 8, 2, 128, 128)
POLARITY_ORDER = ("negative", "positive")


@dataclass(frozen=True)
class Window128:
    """One physical window and the immutable common detection-assisted ROI."""

    start_us: int
    end_us: int
    roi: tuple[float, float, float, float]
    sensor_size: tuple[int, int]
    x_offset: float = 5.0
    polarity_encoding: Literal["zero_one", "signed"] = "zero_one"

    def __post_init__(self) -> None:
        if not isinstance(self.start_us, int) or not isinstance(self.end_us, int):
            raise ValueError("window endpoints must be integer microseconds")
        if self.end_us - self.start_us < 8:
            raise ValueError("E128-T8 requires eight positive-duration bins")
        x0, y0, x1, y1 = self.roi
        if (
            not np.isfinite((*self.roi, self.x_offset)).all()
            or x1 <= x0
            or y1 <= y0
            or not np.isclose(x1 - x0, y1 - y0, rtol=0, atol=1e-6)
            or min(self.sensor_size) <= 0
        ):
            raise ValueError("finite square ROI and positive sensor size required")

    def edges(self) -> np.ndarray:
        """Generate the exact eight-bin integer floor boundaries."""
        return np.asarray(
            [self.start_us + (self.end_us - self.start_us) * i // 8 for i in range(9)],
            dtype=np.int64,
        )

    def support(self) -> np.ndarray:
        """Register ROI pixel-center sensor coverage, shared by every temporal bin."""
        x0, y0, x1, y1 = self.roi
        x = x0 + (np.arange(128) + 0.5) * (x1 - x0) / 128 - self.x_offset
        y = y0 + (np.arange(128) + 0.5) * (y1 - y0) / 128
        return ((y >= 0) & (y < self.sensor_size[1]))[:, None] & (
            (x >= 0) & (x < self.sensor_size[0])
        )[None, :]


def rasterize128(
    x: np.ndarray,
    y: np.ndarray,
    timestamps_us: np.ndarray,
    polarity: np.ndarray,
    window: Window128,
) -> tuple[np.ndarray, dict[str, int]]:
    """Rasterize a monotonic half-open slice without dropping valid observations."""
    arrays = [np.asarray(value) for value in (x, y, timestamps_us, polarity)]
    if any(value.ndim != 1 for value in arrays) or len({len(value) for value in arrays}) != 1:
        raise ValueError("event fields must be aligned vectors")
    if arrays[2].dtype.kind not in "iu":
        raise ValueError("event timestamps must be integer microseconds")
    if arrays[2].size and int(arrays[2].max()) > np.iinfo(np.int64).max:
        raise ValueError("timestamp exceeds int64 clock contract")
    time = arrays[2].astype(np.int64, copy=False)
    if np.any(time < window.start_us) or np.any(time >= window.end_us) or np.any(np.diff(time) < 0):
        raise ValueError("event slice violates monotonic half-open window")
    signed = _signed_polarity(arrays[3], window.polarity_encoding)
    px, py = arrays[0].astype(np.float64), arrays[1].astype(np.float64)
    sensor = (
        np.isfinite(px)
        & np.isfinite(py)
        & (px >= 0)
        & (px < window.sensor_size[0])
        & (py >= 0)
        & (py < window.sensor_size[1])
    )
    x0, y0, x1, y1 = window.roi
    inside = (
        sensor & (px + window.x_offset >= x0) & (px + window.x_offset < x1) & (py >= y0) & (py < y1)
    )
    gx = np.floor((px[inside] + window.x_offset - x0) * 128 / (x1 - x0)).astype(np.int64)
    gy = np.floor((py[inside] - y0) * 128 / (y1 - y0)).astype(np.int64)
    bins = np.searchsorted(window.edges(), time[inside], side="right") - 1
    polarity_index = (signed[inside] > 0).astype(np.int64)
    flat = ((bins * 2 + polarity_index) * 128 + gy) * 128 + gx
    counts = np.bincount(flat, minlength=8 * 2 * 128 * 128).reshape(8, 2, 128, 128)
    if counts.max(initial=0) > np.iinfo(np.uint32).max:
        raise OverflowError("E128-T8 count exceeds uint32")
    return counts.astype(np.uint32), {
        "raw_events": len(time),
        "outside_sensor": int((~sensor).sum()),
        "outside_roi": int((sensor & ~inside).sum()),
        "roi_events": int(inside.sum()),
    }


def checked_count_cast(counts: np.ndarray, dtype: str = "uint16") -> np.ndarray:
    """Never saturate; the publication owner must upgrade the whole cache version."""
    if dtype not in ("uint16", "uint32") or counts.dtype.kind != "u":
        raise ValueError("unsigned count storage required")
    if int(counts.max(initial=0)) > np.iinfo(np.dtype(dtype)).max:
        raise OverflowError("whole cache version requires uint32 upgrade")
    return counts.astype(dtype)


def log_rates(counts: np.ndarray, durations_s: np.ndarray) -> np.ndarray:
    """Apply the fixed float32 rate transform, including empty activity bins."""
    if counts.shape != COUNT_SHAPE or counts.dtype.kind != "u":
        raise ValueError("E128-T8 unsigned count shape mismatch")
    durations = np.asarray(durations_s, dtype=np.float32)
    if durations.shape != (2, 8) or not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("positive physical duration required for every bin")
    return np.log1p(counts.astype(np.float32) / durations[:, :, None, None, None])


@dataclass(frozen=True)
class RateNormalization:
    """Two polarity moments and exact training-source binding, never outer pooled."""

    mean: tuple[float, float]
    std: tuple[float, float]
    training_view_sha256: str
    valid_values_per_polarity: int

    def __post_init__(self) -> None:
        if not np.isfinite((*self.mean, *self.std)).all() or min(self.std) < 1e-6:
            raise ValueError("invalid TRAIN rate normalization")
        if len(self.training_view_sha256) != 64 or self.valid_values_per_polarity <= 0:
            raise ValueError("normalization must bind an observed TRAIN view")


class RateMoments:
    """Streaming float64 pooled population moments over nonmissing sensor support."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = np.zeros(2, dtype=np.float64)
        self.m2 = np.zeros(2, dtype=np.float64)

    def add(self, transformed: np.ndarray, support: np.ndarray) -> None:
        """Merge one TRAIN row; callers must enforce its fold and ancestor role."""
        if transformed.shape != COUNT_SHAPE or support.shape != (128, 128) or support.dtype != bool:
            raise ValueError("rate/support schema mismatch")
        if not np.isfinite(transformed).all():
            raise ValueError("nonfinite transformed counts")
        count = 16 * int(support.sum())
        if not count:
            return
        values = np.stack([transformed[:, :, p, support].reshape(-1) for p in range(2)]).astype(
            np.float64
        )
        mean = values.mean(axis=1)
        m2 = ((values - mean[:, None]) ** 2).sum(axis=1)
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta**2 * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total

    def finish(self, training_view_sha256: str) -> RateNormalization:
        """Freeze the population standard deviation with the prescribed floor."""
        if not self.count:
            raise ValueError("no observed TRAIN support for normalization")
        std = np.sqrt(np.maximum(self.m2 / self.count, 0)).clip(min=1e-6)
        return RateNormalization(
            (float(self.mean[0]), float(self.mean[1])),
            (float(std[0]), float(std[1])),
            training_view_sha256,
            self.count,
        )


@dataclass(frozen=True, slots=True)
class ArchitectureInput:
    """Neural inputs only; targets, identities, individual box heights travel separately."""

    contexts: Tensor
    times: Tensor
    sensor_support: Tensor


def four_contexts(
    counts: np.ndarray,
    durations_s: np.ndarray,
    support: np.ndarray,
    midpoint_twice_relative_us: np.ndarray,
    normalization: RateNormalization,
) -> ArchitectureInput:
    """Materialize a single four-context row without loading a full cache."""
    if support.shape != (128, 128) or support.dtype != bool:
        raise ValueError("expected boolean common sensor support")
    integer_times = np.asarray(midpoint_twice_relative_us)
    if (
        integer_times.shape != (4,)
        or integer_times.dtype.kind != "i"
        or np.any(np.diff(integer_times) <= 0)
    ):
        raise ValueError("four chronological integer relative midpoints required")
    values = log_rates(counts, durations_s)
    mean = np.asarray(normalization.mean, dtype=np.float32)[None, None, :, None, None]
    std = np.asarray(normalization.std, dtype=np.float32)[None, None, :, None, None]
    normalized = (values - mean) / std
    normalized *= support[None, None, None]
    return ArchitectureInput(
        torch.from_numpy(normalized.reshape(4, 8, 128, 128)),
        torch.from_numpy((integer_times.astype(np.float64) / 2_000_000).astype(np.float32)),
        torch.from_numpy(support.copy()),
    )
