"""Causal box-expansion geometry from the supplied 2D boxes of one object track.

Only public per-query inputs are used: ``sequence_id``, ``track_id``, ``timestamp_us``,
``frame_timestamps_us`` and ``boxes_xyxy``.  A frame is usable for a query at ``t_q`` only
if a row of the same track with ``timestamp_us <= t_q`` supplied it and the frame lies in
``[t_q - window_us, t_q]``.  No target, depth or 3D field is read.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

WINDOW_US = 600_000
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
MIN_PHASE = -math.log(2.0) + 1e-6
MAX_PHASE = 8.0
K_MIN = 10.0 * (1.0 - 2.0 * math.exp(-1e-6))
K_MAX = 10.0 * (1.0 - math.exp(-8.0))
LEARNED_PHASE_CLIP = 0.5
PHASE_SCALE = 0.03
MAX_FRAMES_NORMALIZER = 7.0
G_FIELDS = (
    "phi_area",
    "phi_height",
    "phi_width",
    "phi_area_last2",
    "n_frames",
    "span",
    "trunc_x_current",
    "trunc_y_current",
    "rms_residual",
    "log_size",
)
RAW_FIELDS = (
    "k_area",
    "k_height",
    "k_width",
    "k_area_last2",
    "phi_area",
    "phi_height",
    "phi_width",
    "phi_area_last2",
    "n_frames",
    "span_s",
    "trunc_x_current",
    "trunc_y_current",
    "trunc_x_any",
    "trunc_y_any",
    "rms_residual",
    "log_size",
    "phi_trunc_aware",
)

FloatArray = NDArray[np.float64]


def slope_to_phase(k: FloatArray | float) -> FloatArray:
    """Map a log-size slope ``k`` (1/s, positive = expanding) to the benchmark phase.

    With ``T = 1/k`` the phase ``-log(1 - 0.1/T)`` equals ``-log1p(-0.1 k)``; ``k`` is clamped
    so the phase stays inside the released ``[MIN_PHASE, MAX_PHASE]`` output support.
    """
    value = np.clip(np.asarray(k, dtype=np.float64), K_MIN, K_MAX)
    return np.clip(-np.log1p(-0.1 * value), MIN_PHASE, MAX_PHASE)


def truncation_flags(
    boxes: NDArray[Any], width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return per-box (horizontal, vertical) border-contact flags for ``[...,4]`` xyxy boxes."""
    b = np.asarray(boxes, dtype=np.float64)
    trunc_x = (b[..., 0] <= 0) | (b[..., 2] >= width - 1)
    trunc_y = (b[..., 1] <= 0) | (b[..., 3] >= height - 1)
    return trunc_x, trunc_y


def _ols_slope(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    """Least-squares slope and RMS residual (residual 0 for fewer than three points)."""
    xc = x - x.mean()
    denominator = float((xc * xc).sum())
    if denominator <= 0.0:
        return 0.0, 0.0
    slope = float((xc * (y - y.mean())).sum() / denominator)
    if len(x) < 3:
        return slope, 0.0
    residual = y - (y.mean() + slope * xc)
    return slope, float(np.sqrt((residual * residual).mean()))


def window_features(
    frame_us: Sequence[int] | NDArray[Any],
    boxes: Sequence[Sequence[float]] | NDArray[Any],
    query_us: int,
    window_us: int = WINDOW_US,
) -> dict[str, float]:
    """Geometry of the frames inside the causal window ending at ``query_us``.

    ``frame_us``/``boxes`` may contain any frames of the track; frames later than the query
    or older than the window are discarded here, so the result never depends on them.
    """
    times = np.asarray(frame_us, dtype=np.int64).reshape(-1)
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(times) != len(b):
        raise ValueError("frame times and boxes are misaligned")
    keep = (times <= query_us) & (times >= query_us - window_us)
    times, b = times[keep], b[keep]
    order = np.argsort(times, kind="stable")
    times, b = times[order], b[order]
    w = b[:, 2] - b[:, 0] + 1.0
    h = b[:, 3] - b[:, 1] + 1.0
    valid = (w > 0) & (h > 0)
    times, b, w, h = times[valid], b[valid], w[valid], h[valid]
    n = len(times)
    trunc_x, trunc_y = truncation_flags(b)
    out = {name: 0.0 for name in RAW_FIELDS}
    out["n_frames"] = float(n)
    if n == 0:
        return out
    out["trunc_x_current"] = float(trunc_x[-1])
    out["trunc_y_current"] = float(trunc_y[-1])
    out["trunc_x_any"] = float(trunc_x.any())
    out["trunc_y_any"] = float(trunc_y.any())
    out["log_size"] = float(np.log(np.sqrt(w[-1] * h[-1]) / IMAGE_HEIGHT))
    if n >= 2 and times[-1] > times[0]:
        x = (times - query_us).astype(np.float64) * 1e-6
        log_area = 0.5 * (np.log(w) + np.log(h))
        out["k_area"], out["rms_residual"] = _ols_slope(x, log_area)
        out["k_height"], _ = _ols_slope(x, np.log(h))
        out["k_width"], _ = _ols_slope(x, np.log(w))
        dt = x[-1] - x[-2]
        out["k_area_last2"] = float((log_area[-1] - log_area[-2]) / dt) if dt > 0 else 0.0
        out["span_s"] = float(x[-1] - x[0])
    for name in ("area", "height", "width", "area_last2"):
        out[f"phi_{name}"] = float(slope_to_phase(out[f"k_{name}"]))
    if out["trunc_x_any"] and not out["trunc_y_any"]:
        out["phi_trunc_aware"] = out["phi_height"]
    elif out["trunc_y_any"] and not out["trunc_x_any"]:
        out["phi_trunc_aware"] = out["phi_width"]
    else:
        out["phi_trunc_aware"] = out["phi_area"]
    return out


@dataclass(frozen=True)
class TrackPool:
    """All frames of one track with the earliest row time that supplied each frame."""

    frame_us: NDArray[np.int64]
    available_us: NDArray[np.int64]
    boxes: NDArray[np.float64]
    disagreements: int


def build_track_pool(
    row_us: Sequence[int] | NDArray[Any],
    frame_us: Sequence[Sequence[int]] | NDArray[Any],
    boxes: Sequence[Sequence[Sequence[float]]] | NDArray[Any],
) -> TrackPool:
    """Merge the per-row frame lists of one track.

    A frame's box comes from the earliest row (by row time, then input order) that supplied
    it; that row time is the frame's availability time.  Conflicting boxes are counted.
    """
    rows = np.asarray(row_us, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    seen: dict[int, tuple[int, NDArray[np.float64]]] = {}
    disagreements = 0
    for index in order:
        for time, box in zip(frame_us[index], boxes[index], strict=True):
            if int(time) > int(rows[index]):
                raise ValueError("row supplies a frame later than its own timestamp")
            key = int(time)
            value = np.asarray(box, dtype=np.float64)
            if key in seen:
                if not np.array_equal(seen[key][1], value):
                    disagreements += 1
                continue
            seen[key] = (int(rows[index]), value)
    keys = sorted(seen)
    return TrackPool(
        frame_us=np.asarray(keys, dtype=np.int64),
        available_us=np.asarray([seen[k][0] for k in keys], dtype=np.int64),
        boxes=np.asarray([seen[k][1] for k in keys], dtype=np.float64).reshape(-1, 4),
        disagreements=disagreements,
    )


def query_features(pool: TrackPool, query_us: int, window_us: int = WINDOW_US) -> dict[str, float]:
    """Causal geometry for one query: frames available (supplied) no later than the query."""
    available = pool.available_us <= query_us
    return window_features(pool.frame_us[available], pool.boxes[available], query_us, window_us)


def compute_table(
    sequence_id: Sequence[str],
    track_id: Sequence[str],
    timestamp_us: Sequence[int] | NDArray[Any],
    frame_us: Sequence[Sequence[int]],
    boxes: Sequence[Sequence[Sequence[float]]],
    window_us: int = WINDOW_US,
) -> tuple[NDArray[np.float64], int]:
    """Raw geometry ``[N, len(RAW_FIELDS)]`` for aligned query rows and the disagreement count."""
    n = len(timestamp_us)
    times = np.asarray(timestamp_us, dtype=np.int64)
    keys = [f"{s}\x00{t}" for s, t in zip(sequence_id, track_id, strict=True)]
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    table = np.zeros((n, len(RAW_FIELDS)), dtype=np.float64)
    disagreements = 0
    for members in groups.values():
        pool = build_track_pool(
            times[members], [frame_us[i] for i in members], [boxes[i] for i in members]
        )
        disagreements += pool.disagreements
        for index in members:
            values = query_features(pool, int(times[index]), window_us)
            table[index] = [values[name] for name in RAW_FIELDS]
    return table, disagreements


def learned_inputs(raw: NDArray[Any]) -> NDArray[np.float32]:
    """Project the raw table to the pre-registered 10-field learned vector G."""
    r = {name: np.asarray(raw)[:, i] for i, name in enumerate(RAW_FIELDS)}
    clip = LEARNED_PHASE_CLIP
    columns = [
        np.clip(r["phi_area"], -clip, clip) / PHASE_SCALE,
        np.clip(r["phi_height"], -clip, clip) / PHASE_SCALE,
        np.clip(r["phi_width"], -clip, clip) / PHASE_SCALE,
        np.clip(r["phi_area_last2"], -clip, clip) / PHASE_SCALE,
        r["n_frames"] / MAX_FRAMES_NORMALIZER,
        r["span_s"] / (WINDOW_US * 1e-6),
        r["trunc_x_current"],
        r["trunc_y_current"],
        100.0 * r["rms_residual"],
        r["log_size"],
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def raw_column(raw: NDArray[Any], name: str) -> NDArray[np.float64]:
    """Return one named raw geometry column."""
    return np.asarray(raw, dtype=np.float64)[:, RAW_FIELDS.index(name)]


__all__ = [
    "G_FIELDS",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "RAW_FIELDS",
    "WINDOW_US",
    "TrackPool",
    "build_track_pool",
    "compute_table",
    "learned_inputs",
    "query_features",
    "raw_column",
    "slope_to_phase",
    "truncation_flags",
    "window_features",
]
