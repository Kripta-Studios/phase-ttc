"""Causal robust box geometry: per-edge clipping, saturation detection and constant-velocity
extrapolation from the last unclipped measurements (PROTOCOL.md §2-§3).

Everything here is pure numpy and reads only supplied boxes, their frame times and the frame
size.  Frames later than the query, or older than the memory horizon, never influence a result.
Under constant closing speed an image size ``s`` is proportional to ``1/Z``, so ``1/s`` is linear
in time and the expansion rate at the query is ``k = -beta/alpha`` for the least-squares line
``1/s = alpha + beta * t`` (``t`` relative to the query).  This is the causal extrapolation
``TTC(t) = TTC(t0) - (t - t0)`` from the last trustworthy (unclipped) observations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from operational.phase_geometry import geometry

FloatArray = NDArray[np.float64]

SAT_MIN_CLIPPED_EDGES = 3
SAT_MIN_COVERAGE = 0.6
EDGE_MIN_DISTANCE_FRACTION = 0.05
AGE_NORMALIZER_S = 0.6
AUG_SEED = 20261118
AUG_MAX_ZOOM = 6.0
AUG_CENTER_JITTER = 0.25

RAW_R_FIELDS = (
    "phi_robust",
    "k_robust",
    "src_clean",
    "src_free_axis",
    "src_missing",
    "age_free_s",
    "clip_left",
    "clip_right",
    "clip_top",
    "clip_bottom",
    "n_clipped",
    "coverage",
    "clip_run_age_s",
    "sat",
    "phi_edge",
    "edge_valid",
)
R_FIELDS = (
    "phi_robust",
    "src_free_axis",
    "src_missing",
    "age_free",
    "clip_left",
    "clip_right",
    "clip_top",
    "clip_bottom",
    "coverage",
    "clip_run_age",
    "sat",
    "phi_edge",
    "edge_valid",
)


@dataclass(frozen=True)
class FrameSpec:
    """Frame size and the border convention (a coordinate ``<= lo`` touches the near border)."""

    width: int = geometry.IMAGE_WIDTH
    height: int = geometry.IMAGE_HEIGHT
    lo: float = 0.0


EAP = FrameSpec(geometry.IMAGE_WIDTH, geometry.IMAGE_HEIGHT, 0.0)
FCWD = FrameSpec(1280, 720, 1.0)


def edge_flags(boxes: NDArray[Any], spec: FrameSpec = EAP) -> NDArray[np.bool_]:
    """Per-box clipped-edge flags ``[N, 4]`` in the order left, right, top, bottom."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.stack(
        (
            b[:, 0] <= spec.lo,
            b[:, 2] >= spec.width - 1,
            b[:, 1] <= spec.lo,
            b[:, 3] >= spec.height - 1,
        ),
        axis=1,
    )


def coverage(box: NDArray[Any], spec: FrameSpec = EAP) -> float:
    """Fraction of the frame covered by the box intersected with the frame."""
    b = np.asarray(box, dtype=np.float64).reshape(4)
    w = min(b[2], spec.width - 1) - max(b[0], spec.lo) + 1.0
    h = min(b[3], spec.height - 1) - max(b[1], spec.lo) + 1.0
    if w <= 0 or h <= 0:
        return 0.0
    return float(w * h / (spec.width * spec.height))


def is_saturated(flags: NDArray[Any], cover: float) -> bool:
    """Pre-registered SAT rule: >= 3 clipped edges or frame coverage > 0.6."""
    return bool(int(np.asarray(flags).sum()) >= SAT_MIN_CLIPPED_EDGES or cover > SAT_MIN_COVERAGE)


def inverse_size_rate(t_s: NDArray[Any], size: NDArray[Any]) -> float:
    """Expansion rate at ``t = 0`` from the OLS line ``1/size = alpha + beta*t``.

    Returns ``-beta/alpha`` (1/s, positive = approaching); if the extrapolated inverse size at
    the query is not positive the contact would already have happened and ``K_MAX`` is returned.
    """
    t = np.asarray(t_s, dtype=np.float64)
    u = 1.0 / np.asarray(size, dtype=np.float64)
    tc = t - t.mean()
    denominator = float((tc * tc).sum())
    if len(t) < 2 or denominator <= 0.0:
        raise ValueError("need at least two distinct times")
    beta = float((tc * (u - u.mean())).sum() / denominator)
    alpha = float(u.mean() - beta * t.mean())
    if alpha <= 0.0:
        return float(geometry.K_MAX)
    return -beta / alpha


def _ols_slope(x: FloatArray, y: FloatArray) -> float:
    xc = x - x.mean()
    denominator = float((xc * xc).sum())
    return float((xc * (y - y.mean())).sum() / denominator) if denominator > 0 else 0.0


def _phase(k: float) -> float:
    return float(geometry.slope_to_phase(k))


def robust_features(
    frame_us: Sequence[int] | NDArray[Any],
    boxes: Sequence[Sequence[float]] | NDArray[Any],
    query_us: int,
    g_raw: Mapping[str, float],
    *,
    window_us: int = geometry.WINDOW_US,
    memory_us: int = geometry.WINDOW_US,
    spec: FrameSpec = EAP,
) -> dict[str, float]:
    """Raw robust geometry (``RAW_R_FIELDS``) for one query.

    ``g_raw`` is the plain geometry of the same query (``phi_area`` and ``phi_trunc_aware`` are
    used as the clean-window value and the no-estimate fallback).  Frames later than the query
    or older than ``memory_us`` are discarded here.
    """
    times = np.asarray(frame_us, dtype=np.int64).reshape(-1)
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(times) != len(b):
        raise ValueError("frame times and boxes are misaligned")
    keep = (times <= query_us) & (times >= query_us - memory_us)
    times, b = times[keep], b[keep]
    order = np.argsort(times, kind="stable")
    times, b = times[order], b[order]
    w = b[:, 2] - b[:, 0] + 1.0
    h = b[:, 3] - b[:, 1] + 1.0
    valid = (w > 0) & (h > 0)
    times, b, w, h = times[valid], b[valid], w[valid], h[valid]
    out = {name: 0.0 for name in RAW_R_FIELDS}
    out["phi_robust"] = float(g_raw["phi_area"])
    if len(times) == 0:
        out["src_missing"] = 1.0
        out["age_free_s"] = memory_us * 1e-6
        return out
    flags = edge_flags(b, spec)
    clipped = flags.any(axis=1)
    current = flags[-1]
    out["clip_left"], out["clip_right"], out["clip_top"], out["clip_bottom"] = (
        float(v) for v in current
    )
    out["n_clipped"] = float(current.sum())
    out["coverage"] = coverage(b[-1], spec)
    out["sat"] = float(is_saturated(current, out["coverage"]))
    t_s = (times - query_us).astype(np.float64) * 1e-6
    if clipped[-1]:
        run_start = len(clipped) - 1
        while run_start > 0 and clipped[run_start - 1]:
            run_start -= 1
        out["clip_run_age_s"] = float(-t_s[run_start])
    in_window = times >= query_us - window_us
    if not clipped[in_window].any():
        out["src_clean"] = 1.0
        out["k_robust"] = 0.0
        out["phi_robust"] = float(g_raw["phi_area"])
        return out
    candidates: list[tuple[float, float]] = []
    for axis_flags, size in ((flags[:, 0] | flags[:, 1], w), (flags[:, 2] | flags[:, 3], h)):
        free = np.flatnonzero(~axis_flags)
        if len(free) < 2:
            continue
        last = times[free[-1]]
        use = free[times[free] >= last - window_us]
        if len(use) < 2 or times[use[-1]] <= times[use[0]]:
            continue
        candidates.append((float(-t_s[use[-1]]), inverse_size_rate(t_s[use], size[use])))
    if candidates:
        age = min(a for a, _ in candidates)
        rates = [k for a, k in candidates if abs(a - age) < 1e-9]
        k = float(np.mean(rates))
        out["src_free_axis"] = 1.0
        out["age_free_s"] = age
        out["k_robust"] = k
        out["phi_robust"] = _phase(k)
    else:
        out["src_missing"] = 1.0
        out["age_free_s"] = memory_us * 1e-6
        out["k_robust"] = 0.0
        out["phi_robust"] = float(g_raw["phi_trunc_aware"])
    window_rows = np.flatnonzero(in_window)
    centers = ((spec.lo + spec.width - 1) / 2.0, (spec.lo + spec.height - 1) / 2.0)
    dims = (spec.width, spec.height)
    edge_rates: list[float] = []
    for edge, (column, axis) in enumerate(((0, 0), (2, 0), (1, 1), (3, 1))):
        rows = window_rows[~flags[window_rows, edge]]
        if len(rows) < 2 or times[rows[-1]] <= times[rows[0]]:
            continue
        distance = b[rows, column] - centers[axis]
        if (np.abs(distance) < EDGE_MIN_DISTANCE_FRACTION * dims[axis]).any():
            continue
        if not (np.all(distance > 0) or np.all(distance < 0)):
            continue
        edge_rates.append(_ols_slope(t_s[rows], np.log(np.abs(distance))))
    if edge_rates:
        out["edge_valid"] = 1.0
        out["phi_edge"] = _phase(float(np.mean(edge_rates)))
    return out


def learned_r(raw: NDArray[Any]) -> NDArray[np.float32]:
    """Project a raw robust table ``[N, len(RAW_R_FIELDS)]`` to the 13 learned inputs R."""
    r = {name: np.asarray(raw, dtype=np.float64)[:, i] for i, name in enumerate(RAW_R_FIELDS)}
    clip = geometry.LEARNED_PHASE_CLIP
    columns = [
        np.clip(r["phi_robust"], -clip, clip) / geometry.PHASE_SCALE,
        r["src_free_axis"],
        r["src_missing"],
        r["age_free_s"] / AGE_NORMALIZER_S,
        r["clip_left"],
        r["clip_right"],
        r["clip_top"],
        r["clip_bottom"],
        r["coverage"],
        r["clip_run_age_s"] / AGE_NORMALIZER_S,
        r["sat"],
        np.clip(r["phi_edge"], -clip, clip) / geometry.PHASE_SCALE,
        r["edge_valid"],
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def raw_r_column(raw: NDArray[Any], name: str) -> FloatArray:
    """One named raw robust column."""
    return np.asarray(raw, dtype=np.float64)[:, RAW_R_FIELDS.index(name)]


def robust_anchor(raw: NDArray[Any]) -> NDArray[np.float32]:
    """Anchor of the GEO_ROBUST* heads: ``phi_robust`` clipped to the learned phase range."""
    clip = geometry.LEARNED_PHASE_CLIP
    return np.clip(raw_r_column(raw, "phi_robust"), -clip, clip).astype(np.float32)


# ----------------------------------------------------------------------- virtual-frame zoom


def draw_crops(current_boxes: NDArray[Any], seed: int = AUG_SEED) -> FloatArray:
    """Pre-registered virtual crops ``[N, 3]`` = (x0, y0, zoom), one per row in row order.

    Zoom ``z = exp(U(0, log 6))``; a 16:9 crop of ``1280/z x 720/z`` centred on the current box
    centre plus ``U(-0.25, 0.25)`` crop sizes per axis, shifted to lie inside the image.
    """
    b = np.asarray(current_boxes, dtype=np.float64).reshape(-1, 4)
    rng = np.random.default_rng(seed)
    n = len(b)
    zoom = np.exp(rng.uniform(0.0, math.log(AUG_MAX_ZOOM), n))
    jitter = rng.uniform(-AUG_CENTER_JITTER, AUG_CENTER_JITTER, (n, 2))
    cw, ch = EAP.width / zoom, EAP.height / zoom
    cx = 0.5 * (b[:, 0] + b[:, 2]) + jitter[:, 0] * cw
    cy = 0.5 * (b[:, 1] + b[:, 3]) + jitter[:, 1] * ch
    x0 = np.clip(cx - cw / 2, 0.0, EAP.width - cw)
    y0 = np.clip(cy - ch / 2, 0.0, EAP.height - ch)
    return np.stack((x0, y0, zoom), axis=1)


def crop_boxes(boxes: NDArray[Any], crop: NDArray[Any]) -> FloatArray:
    """Map xyxy boxes into a zoomed virtual frame, clipped to it; non-overlapping boxes are
    returned with zero width (dropped by every geometry function)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x0, y0, z = (float(v) for v in np.asarray(crop, dtype=np.float64).reshape(3))
    out = np.empty_like(b)
    out[:, [0, 2]] = (b[:, [0, 2]] - x0) * z
    out[:, [1, 3]] = (b[:, [1, 3]] - y0) * z
    outside = (
        (out[:, 2] < 0)
        | (out[:, 0] > EAP.width - 1)
        | (out[:, 3] < 0)
        | (out[:, 1] > EAP.height - 1)
    )
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0.0, EAP.width - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0.0, EAP.height - 1)
    out[outside] = (1.0, 1.0, 0.0, 0.0)
    return out


# ------------------------------------------------------------------------------- tables


def compute_tables(
    sequence_id: Sequence[str],
    track_id: Sequence[str],
    timestamp_us: Sequence[int] | NDArray[Any],
    frame_us: Sequence[Sequence[int]],
    boxes: Sequence[Sequence[Sequence[float]]],
    *,
    memories_us: Sequence[int],
    crops: NDArray[Any] | None = None,
    window_us: int = geometry.WINDOW_US,
) -> tuple[FloatArray, dict[int, FloatArray]]:
    """Plain geometry ``[N, len(geometry.RAW_FIELDS)]`` and robust tables per memory (eAP spec).

    With ``crops`` every row's whole causal track pool is mapped into that row's virtual frame
    before any geometry is computed (augmented copies).
    """
    n = len(timestamp_us)
    times = np.asarray(timestamp_us, dtype=np.int64)
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(f"{s}\x00{t}" for s, t in zip(sequence_id, track_id, strict=True)):
        groups.setdefault(key, []).append(index)
    g_table = np.zeros((n, len(geometry.RAW_FIELDS)), dtype=np.float64)
    r_tables = {m: np.zeros((n, len(RAW_R_FIELDS)), dtype=np.float64) for m in memories_us}
    for members in groups.values():
        pool = geometry.build_track_pool(
            times[members], [frame_us[i] for i in members], [boxes[i] for i in members]
        )
        for index in members:
            query = int(times[index])
            available = pool.available_us <= query
            f_us, f_boxes = pool.frame_us[available], pool.boxes[available]
            if crops is not None:
                f_boxes = crop_boxes(f_boxes, crops[index])
            g = geometry.window_features(f_us, f_boxes, query, window_us)
            g_table[index] = [g[name] for name in geometry.RAW_FIELDS]
            for memory in memories_us:
                r = robust_features(
                    f_us, f_boxes, query, g, window_us=window_us, memory_us=memory, spec=EAP
                )
                r_tables[memory][index] = [r[name] for name in RAW_R_FIELDS]
    return g_table, r_tables


__all__ = [
    "EAP",
    "FCWD",
    "RAW_R_FIELDS",
    "R_FIELDS",
    "FrameSpec",
    "compute_tables",
    "coverage",
    "crop_boxes",
    "draw_crops",
    "edge_flags",
    "inverse_size_rate",
    "is_saturated",
    "learned_r",
    "raw_r_column",
    "robust_anchor",
    "robust_features",
]
