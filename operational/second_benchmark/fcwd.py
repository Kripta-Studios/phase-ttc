"""Label-free FCWD queries and the causal box-geometry arms of PHASE-TTC-Geo.

Only ``bounding_box.csv`` (contractual prediction input) and the event time span of each HDF5
are read here; ``gt_ttc.csv`` is opened only by :mod:`operational.second_benchmark.score` after
the predictions are sealed.  The dataset root comes from ``PHASE_TTC_FCWD_ROOT``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from operational.phase_geometry import geometry

ENV_FCWD_ROOT = "PHASE_TTC_FCWD_ROOT"
SEQUENCES = (1, 2, 3)
HDF5_NAMES = {
    1: "2024-03-02-10-35-27.hdf5",
    2: "2024-03-02-10-37-09.hdf5",
    3: "2024-03-02-10-38-25.hdf5",
}
EVENT_T_PATH = "prophesee/event_cam_right/t"
SENSOR_WIDTH = 1280
SENSOR_HEIGHT = 720
# Cohort filter of operational/sota_eval/fcwd_inputs.build_manifest (re-implemented, not edited).
HISTORY_US = 650_000
BBOX_AVAILABILITY_MARGIN_US = 50_000
BBOX_MAX_POST_AVAILABILITY_AGE_US = 51_000
COHORT_ENDPOINT_OFFSETS_US = (-200_000, -100_000, 0)
# Addendum A1: 600 ms + half a 20 Hz frame interval.
WINDOW_US = 625_000
TTC_CAP_S = 60.0
LATENCIES_US = {"P": 0, "L50": 50_000}
RATES = (20, 10)

FloatArray = NDArray[np.float64]


def fcwd_root(environ: Mapping[str, str] | None = None) -> Path:
    """FCWD dataset root from ``PHASE_TTC_FCWD_ROOT``; fails with an actionable error."""
    env = os.environ if environ is None else environ
    value = env.get(ENV_FCWD_ROOT)
    if not value:
        raise RuntimeError(
            f"{ENV_FCWD_ROOT} is not set: point it at the FCWD root that contains "
            "inputs/sequence_{1,2,3}/*.hdf5 and sealed_labels/sequence_{1,2,3}/bounding_box.csv"
        )
    root = Path(value)
    if not (root / "inputs").is_dir() or not (root / "sealed_labels").is_dir():
        raise RuntimeError(f"{ENV_FCWD_ROOT}={root} lacks inputs/ and sealed_labels/")
    return root


@dataclass(frozen=True)
class BoxTable:
    """Published FCWD boxes of one sequence in file (time) order."""

    sequence_id: str
    frame_index: NDArray[np.int64]
    time_us: NDArray[np.int64]
    boxes: FloatArray


def parse_boxes(values: NDArray[Any], sequence_id: str) -> BoxTable:
    """Validate a ``[N, 6]`` bbox array (frame index, seconds, x1, y1, x2, y2)."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 6 or not np.isfinite(v).all():
        raise ValueError(f"{sequence_id}: bbox table must be finite [N, 6]")
    index = v[:, 0].astype(np.int64)
    if (v[:, 0] != index).any():
        raise ValueError(f"{sequence_id}: non-integer frame index")
    times = np.rint(v[:, 1] * 1_000_000).astype(np.int64)
    if not np.all(np.diff(times) > 0) or not np.all(np.diff(index) > 0):
        raise ValueError(f"{sequence_id}: bbox rows must be strictly increasing in time and index")
    return BoxTable(sequence_id, index, times, v[:, 2:6].copy())


def load_boxes(path: Path, sequence_id: str) -> BoxTable:
    """Read one ``bounding_box.csv`` (no header)."""
    return parse_boxes(np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2), sequence_id)


def event_span_us(path: Path) -> tuple[int, int]:
    """First event time and last event time + 1 (half-open), reading two scalars only."""
    import h5py

    with h5py.File(path, "r") as handle:
        dataset = handle[EVENT_T_PATH]
        assert isinstance(dataset, h5py.Dataset)
        return int(dataset[0]), int(dataset[-1]) + 1


def cohort_anchors(table: BoxTable, stream_start: int, stream_end: int) -> NDArray[np.int64]:
    """Anchors of the 2026-10-08 label-free cohort (event history and causal-box filter)."""
    available = table.time_us + BBOX_AVAILABILITY_MARGIN_US
    kept: list[int] = []
    for anchor in table.time_us:
        a = int(anchor)
        if a - HISTORY_US < stream_start or a > stream_end:
            continue
        ok = True
        for offset in COHORT_ENDPOINT_OFFSETS_US:
            endpoint = a + offset
            eligible = np.flatnonzero(available <= endpoint)
            if len(eligible) == 0 or endpoint - int(available[eligible[-1]]) > (
                BBOX_MAX_POST_AVAILABILITY_AGE_US
            ):
                ok = False
                break
        if ok:
            kept.append(a)
    return np.asarray(kept, dtype=np.int64)


def select_frames(
    table: BoxTable, anchor_us: int, latency_us: int, rate_hz: int, window_us: int = WINDOW_US
) -> NDArray[np.int64]:
    """Row indices usable at ``anchor_us``: available (time <= anchor - latency), inside the
    window ending at the anchor, and at 10 Hz only frames whose index has the parity of the
    latest available frame.  Rows later than the cutoff can never be selected."""
    if rate_hz not in RATES:
        raise ValueError(f"rate must be one of {RATES}")
    cutoff = anchor_us - latency_us
    rows = np.flatnonzero((table.time_us <= cutoff) & (table.time_us >= anchor_us - window_us))
    if rate_hz == 10 and len(rows):
        last = table.frame_index[rows[-1]]
        rows = rows[(last - table.frame_index[rows]) % 2 == 0]
    return rows


def truncation_fcwd(boxes: NDArray[Any]) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """FCWD border contact (published minimum coordinate is 1, maximum 1281/721)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    trunc_x = (b[:, 0] <= 1) | (b[:, 2] >= SENSOR_WIDTH - 1)
    trunc_y = (b[:, 1] <= 1) | (b[:, 3] >= SENSOR_HEIGHT - 1)
    return trunc_x, trunc_y


def raw_features(table: BoxTable, rows: NDArray[np.int64], anchor_us: int) -> dict[str, float]:
    """``geometry.window_features`` on the selected rows with FCWD truncation semantics."""
    out = geometry.window_features(
        table.time_us[rows], table.boxes[rows], anchor_us, window_us=WINDOW_US
    )
    if len(rows):
        tx, ty = truncation_fcwd(table.boxes[rows])
        out["trunc_x_current"] = float(tx[-1])
        out["trunc_y_current"] = float(ty[-1])
        out["trunc_x_any"] = float(tx.any())
        out["trunc_y_any"] = float(ty.any())
        if out["trunc_x_any"] and not out["trunc_y_any"]:
            out["phi_trunc_aware"] = out["phi_height"]
        elif out["trunc_y_any"] and not out["trunc_x_any"]:
            out["phi_trunc_aware"] = out["phi_width"]
        else:
            out["phi_trunc_aware"] = out["phi_area"]
    return out


ZERO_PHASE_TOLERANCE = 1e-12


def phase_ttc(phase: NDArray[Any] | float) -> FloatArray:
    """Released ``phase_to_ttc`` (|T| <= 60 s); zero expansion gives +60 s.

    Identical boxes give an OLS slope of ±1e-17 from rounding; such phases are set to exactly
    zero so the capped TTC does not take a random sign (addendum A2).
    """
    from operational.phase_geometry.metrics import phase_to_ttc

    phi = np.asarray(phase, dtype=np.float64)
    return phase_to_ttc(np.where(np.abs(phi) < ZERO_PHASE_TOLERANCE, 0.0, phi))


def size_ratio_last2(table: BoxTable, rows: NDArray[np.int64]) -> float:
    """Classical size-change TTC from the last two usable frames, capped to |T| <= 60 s."""
    if len(rows) < 2:
        return TTC_CAP_S
    b = table.boxes[rows[-2:]]
    w = b[:, 2] - b[:, 0] + 1.0
    h = b[:, 3] - b[:, 1] + 1.0
    s_prev, s_now = float(np.sqrt(w[0] * h[0])), float(np.sqrt(w[1] * h[1]))
    dt = float(table.time_us[rows[-1]] - table.time_us[rows[-2]]) * 1e-6
    growth = s_now - s_prev
    if growth == 0.0 or dt <= 0.0:
        return TTC_CAP_S
    ttc = dt * s_prev / growth
    return float(np.sign(ttc) * min(abs(ttc), TTC_CAP_S))


@dataclass(frozen=True)
class QueryGeometry:
    """Raw geometry ``[N, len(RAW_FIELDS)]`` and the last-two size-ratio TTC for one setting."""

    raw: FloatArray
    size_ratio_ttc: FloatArray
    n_frames: NDArray[np.int64]
    both_truncated_current: NDArray[np.bool_]


def query_geometry(
    table: BoxTable, anchors: NDArray[np.int64], latency_us: int, rate_hz: int
) -> QueryGeometry:
    """Geometry of every anchor for one (latency, rate) setting."""
    raw = np.zeros((len(anchors), len(geometry.RAW_FIELDS)), dtype=np.float64)
    ratio = np.zeros(len(anchors), dtype=np.float64)
    n = np.zeros(len(anchors), dtype=np.int64)
    both = np.zeros(len(anchors), dtype=bool)
    for i, anchor in enumerate(anchors):
        rows = select_frames(table, int(anchor), latency_us, rate_hz)
        values = raw_features(table, rows, int(anchor))
        raw[i] = [values[name] for name in geometry.RAW_FIELDS]
        ratio[i] = size_ratio_last2(table, rows)
        n[i] = len(rows)
        both[i] = bool(values["trunc_x_current"] and values["trunc_y_current"])
    return QueryGeometry(raw, ratio, n, both)


def fixed_arms(q: QueryGeometry) -> dict[str, FloatArray]:
    """Non-learned geometric TTC estimates (seconds, |T| <= 60)."""
    return {
        "SIZE_RATIO_LAST2": q.size_ratio_ttc,
        "GEO_AREA": phase_ttc(geometry.raw_column(q.raw, "phi_area")),
        "GEO_TRUNC_AWARE": phase_ttc(geometry.raw_column(q.raw, "phi_trunc_aware")),
    }


def learned_design(raw: NDArray[Any]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """GEO_LEARNED inputs G and anchor (clipped phi_area), identical to ``variants.design``."""
    x = geometry.learned_inputs(raw)
    phi = geometry.raw_column(raw, "phi_area")
    anchor = np.clip(phi, -geometry.LEARNED_PHASE_CLIP, geometry.LEARNED_PHASE_CLIP)
    return x, anchor.astype(np.float32)


__all__ = [
    "ENV_FCWD_ROOT",
    "HDF5_NAMES",
    "LATENCIES_US",
    "RATES",
    "SEQUENCES",
    "TTC_CAP_S",
    "WINDOW_US",
    "BoxTable",
    "QueryGeometry",
    "cohort_anchors",
    "event_span_us",
    "fcwd_root",
    "fixed_arms",
    "learned_design",
    "load_boxes",
    "parse_boxes",
    "phase_ttc",
    "query_geometry",
    "raw_features",
    "select_frames",
    "size_ratio_last2",
    "truncation_fcwd",
]
