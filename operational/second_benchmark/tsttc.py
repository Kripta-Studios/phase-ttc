"""TSTTC official metrics, a safe annotation reader and the geometry arms (ready, not executed).

The TSTTC data could not be obtained without an account (see the campaign PROTOCOL.md), so
this module is exercised only by unit tests on synthetic data.  Definitions mirror
``tusen-ai/TSTTC@b59e7a1`` (``data/dataset_api.py::evalRes`` and ``core/evaluator.py``):

* ``alpha(T) = 1 / (1/(T*fps) + 1 + 1e-6)`` with ``fps = 10`` (scale ratio over 0.1 s),
* ``MiD = 1e4 * |log alpha(T_gt) - log alpha(clip(T_hat, -20, 20))|``,
* ``RTE = 100 * |clip(T_hat) - T_gt| / |T_gt|``,
* ranges on ``T_gt``: negative [-20, 0], crucial [0, 3], small [3, 6], large [6, 20]
  (``evalRes`` uses inclusive bounds; ``TTCEvaluator.cal_error`` uses strict ones).

Annotation ``.pkl`` files are pickles of objects of an unpublished class; they are read with a
restricted unpickler that maps every non-builtin class to :class:`AnnotationObject` and refuses
any callable, so no code from the archive can run.
"""

from __future__ import annotations

import io
import json
import math
import os
import pickle
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from operational.phase_geometry import geometry

ENV_BENCH2_ROOT = "PHASE_TTC_BENCH2_ROOT"
FPS = 10
PRED_CLIP_S = 20.0
SEQUENCE_LENGTH = 6
FRAME_STEP_NS = 100_000_000
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 576
RANGES = {
    "negative_ttc(-20~0)": (-20.0, 0.0),
    "crucial_ttc(0~3)": (0.0, 3.0),
    "small_ttc(3~6)": (3.0, 6.0),
    "large_ttc(6~20)": (6.0, 20.0),
}

FloatArray = NDArray[np.float64]


def bench2_root(environ: Mapping[str, str] | None = None) -> Path:
    """TSTTC root from ``PHASE_TTC_BENCH2_ROOT`` (expects ``tsttc/{train,val}``)."""
    env = os.environ if environ is None else environ
    value = env.get(ENV_BENCH2_ROOT)
    if not value:
        raise RuntimeError(
            f"{ENV_BENCH2_ROOT} is not set: point it at the download root that contains "
            "tsttc/val/<bag>.pkl (TSTTC must first be downloaded by the owner from Baidu Netdisk)"
        )
    return Path(value)


def ttc_to_scale_ratio(ttc: NDArray[Any] | float, fps: float = FPS) -> FloatArray:
    """Official ``ttc_to_scale_ratio`` (vectorized, same 1e-6 term)."""
    t = np.asarray(ttc, dtype=np.float64)
    return 1.0 / ((1.0 / (t * fps) + 1.0) + 1e-6)


def scale_ratio_to_ttc(ratio: NDArray[Any] | float, fps: float = FPS) -> FloatArray:
    """Official ``scale_ratio_to_ttc`` of ``dataset_api.py`` (1e-6 term)."""
    r = np.asarray(ratio, dtype=np.float64)
    return 1.0 / ((fps * (1.0 / r - 1.0)) + 1e-6)


def row_metrics(truth: NDArray[Any], prediction: NDArray[Any]) -> tuple[FloatArray, FloatArray]:
    """Per-object (MiD, RTE %) exactly as ``evalRes``; zero predictions are rejected."""
    gt = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if (pred == 0).any():
        raise ValueError("ttc prediction cannot be zero")
    clipped = np.clip(pred, -PRED_CLIP_S, PRED_CLIP_S)
    rte = np.abs((clipped - gt) / gt) * 100.0
    mid = np.abs(np.log(ttc_to_scale_ratio(gt)) - np.log(ttc_to_scale_ratio(clipped))) * 1e4
    return mid, rte


def summarize(truth: NDArray[Any], prediction: NDArray[Any]) -> dict[str, float]:
    """Average MiD/RTE overall and per official range (inclusive ``evalRes`` bounds)."""
    gt = np.asarray(truth, dtype=np.float64)
    mid, rte = row_metrics(gt, prediction)
    out = {"MiD": float(mid.mean()), "RTE": float(rte.mean()), "n": float(len(gt))}
    for name, (lo, hi) in RANGES.items():
        mask = (gt >= lo) & (gt <= hi)
        out[f"MiD {name}"] = float(mid[mask].mean()) if mask.any() else math.nan
        out[f"RTE {name}"] = float(rte[mask].mean()) if mask.any() else math.nan
        out[f"n {name}"] = float(mask.sum())
    return out


class AnnotationObject:
    """Attribute holder standing in for TSTTC's unpublished annotation class."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __setstate__(self, state: object) -> None:
        if isinstance(state, tuple) and len(state) == 2:
            state = {**(state[0] or {}), **(state[1] or {})}
        if not isinstance(state, dict):
            raise pickle.UnpicklingError("unsupported annotation object state")
        self.__dict__.update(state)


def _reconstructor(*args: object) -> AnnotationObject:
    """Stand-in for ``copyreg._reconstructor`` (protocol 0/1 object pickles)."""
    del args
    return AnnotationObject()


_SAFE_BUILTINS: dict[tuple[str, str], Any] = {
    ("builtins", "dict"): dict,
    ("builtins", "list"): list,
    ("builtins", "tuple"): tuple,
    ("builtins", "set"): set,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "object"): object,
    ("copyreg", "_reconstructor"): _reconstructor,
}
_SAFE_NUMPY = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
}
_REFUSED_MODULES = {"builtins", "os", "posix", "nt", "subprocess", "sys", "importlib", "io"}


class SafeAnnotationUnpickler(pickle.Unpickler):
    """Unpickler that never imports archive code: unknown classes become AnnotationObject.

    Only plain containers, numpy array reconstruction and attribute holders are produced;
    any builtin/os/sys/subprocess global (the usual code-execution vectors) is refused.
    """

    def find_class(self, module: str, name: str) -> object:  # noqa: D102
        if (module, name) in _SAFE_BUILTINS:
            return _SAFE_BUILTINS[(module, name)]
        if (module, name) in _SAFE_NUMPY:
            return super().find_class(module, name)
        if module.split(".")[0] in _REFUSED_MODULES | {"numpy", "copyreg", "codecs", "shutil"}:
            raise pickle.UnpicklingError(f"refusing global {module}.{name}")
        return AnnotationObject


def load_annotations(data: bytes) -> list[dict[Any, list[AnnotationObject]]]:
    """Parse one ``<bag>.pkl`` payload (list of per-camera ``{timestamp: [objects]}``)."""
    value = SafeAnnotationUnpickler(io.BytesIO(data)).load()
    if not isinstance(value, list):
        raise ValueError("TSTTC annotation file must be a list of per-camera dicts")
    return value


@dataclass(frozen=True)
class ObjectSequence:
    """Six consecutive 10 Hz observations of one object; the last is the target frame."""

    bag_stamp: str
    cam_id: int
    object_id: int
    ts_ns: NDArray[np.int64]
    boxes_px: FloatArray
    ttc_gt: float | None


@dataclass(frozen=True)
class TargetFreeObjectSequence:
    """Prediction input built without reading or retaining ``ttc_imu``."""

    bag_stamp: str
    cam_id: int
    object_id: int
    ts_ns: NDArray[np.int64]
    boxes_px: FloatArray


def build_prediction_sequences(
    cameras: Iterable[Mapping[Any, Sequence[Any]]], sequence_len: int = SEQUENCE_LENGTH
) -> list[TargetFreeObjectSequence]:
    """Build prediction inputs while never accessing ``ttc_imu`` or another target field."""
    out: list[TargetFreeObjectSequence] = []
    for camera in cameras:
        pending: dict[int, list[Any]] = {}
        for _, objects in sorted(camera.items(), key=lambda item: item[0]):
            for obj in objects:
                run = pending.get(obj.id)
                if run is not None and obj.ts - run[-1].ts == FRAME_STEP_NS:
                    run.append(obj)
                else:
                    pending[obj.id] = [obj]
            done = [key for key, run in pending.items() if len(run) == sequence_len]
            for key in done:
                run = pending.pop(key)
                boxes = np.asarray([o.box2d for o in run], dtype=np.float64)
                boxes = boxes * np.asarray([IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT])
                last = run[-1]
                out.append(
                    TargetFreeObjectSequence(
                        bag_stamp=os.path.basename(str(last.bag_stamp)),
                        cam_id=int(last.cam_id),
                        object_id=int(last.id),
                        ts_ns=np.asarray([o.ts for o in run], dtype=np.int64),
                        boxes_px=boxes,
                    )
                )
    return out


def build_sequences(
    cameras: Iterable[Mapping[Any, Sequence[Any]]], sequence_len: int = SEQUENCE_LENGTH
) -> list[ObjectSequence]:
    """Mirror ``TSTTC.createIndex``: consecutive (``ts`` step 1e8) runs of ``sequence_len``."""
    # Compatibility reader for metric tests.  Operational prediction must call
    # build_prediction_sequences, whose return type has no target attribute.
    camera_list = list(cameras)
    target_free = build_prediction_sequences(camera_list, sequence_len)
    labels: dict[tuple[str, int, int, int], float] = {}
    for camera in camera_list:
        for _, objects in camera.items():
            for obj in objects:
                if hasattr(obj, "ttc_imu"):
                    labels[
                        (
                            os.path.basename(str(obj.bag_stamp)),
                            int(obj.cam_id),
                            int(obj.id),
                            int(obj.ts),
                        )
                    ] = float(obj.ttc_imu)
    return [
        ObjectSequence(
            bag_stamp=seq.bag_stamp,
            cam_id=seq.cam_id,
            object_id=seq.object_id,
            ts_ns=seq.ts_ns,
            boxes_px=seq.boxes_px,
            ttc_gt=labels.get((seq.bag_stamp, seq.cam_id, seq.object_id, int(seq.ts_ns[-1]))),
        )
        for seq in target_free
    ]


def detection_baseline(seq: ObjectSequence | TargetFreeObjectSequence) -> float:
    """Paper's "Detection" baseline: first/last box area ratio, ``T = gap*a/(1-a)``, a = s0/s1."""
    b = seq.boxes_px[[0, -1]]
    s = np.sqrt((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
    gap = float(seq.ts_ns[-1] - seq.ts_ns[0]) * 1e-9
    return float(scale_ratio_to_ttc(s[0] / s[1], fps=1.0 / gap))


def geometry_raw(seq: ObjectSequence | TargetFreeObjectSequence) -> dict[str, float]:
    """``geometry.window_features`` over the six frames (500 ms span), query at the last one."""
    t_us = (seq.ts_ns - seq.ts_ns[-1]) // 1000
    return geometry.window_features(t_us, seq.boxes_px, 0, window_us=geometry.WINDOW_US)


def prediction_bytes(sequences: Sequence[TargetFreeObjectSequence]) -> bytes:
    """Canonical, target-free bytes for the official two-box Detection baseline."""
    rows = [
        {
            "bag_stamp": seq.bag_stamp,
            "cam_id": seq.cam_id,
            "object_id": seq.object_id,
            "target_ts_ns": int(seq.ts_ns[-1]),
            "ttc_prediction": detection_baseline(seq),
        }
        for seq in sequences
    ]
    return json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


__all__ = [
    "ENV_BENCH2_ROOT",
    "RANGES",
    "AnnotationObject",
    "ObjectSequence",
    "TargetFreeObjectSequence",
    "SafeAnnotationUnpickler",
    "bench2_root",
    "build_sequences",
    "build_prediction_sequences",
    "detection_baseline",
    "geometry_raw",
    "load_annotations",
    "prediction_bytes",
    "row_metrics",
    "scale_ratio_to_ttc",
    "summarize",
    "ttc_to_scale_ratio",
]
