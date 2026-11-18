"""Robust-geometry work inputs for eAP TRAIN40 (originals and virtual-zoom copies) and Test12.

``python -m operational.geometry_robust.data --geometry-work <phase_geometry work> --work <dir>``

Boxes are read from the pinned public sources of :mod:`operational.phase_geometry.data` (train
parquet; Test12 ``test_inputs.parquet`` in memory from its archive).  Private Test12 labels are
never opened.  The plain geometry recomputed here must equal ``geometry_raw`` of the existing work
files exactly, otherwise the run stops.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from operational.phase_geometry import data as pg_data
from operational.phase_geometry import geometry
from operational.phase_geometry.metrics import json_safe

from . import robust

MEMORY_SHORT_US = geometry.WINDOW_US
MEMORY_LONG_US = 2_000_000
MEMORIES = (MEMORY_SHORT_US, MEMORY_LONG_US)


def _npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _columns(frame: pd.DataFrame) -> tuple[list[str], list[str], NDArray[np.int64], list, list]:
    return (
        frame["sequence_id"].astype(str).tolist(),
        frame["track_id"].astype(str).tolist(),
        frame["timestamp_us"].to_numpy(np.int64),
        [np.asarray(v, dtype=np.int64) for v in frame["frame_timestamps_us"]],
        [np.stack(v).astype(np.float64) for v in frame["boxes_xyxy"]],
    )


def _chunk(arguments: tuple[Any, ...]) -> tuple[NDArray[np.int64], Any, Any]:
    rows, seq, track, stamps, frames, boxes, crops, memories = arguments
    g, r = robust.compute_tables(
        seq, track, stamps, frames, boxes, memories_us=memories, crops=crops
    )
    return rows, g, r


def tables(
    frame: pd.DataFrame,
    memories: tuple[int, ...],
    crops: NDArray[Any] | None,
    workers: int,
) -> tuple[NDArray[np.float64], dict[int, NDArray[np.float64]]]:
    """Plain and robust tables for every row of ``frame`` (split by sequence across workers)."""
    seq, track, stamps, frames, boxes = _columns(frame)
    names = np.asarray(seq)
    unique = sorted(set(seq))
    parts = [unique[i::workers] for i in range(workers)]
    tasks = []
    for part in parts:
        rows = np.flatnonzero(np.isin(names, part))
        tasks.append(
            (
                rows,
                [seq[i] for i in rows],
                [track[i] for i in rows],
                stamps[rows],
                [frames[i] for i in rows],
                [boxes[i] for i in rows],
                None if crops is None else np.asarray(crops)[rows],
                memories,
            )
        )
    g_all = np.zeros((len(frame), len(geometry.RAW_FIELDS)))
    r_all = {m: np.zeros((len(frame), len(robust.RAW_R_FIELDS))) for m in memories}
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers) as pool:
        for rows, g, r in pool.imap_unordered(_chunk, tasks):
            g_all[rows] = g
            for m in memories:
                r_all[m][rows] = r[m]
    return g_all, r_all


def main() -> None:
    """Write ``robust_work.npz`` and ``ROBUST_DATA_RECEIPT.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--geometry-work", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    train = _npz(args.geometry_work / "train_inputs.npz")
    test = _npz(args.geometry_work / "test_inputs.npz")
    tokens = train["sample_token"].astype(str)
    frame = pd.read_parquet(pg_data.verified_file("train_data", root), columns=pg_data.BOX_COLUMNS)
    frame = frame.set_index("sample_token").loc[tokens].reset_index()
    if not np.array_equal(frame["timestamp_us"].to_numpy(np.int64), train["timestamp_us"]):
        raise ValueError("train parquet rows differ from the geometry work rows")
    g, r = tables(frame, MEMORIES, None, args.workers)
    if not np.array_equal(g, train["geometry_raw"]):
        raise ValueError("recomputed plain geometry differs from train_inputs.npz geometry_raw")
    current = np.stack([np.stack(v)[-1] for v in frame["boxes_xyxy"]]).astype(np.float64)
    last_frame = np.asarray([int(v[-1]) for v in frame["frame_timestamps_us"]], dtype=np.int64)
    if not np.array_equal(last_frame, train["timestamp_us"]):
        raise ValueError("the last supplied frame of a train row is not its own frame")
    crops = robust.draw_crops(current)
    g_aug, r_aug = tables(frame, MEMORIES, crops, args.workers)
    test_frame = pd.read_parquet(io.BytesIO(pg_data.read_member("test_inputs")))
    if pg_data.FORBIDDEN_TEST_COLUMNS.intersection(test_frame.columns):
        raise ValueError("Public test inputs unexpectedly contain a label column")
    official = test_frame["sample_token"].to_numpy(dtype=str)
    if not np.array_equal(official, test["official_token"].astype(str)):
        raise ValueError("Test12 official token order differs from test_inputs.npz")
    g_test, r_test = tables(test_frame, (MEMORY_SHORT_US,), None, args.workers)
    if not np.array_equal(g_test, test["official_geometry_raw"]):
        raise ValueError("recomputed Test12 geometry differs from test_inputs.npz")
    position = {t: i for i, t in enumerate(official)}
    canonical = np.asarray([position[t] for t in test["sample_token"].astype(str)])
    if not np.array_equal(g_test[canonical], test["geometry_raw"]):
        raise ValueError("canonical Test12 geometry differs from test_inputs.npz")
    arrays = {
        "sample_token": tokens,
        "r_short": r[MEMORY_SHORT_US],
        "r_long": r[MEMORY_LONG_US],
        "crops": crops,
        "aug_geometry_raw": g_aug,
        "aug_geometry_g": geometry.learned_inputs(g_aug),
        "aug_r_short": r_aug[MEMORY_SHORT_US],
        "aug_r_long": r_aug[MEMORY_LONG_US],
        "test_official_r_short": r_test[MEMORY_SHORT_US],
        "test_r_short": r_test[MEMORY_SHORT_US][canonical],
    }
    np.savez_compressed(work / "robust_work.npz", **arrays)
    sat = robust.raw_r_column(r[MEMORY_SHORT_US], "sat") > 0
    sat_aug = robust.raw_r_column(r_aug[MEMORY_SHORT_US], "sat") > 0
    receipt = {
        "schema": "geometry_robust_data_receipt_v1",
        "geometry_work_inputs_sha256": {
            n: pg_data.sha256_file(args.geometry_work / n)
            for n in ("train_inputs.npz", "test_inputs.npz")
        },
        "train_data": pg_data.resolved_files()["train_data"],
        "test_inputs_member": pg_data.resolved_sources()["test_inputs"],
        "plain_geometry_equal_train": True,
        "plain_geometry_equal_test_official_and_canonical": True,
        "memories_us": list(MEMORIES),
        "augmentation": {
            "seed": robust.AUG_SEED,
            "max_zoom": robust.AUG_MAX_ZOOM,
            "center_jitter": robust.AUG_CENTER_JITTER,
        },
        "rows": len(tokens),
        "sat_rows_real": int(sat.sum()),
        "sat_rows_augmented": int(sat_aug.sum()),
        "test_official_rows": len(official),
        "test_physical_rows": len(canonical),
        "private_labels_read": False,
        "output_sha256": pg_data.sha256_file(work / "robust_work.npz"),
        "code_sha256": {
            f: pg_data.sha256_file(Path(__file__).with_name(f)) for f in ("data.py", "robust.py")
        },
        "seconds": time.time() - started,
    }
    (work / "ROBUST_DATA_RECEIPT.json").write_text(
        json.dumps(json_safe(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(receipt), indent=1)[:2500])


if __name__ == "__main__":
    main()
