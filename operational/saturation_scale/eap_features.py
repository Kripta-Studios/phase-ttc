"""Content-scale fields for every eAP TRAIN40 row and its virtual-zoom copy (streaming, resumable).

``python -m operational.saturation_scale.eap_features run --out <dir> --workers 8``
``python -m operational.saturation_scale.eap_features merge --out <dir>``

Per sequence (one chunk file each, skipped when present) the rows are visited in frame-time
order; each event window ``[end - TAU, end)`` is read once from ``events.h5`` and kept as an
integral image in a small LRU.  Only public TRAIN inputs are read: boxes, frame times, the
per-frame event windows (for the frame-to-event clock) and the event stream.  No target is read.
The window of the current frame ends at that frame's event time (the end of its official event
window), so every window is causal.  The augmented copy uses the same images restricted to the
box inside the row's pre-registered virtual crop of ``geometry_robust``.

Environment: ``PHASE_TTC_GARLTTC_DATASET`` (train parquet) and ``PHASE_TTC_EAP_RAW_ROOT`` (eAP
root holding ``data/train/<sequence>/events.h5``).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from . import content_scale as cs

if TYPE_CHECKING:
    import pandas as pd

GEOMETRY_WORK = Path("artifacts/sota_20261118/geometry/work")
ROBUST_WORK = Path("artifacts/sota_20261118/geometry_robust/work")
ENV_EAP_RAW_ROOT = "PHASE_TTC_EAP_RAW_ROOT"
FRAME_STEP_US = 100_000
CACHE_WINDOWS = 16
COLUMNS = [
    "sample_token",
    "sequence_id",
    "track_id",
    "timestamp_us",
    "frame_timestamps_us",
    "boxes_xyxy",
    "event_windows_us",
    "events_path",
]


def env_dir(variable: str, purpose: str) -> Path:
    """Directory named by an environment variable; actionable error if unset or missing."""
    value = os.environ.get(variable, "").strip()
    if not value or not Path(value).is_dir():
        raise RuntimeError(f"Set {variable} to {purpose} (currently {value!r}).")
    return Path(value)


def _npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class EventWindowReader(Protocol):
    """Half-open event window reader (eAP or FCWD)."""

    def read_window(self, start_us: int, end_us: int) -> dict[str, NDArray[Any]]:
        """Events in ``[start_us, end_us)`` with keys ``x``, ``y``, ``t``, ``p``."""
        ...


class WindowCache:
    """LRU of integral images keyed by the causal window end (ms)."""

    def __init__(
        self, reader: EventWindowReader, stream_start_us: int, size: int = CACHE_WINDOWS
    ) -> None:
        self.reader = reader
        self.stream_start_us = int(stream_start_us)
        self.size = size
        self.items: OrderedDict[int, NDArray[np.int64] | None] = OrderedDict()
        self.reads = 0
        self.read_seconds = 0.0

    def get(self, end_us: int) -> NDArray[np.int64] | None:
        """Integral image of ``[key*1000 - TAU, key*1000)`` (``None`` before the stream)."""
        key = cs.window_key_ms(end_us)
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        end = key * 1000
        start = end - cs.TAU_US
        value: NDArray[np.int64] | None = None
        if start >= self.stream_start_us:
            began = time.perf_counter()
            events = self.reader.read_window(start, end)
            value = cs.integral_image(events["x"], events["y"])
            self.reads += 1
            self.read_seconds += time.perf_counter() - began
        self.items[key] = value
        while len(self.items) > self.size:
            self.items.popitem(last=False)
        return value


def _region_or_none(region: cs.Region) -> NDArray[np.float64] | None:
    return None if region.empty else region.center


def sequence_rows(
    frame: pd.DataFrame, rows: NDArray[np.int64], crops: NDArray[Any]
) -> list[dict[str, Any]]:
    """Per-row query descriptions (frame-time order) for one sequence; no event payloads."""
    from operational.phase_geometry import geometry

    sub = frame.iloc[rows]
    clock: dict[int, int] = {}
    for stamps, windows in zip(sub["frame_timestamps_us"], sub["event_windows_us"], strict=True):
        for stamp, window in zip(stamps, windows, strict=True):
            end = int(np.asarray(window, dtype=np.int64)[1])
            if clock.setdefault(int(stamp), end) != end:
                raise ValueError("inconsistent frame-to-event clock")
    pools: dict[str, Any] = {}
    for track, part in sub.groupby("track_id", sort=False):
        pools[str(track)] = geometry.build_track_pool(
            part["timestamp_us"].to_numpy(np.int64),
            [np.asarray(v, dtype=np.int64) for v in part["frame_timestamps_us"]],
            [np.stack(v).astype(np.float64) for v in part["boxes_xyxy"]],
        )
    out = []
    for local, row in enumerate(rows):
        r = sub.iloc[local]
        stamp = int(r["timestamp_us"])
        pool = pools[str(r["track_id"])]
        available = pool.available_us <= stamp
        frames = {
            int(t): b for t, b in zip(pool.frame_us[available], pool.boxes[available], strict=True)
        }
        if stamp not in frames:
            raise ValueError("a row does not supply its own frame")
        current = frames[stamp]
        crop = cs.crop_region(crops[row])
        orig = cs.box_region(current)
        aug = orig.intersect(crop)
        centers_orig: dict[int, Any] = {}
        centers_aug: dict[int, Any] = {}
        for lag in cs.LAGS_US:
            box = frames.get(stamp - lag)
            if box is None:
                centers_orig[lag] = centers_aug[lag] = None
                continue
            region = cs.box_region(box)
            centers_orig[lag] = _region_or_none(region)
            centers_aug[lag] = _region_or_none(region.intersect(crop))
        out.append(
            {
                "row": int(row),
                "event_end_us": clock[stamp],
                "orig": orig,
                "aug": aug,
                "centers_orig": centers_orig,
                "centers_aug": centers_aug,
            }
        )
    out.sort(key=lambda q: (q["event_end_us"], q["row"]))
    return out


def run_sequence(arguments: tuple[str, str, str, int | None]) -> dict[str, Any]:
    """Compute and save one sequence chunk (no-op if it exists)."""
    sequence, out_dir, root_dir, limit = arguments
    import pandas as pd

    from e_jepa_ttc.data.eap import EAPEventReader
    from operational.phase_geometry import data as pg_data
    from operational.train40_system.data_audit import safe_media_path

    out = Path(out_dir) / "chunks"
    target = out / f"{sequence}.npz"
    if target.exists():
        return {"sequence": sequence, "skipped": True}
    raw_root = env_dir(ENV_EAP_RAW_ROOT, "the eAP root holding data/train/<sequence>/events.h5")
    root = Path(root_dir)
    train = _npz(root / GEOMETRY_WORK / "train_inputs.npz")
    rob = _npz(root / ROBUST_WORK / "robust_work.npz")
    tokens = train["sample_token"].astype(str)
    if not np.array_equal(tokens, rob["sample_token"].astype(str)):
        raise ValueError("robust work rows differ from geometry work rows")
    rows = np.flatnonzero(train["group_id"].astype(str) == sequence)
    frame = pd.read_parquet(
        pg_data.verified_file("train_data", root),
        columns=COLUMNS,
        filters=[("sequence_id", "==", sequence)],
    )
    frame = frame.set_index("sample_token").loc[tokens[rows]].reset_index()
    if not np.array_equal(frame["timestamp_us"].to_numpy(np.int64), train["timestamp_us"][rows]):
        raise ValueError(f"{sequence}: parquet rows differ from the geometry work rows")
    paths = set(frame["events_path"].astype(str))
    if len(paths) != 1:
        raise ValueError(f"{sequence}: expected one event stream")
    events_path = safe_media_path(paths.pop(), sequence, raw_root)
    # positions inside `frame` are 0..len(rows)-1; crops are indexed by global row
    queries = sequence_rows(frame, np.arange(len(rows)), rob["crops"][rows])
    if limit is not None:
        queries = queries[:limit]
    c_orig = np.zeros((len(queries), len(cs.RAW_C_FIELDS)))
    c_aug = np.zeros_like(c_orig)
    began = time.perf_counter()
    compute_seconds = 0.0
    with EAPEventReader(events_path) as reader:
        cache = WindowCache(reader, reader.t_start_us)
        for i, q in enumerate(queries):
            end = q["event_end_us"]
            current = cache.get(end)
            previous = {lag: cache.get(end - lag) for lag in cs.LAGS_US}
            t0 = time.perf_counter()
            for table, region, centers in (
                (c_orig, q["orig"], q["centers_orig"]),
                (c_aug, q["aug"], q["centers_aug"]),
            ):
                values = cs.query_content(current, previous, region, centers)
                table[i] = [values[name] for name in cs.RAW_C_FIELDS]
            compute_seconds += time.perf_counter() - t0
        reads, read_seconds = cache.reads, cache.read_seconds
    local = np.asarray([q["row"] for q in queries], dtype=np.int64)
    global_rows = rows[local]
    out.mkdir(parents=True, exist_ok=True)
    pending = out / f"{sequence}.pending.npz"
    np.savez_compressed(
        pending,
        rows=global_rows,
        tokens=tokens[global_rows],
        event_end_us=np.asarray([q["event_end_us"] for q in queries], dtype=np.int64),
        c_orig=c_orig,
        c_aug=c_aug,
    )
    final = target if limit is None else out / f"{sequence}.limit{limit}.npz"
    os.replace(pending, final)
    record = {
        "sequence": sequence,
        "rows": len(queries),
        "window_reads": reads,
        "read_seconds": read_seconds,
        "compute_seconds": compute_seconds,
        "seconds": time.perf_counter() - began,
        "valid_orig": int(cs.raw_c_column(c_orig, "valid").sum()),
        "valid_aug": int(cs.raw_c_column(c_aug, "valid").sum()),
    }
    (out / f"{final.stem}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def merge(out_dir: Path, root: Path) -> dict[str, Any]:
    """Assemble ``content_work.npz`` in ``train_inputs.npz`` row order and write a receipt."""
    from operational.phase_geometry import data as pg_data

    train = _npz(root / GEOMETRY_WORK / "train_inputs.npz")
    tokens = train["sample_token"].astype(str)
    n = len(tokens)
    c_orig = np.full((n, len(cs.RAW_C_FIELDS)), np.nan)
    c_aug = np.full_like(c_orig, np.nan)
    end_us = np.zeros(n, dtype=np.int64)
    records = []
    for sequence in sorted(set(train["group_id"].astype(str).tolist())):
        chunk = _npz(out_dir / "chunks" / f"{sequence}.npz")
        if not np.array_equal(tokens[chunk["rows"]], chunk["tokens"].astype(str)):
            raise ValueError(f"{sequence}: chunk tokens out of order")
        c_orig[chunk["rows"]] = chunk["c_orig"]
        c_aug[chunk["rows"]] = chunk["c_aug"]
        end_us[chunk["rows"]] = chunk["event_end_us"]
        records.append(
            json.loads((out_dir / "chunks" / f"{sequence}.json").read_text(encoding="utf-8"))
        )
    if not (np.isfinite(c_orig).all() and np.isfinite(c_aug).all()):
        raise ValueError("missing content rows")
    path = out_dir / "content_work.npz"
    np.savez_compressed(path, sample_token=tokens, c_orig=c_orig, c_aug=c_aug, event_end_us=end_us)
    receipt = {
        "schema": "saturation_scale_eap_content_v1",
        "rows": n,
        "fields": list(cs.RAW_C_FIELDS),
        "parameters": {
            "tau_us": cs.TAU_US,
            "lags_us": list(cs.LAGS_US),
            "grid": cs.GRID,
            "template": cs.TEMPLATE,
            "k_search": list(cs.K_SEARCH),
            "coarse_step": cs.COARSE_STEP,
            "fine_step": cs.FINE_STEP,
            "min_score": cs.MIN_SCORE,
            "min_template_events": cs.MIN_TEMPLATE_EVENTS,
        },
        "valid_orig": int(cs.raw_c_column(c_orig, "valid").sum()),
        "valid_aug": int(cs.raw_c_column(c_aug, "valid").sum()),
        "window_reads": int(sum(r["window_reads"] for r in records)),
        "read_seconds": float(sum(r["read_seconds"] for r in records)),
        "compute_seconds": float(sum(r["compute_seconds"] for r in records)),
        "worker_seconds": float(sum(r["seconds"] for r in records)),
        "targets_read": False,
        "output_sha256": pg_data.sha256_file(path),
        "inputs_sha256": {
            "train_inputs.npz": pg_data.sha256_file(root / GEOMETRY_WORK / "train_inputs.npz"),
            "robust_work.npz": pg_data.sha256_file(root / ROBUST_WORK / "robust_work.npz"),
        },
        "code_sha256": {
            f: pg_data.sha256_file(Path(__file__).with_name(f))
            for f in ("content_scale.py", "eap_features.py")
        },
    }
    (out_dir / "CONTENT_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "merge"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sequences", nargs="*", help="subset (pilot)")
    parser.add_argument("--limit", type=int, help="first rows per sequence (pilot)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = args.out.resolve()
    if args.stage == "merge":
        print(json.dumps(merge(out, root), indent=1))
        return 0
    train = _npz(root / GEOMETRY_WORK / "train_inputs.npz")
    sequences = args.sequences or sorted(set(train["group_id"].astype(str).tolist()))
    tasks = [(s, str(out), str(root), args.limit) for s in sequences]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.workers, maxtasksperchild=1) as pool:
        for record in pool.imap_unordered(run_sequence, tasks):
            print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
