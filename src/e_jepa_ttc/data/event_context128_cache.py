"""Bounded, atomic E128-T8 shard publication using the existing raw binding contract."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.cache_components import component_record, verify_components
from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE
from e_jepa_ttc.data.event_context128 import (
    COUNT_SHAPE,
    Window128,
    checked_count_cast,
    rasterize128,
)
from e_jepa_ttc.data.raw_event_binding import ReadOnlyTrainAccess, _reject_external_links
from e_jepa_ttc.data.stage70_clock import context_clock

EVENT_CHUNK = 262_144


def _publish_array(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class WindowReader:
    """One HDF5 handle and at most two 64-MiB physical windows, without targets."""

    def __init__(self, access: ReadOnlyTrainAccess, source_hashes: dict[str, Any]) -> None:
        self.access = access
        self.source_hashes = source_hashes
        self.handle: h5py.File | None = None
        self.sequence: str | None = None
        self.path: Path | None = None
        self.datasets: dict[str, h5py.Dataset] = {}
        self.memo: OrderedDict[tuple[int, int], list[np.ndarray]] = OrderedDict()

    def close(self) -> None:
        """Release dataset and file handles before moving to a new sequence."""
        self.memo.clear()
        self.datasets.clear()
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.sequence = None

    def _open(self, row: pd.Series) -> None:
        sequence = str(row["sequence_id"])
        if self.sequence == sequence:
            return
        self.close()
        self.path = self.access.event_path(
            sequence, str(row["events_path_relative"]), "E128 cache source open"
        )
        expected = self.source_hashes[sequence]
        info = self.path.stat()
        if (
            str(self.path) != expected["path"]
            or info.st_size != expected["bytes"]
            or info.st_mtime_ns != expected["mtime_ns"]
            or str(row["h5_file_sha256"]) != expected["sha256"]
        ):
            raise ValueError("raw source changed since its full hash verification")
        handle = h5py.File(self.path, "r")
        self.handle = handle
        for key in ("events", "ms_to_idx"):
            if isinstance(handle.get(key, getlink=True), (h5py.ExternalLink, h5py.SoftLink)):
                raise ValueError("indirect HDF5 parent is forbidden")
        _reject_external_links(handle, self.path)
        for key in "xytp":
            name = f"events/{key}"
            dataset = cast(h5py.Dataset, handle[name])
            if (
                isinstance(handle.get(name, getlink=True), h5py.SoftLink)
                or dataset.is_virtual
                or dataset.external
            ):
                raise ValueError("indirect HDF5 event storage is forbidden")
            self.datasets[key] = dataset
        self.sequence = sequence

    def count(self, row: pd.Series, window: Window128) -> tuple[np.ndarray, dict[str, int]]:
        """Read actual event slices in bounded chunks, reusing repeated object windows."""
        self._open(row)
        assert self.path is not None
        start, stop = (
            int(cast(Any, row["start_event_index"])),
            int(cast(Any, row["stop_event_index"])),
        )
        if not 0 <= start <= stop <= len(self.datasets["t"]):
            raise ValueError("bound event indices outside the physical source")
        key = start, stop
        if key in self.memo:
            self.memo.move_to_end(key)
            cached = self.memo[key]
            return rasterize128(cached[0], cached[1], cached[2], cached[3], window)
        self.access.record_h5_slice(self.path, "E128 physical window read", start, stop)
        estimated = (stop - start) * sum(
            dataset.dtype.itemsize for dataset in self.datasets.values()
        )
        if estimated <= 64 * 1024**2:
            arrays = [np.empty(stop - start, dtype=self.datasets[field].dtype) for field in "xytp"]
            for begin in range(start, stop, EVENT_CHUNK):
                end = min(stop, begin + EVENT_CHUNK)
                for field, array in zip("xytp", arrays, strict=True):
                    array[begin - start : end - start] = self.datasets[field][begin:end]
            self.memo[key] = arrays
            while len(self.memo) > 2:
                self.memo.popitem(last=False)
            return rasterize128(arrays[0], arrays[1], arrays[2], arrays[3], window)
        total = np.zeros((8, 2, 128, 128), dtype=np.uint64)
        diagnostics = dict(raw_events=0, outside_sensor=0, outside_roi=0, roi_events=0)
        previous: int | None = None
        for begin in range(start, stop, EVENT_CHUNK):
            end = min(stop, begin + EVENT_CHUNK)
            arrays = [np.asarray(self.datasets[field][begin:end]) for field in "xytp"]
            if previous is not None and int(arrays[2][0]) < previous:
                raise ValueError("timestamp rollback across physical chunks")
            previous = int(arrays[2][-1])
            counts, record = rasterize128(arrays[0], arrays[1], arrays[2], arrays[3], window)
            total += counts
            for name, value in record.items():
                diagnostics[name] += value
        return checked_count_cast(total, "uint32"), diagnostics


def build_count_cache(
    binding_csv: Path,
    source_hashes: dict[str, Any],
    access: ReadOnlyTrainAccess,
    output: Path,
    *,
    shard_rows: int = 64,
    dtype: str = "uint16",
) -> dict[str, Any]:
    """Publish complete bounded shards; resume only verified, flushed predecessors.

    Overflow stops publication with all completed shards intact. A caller must
    reserve and build a whole uint32 version; mixed storage is never certified.
    """
    if not 1 <= shard_rows <= 64 or dtype not in ("uint16", "uint32"):
        raise ValueError("invalid bounded cache execution settings")
    table = pd.read_csv(binding_csv)
    if table.duplicated(["sample_token", "window_id"]).any():
        raise ValueError("duplicate bound token/window")
    if not table.groupby("sample_token").window_id.apply(lambda ids: sorted(ids) == [0, 1]).all():
        raise ValueError("every token requires exactly the original two windows")
    token_rows = table.loc[table.window_id == 0].sort_values(
        ["sequence_id", "window_start_us", "sample_token"]
    )
    metadata = token_rows[["sample_token", "sequence_id", "track_id", "outer_fold"]].reset_index(
        drop=True
    )
    indexed = table.set_index(["sample_token", "window_id"])
    identity = {
        "producer_code": [
            binding(Path(__file__).with_name(name))
            for name in (
                "event_context128_cache.py",
                "event_context128.py",
                "stage70_clock.py",
                "raw_temporal_cache.py",
                "raw_event_binding.py",
                "eap.py",
            )
        ],
        "binding": binding(binding_csv),
        "sources": source_hashes,
        "shape": [len(metadata), *COUNT_SHAPE],
        "dtype": dtype,
        "shard_rows": shard_rows,
        "sensor_size": list(EAP_IMAGE_SIZE),
        "channel_order": "bin-major then negative,positive",
        "bin_edges": "integer_floor_8_half_open",
        "normalization": "not fitted in count cache",
        "targets_present": False,
        "sensor_support": "ROI pixel-center coverage of original sensor before x+5 calibration",
    }
    identity_hash = object_digest(identity)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "CACHE_MANIFEST.json"
    progress_path = output / "CACHE_PROGRESS.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["identity_sha256"] != identity_hash:
            raise ValueError("completed cache identity mismatch")
        unsigned = {key: value for key, value in existing.items() if key != "artifact_sha256"}
        if object_digest(unsigned) != existing.get("artifact_sha256"):
            raise ValueError("completed cache manifest signature mismatch")
        for shard in existing["shards"]:
            verify_components(output, shard["components"], set(shard["components"]))
        verify_components(output, {"metadata.csv": existing["metadata"]}, {"metadata.csv"})
        pd.testing.assert_frame_equal(
            pd.read_csv(output / "metadata.csv"), metadata, check_dtype=False, check_exact=True
        )
        return existing
    progress: dict[str, Any] = {"identity_sha256": identity_hash, "completed_rows": 0, "shards": []}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["identity_sha256"] != identity_hash:
            raise ValueError("cache resume input identity mismatch")
    for shard in progress["shards"]:
        verify_components(output, shard["components"], set(shard["components"]))
    position = 0
    for number, shard in enumerate(progress["shards"]):
        expected_rows = min(shard_rows, len(metadata) - position)
        if shard["first_row"] != position or shard["rows"] != expected_rows or expected_rows <= 0:
            raise ValueError("cache shard order or row count mismatch")
        count_record = shard["components"][f"counts_{number:05d}.npy"]
        if count_record["shape"] != [expected_rows, *COUNT_SHAPE] or count_record["dtype"] != dtype:
            raise ValueError("cache count tensor schema mismatch")
        position += expected_rows
    completed = int(progress["completed_rows"])
    if completed != sum(shard["rows"] for shard in progress["shards"]):
        raise ValueError("cache progress row accounting mismatch")
    row_bytes = int(np.prod(COUNT_SHAPE)) * np.dtype(dtype).itemsize + 4096
    reader = WindowReader(access, source_hashes)
    with ResourceOwner(
        output,
        remaining_bytes=(len(metadata) - completed) * row_bytes + 4 * 1024**2,
        planned_ram_bytes=2 * 1024**3,
    ) as owner:
        try:
            for first in range(completed, len(metadata), shard_rows):
                owner.check()
                last = min(len(metadata), first + shard_rows)
                count_rows = np.empty((last - first, *COUNT_SHAPE), dtype=dtype)
                durations = np.empty((last - first, 2, 8), dtype=np.float64)
                times = np.empty((last - first, 4), dtype=np.int64)
                support_bits = np.empty((last - first, 2048), dtype=np.uint8)
                diagnostics = []
                for local, index in enumerate(range(first, last)):
                    token = str(metadata.iloc[index].sample_token)
                    pair = [cast(pd.Series, indexed.loc[(token, window)]) for window in (0, 1)]
                    if any(
                        str(row.h5_file_sha256) != source_hashes[str(row.sequence_id)]["sha256"]
                        for row in pair
                    ):
                        raise ValueError("bound raw hash differs from verified source")
                    bounds = [(int(row.window_start_us), int(row.window_end_us)) for row in pair]
                    frames = [
                        int(row.window_end_us) + int(row.frame_to_event_clock_offset_us)
                        for row in pair
                    ]
                    clock = context_clock(bounds, frames, int(pair[1].target_anchor_us))
                    rois = [
                        tuple(
                            float(cast(Any, row[f"roi_{axis}"]))
                            for axis in ("x0", "y0", "x1", "y1")
                        )
                        for row in pair
                    ]
                    if rois[0] != rois[1] or any(
                        float(row.event_x_offset_px) != 5.0 for row in pair
                    ):
                        raise ValueError("common ROI/calibration changed across time")
                    token_diagnostics = []
                    for window_id, row in enumerate(pair):
                        window = Window128(
                            *bounds[window_id],
                            cast(tuple[float, float, float, float], rois[window_id]),
                            EAP_IMAGE_SIZE,
                            polarity_encoding=cast(Any, str(row.polarity_encoding)),
                        )
                        counts, record = reader.count(row, window)
                        count_rows[local, window_id] = checked_count_cast(counts, dtype)
                        durations[local, window_id] = np.diff(window.edges()) * 1e-6
                        token_diagnostics.append(record)
                    times[local] = clock.midpoint_twice_relative_us
                    support_bits[local] = np.packbits(
                        window.support().reshape(-1), bitorder="little"
                    )
                    diagnostics.append({"sample_token": token, "windows": token_diagnostics})
                shard_number = len(progress["shards"])
                count_name, auxiliary_name = (
                    f"counts_{shard_number:05d}.npy",
                    f"observations_{shard_number:05d}.npz",
                )
                _publish_array(output / count_name, count_rows)
                temporary = output / (auxiliary_name + ".tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        durations_s=durations,
                        midpoint_twice_relative_us=times,
                        support_bits=support_bits,
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, output / auxiliary_name)
                diagnostic_name = f"diagnostics_{shard_number:05d}.json"
                atomic_json(output / diagnostic_name, diagnostics)
                progress["shards"].append(
                    {
                        "first_row": first,
                        "rows": last - first,
                        "max_count": int(count_rows.max(initial=0)),
                        "components": {
                            name: component_record(output / name)
                            for name in (count_name, auxiliary_name, diagnostic_name)
                        },
                    }
                )
                progress["completed_rows"] = last
                atomic_json(progress_path, progress)
                owner.estimated_remaining_bytes = (len(metadata) - last) * row_bytes + 4 * 1024**2
                print(
                    f"E128 cache {last}/{len(metadata)} rows; shard {shard_number} published",
                    flush=True,
                )
        finally:
            reader.close()
        for sequence in set(metadata.sequence_id.astype(str)):
            source = access.event_path(sequence, f"{sequence}/events.h5", "cache final source stat")
            expected = source_hashes[sequence]
            info = source.stat()
            if info.st_size != expected["bytes"] or info.st_mtime_ns != expected["mtime_ns"]:
                raise ValueError("raw source changed during cache construction")
        metadata.to_csv(output / "metadata.csv", index=False, lineterminator="\n")
        result = {
            "status": "COMPLETE_COUNTS_ONLY",
            "identity_sha256": identity_hash,
            "identity": identity,
            "shards": progress["shards"],
            "metadata": component_record(output / "metadata.csv"),
            "normalization_fitted": False,
            "optimizer_updates": 0,
        }
        result["artifact_sha256"] = object_digest(result)
        atomic_json(manifest_path, result)
    return result
