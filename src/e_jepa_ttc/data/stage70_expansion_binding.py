"""Bounded role-safe expansion bindings using the proven event-clock search helpers."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from e_jepa_ttc.data.raw_event_binding import (
    AccessRecord,
    RawWindowBinding,
    ReadOnlyTrainAccess,
    _as_boxes,
    _polarity_encoding,
    _reject_external_links,
    _stable_file_hashes,
    exact_window_bounds,
)
from e_jepa_ttc.data.stage70_clock import context_clock
from e_jepa_ttc.data.stage70_roles import resolve_train_event


class ExpansionTrainAccess(ReadOnlyTrainAccess):
    """Extend the source guard via an explicit role allowlist; retain its durable read journal."""

    def __init__(
        self,
        raw_train_root: Path,
        metadata_path: Path,
        *,
        allowed_sequences: frozenset[str],
        journal_path: Path,
        resource_check: Callable[[], None],
    ) -> None:
        if not allowed_sequences:
            raise ValueError("explicit nonempty expansion sequence allowlist required")
        self.allowed_sequences = allowed_sequences
        super().__init__(
            raw_train_root, metadata_path, journal_path=journal_path, resource_check=resource_check
        )

    def _record(
        self, path: Path, purpose: str, byte_start: int | None = None, byte_stop: int | None = None
    ) -> None:
        if self.resource_check is not None:
            self.resource_check()
        resolved = path.resolve(strict=True)
        if resolved != self.train_parquet:
            sequence = resolved.parent.name
            if resolved != resolve_train_event(
                self.raw_train_root, sequence, f"{sequence}/events.h5", self.allowed_sequences
            ):
                raise ValueError("physical read is outside the expansion TRAIN role")
        record = AccessRecord(str(resolved), purpose, byte_start, byte_stop, time.time_ns())
        assert self.journal_path is not None
        with self.journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._records.append(record)

    def event_path(self, sequence_id: str, source_value: object, purpose: str) -> Path:
        """Accept only exact TRAIN paths whose sequence belongs to this fixed role."""
        path = resolve_train_event(
            self.raw_train_root, sequence_id, str(source_value), self.allowed_sequences
        )
        self._record(path, purpose)
        return path


def expansion_bindings(
    metadata_path: Path, access: ExpansionTrainAccess, output: Path
) -> dict[str, Any]:
    """Hash media then bind one sequence at a time, with reusable verified partial tables."""
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_parquet(access.parquet_path("expansion raw binding metadata"))
    if (
        metadata.sample_token.duplicated().any()
        or not set(metadata.sequence_id) <= access.allowed_sequences
    ):
        raise ValueError("expansion metadata identity or source role mismatch")
    sources = {}
    for sequence, rows in metadata.groupby("sequence_id", sort=True):
        if rows.events_path.nunique() != 1:
            raise ValueError("sequence has multiple source paths")
        sources[str(sequence)] = access.event_path(
            str(sequence), rows.events_path.iloc[0], "expansion source resolution for hash check"
        )
    progress_path = output / "RAW_HASH_PROGRESS.json"
    hashes = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    if not set(hashes) <= set(sources) or any(
        record["path"] != str(sources[sequence]) for sequence, record in hashes.items()
    ):
        raise ValueError("hash resume journal names an unregistered source")
    for index, (sequence, source) in enumerate(sorted(sources.items()), 1):
        before = hashes.get(sequence)
        stamp = source.stat()
        reused = before is not None and (before["bytes"], before["mtime_ns"]) == (
            stamp.st_size,
            stamp.st_mtime_ns,
        )
        hashes = _stable_file_hashes({sequence: source}, progress_path, access.resource_check)
        access._record(
            source,
            "previous full hash reused; stat only"
            if reused
            else "completed full-file SHA256 byte read",
        )
        print(f"expansion source hash {index}/{len(sources)}:{sequence}", flush=True)
    identity = {
        "metadata": binding(metadata_path),
        "sources": hashes,
        "allowed_sequences": sorted(access.allowed_sequences),
        "producer_code": [
            binding(Path(__file__)),
            binding(Path(__file__).with_name("raw_event_binding.py")),
            binding(Path(__file__).with_name("stage70_clock.py")),
            binding(Path(__file__).with_name("event_v4_geometry.py")),
        ],
    }
    identity_hash = object_digest(identity)
    manifest_path = output / "RAW_BINDING_MANIFEST.json"
    if manifest_path.exists():
        complete = json.loads(manifest_path.read_text())
        signature = complete.pop("artifact_sha256")
        if object_digest(complete) != signature or complete["identity_sha256"] != identity_hash:
            raise ValueError("completed expansion binding identity mismatch")
        verify(complete["binding"])
        return {**complete, "artifact_sha256": signature}
    all_rows = []
    for sequence, rows in metadata.groupby("sequence_id", sort=True):
        partial_path = output / f"BINDINGS_{sequence}.json"
        if partial_path.exists():
            partial = json.loads(partial_path.read_text())
            signature = partial.pop("sha256")
            if partial["identity"] != identity_hash or object_digest(partial) != signature:
                raise ValueError("partial expansion binding identity changed")
            all_rows.extend(partial["rows"])
            continue
        path = sources[str(sequence)]
        bound = []
        with h5py.File(path, "r") as handle:
            _reject_external_links(handle, path)
            for name in ("events", "ms_to_idx", *(f"events/{key}" for key in "xytp")):
                if isinstance(handle.get(name, getlink=True), (h5py.SoftLink, h5py.ExternalLink)):
                    raise ValueError("indirect expansion HDF5 storage")
            datasets = {key: cast(h5py.Dataset, handle[f"events/{key}"]) for key in "xytp"}
            if any(dataset.is_virtual or dataset.external for dataset in datasets.values()):
                raise ValueError("external or virtual expansion event storage")
            if len({len(dataset) for dataset in datasets.values()}) != 1 or not len(datasets["t"]):
                raise ValueError("expansion event fields are empty or misaligned")
            times = datasets["t"]
            access.record_h5_slice(
                path, "timestamp monotonicity sampled at <=4096 sites", 0, len(times)
            )
            probes = np.asarray(times[:: max(1, len(times) // 4096)], dtype=np.int64)
            if np.any(np.diff(probes) < 0):
                raise ValueError("expansion timestamp rollback")
            coarse = np.asarray(cast(h5py.Dataset, handle["ms_to_idx"]), dtype=np.int64)
            access._record(path, "ms_to_idx and bounded polarity encoding probes")
            origin = 0 if len(coarse) > int(times[0]) // 1000 else int(times[0]) // 1000 * 1000
            polarity = _polarity_encoding(datasets["p"])
            for row in rows.itertuples(index=False):
                row = cast(Any, row)
                clock = context_clock(
                    row.event_windows_us, row.frame_timestamps_us, row.timestamp_us
                )
                windows = [tuple(int(value) for value in pair) for pair in row.event_windows_us]
                square = common_square_from_boxes(
                    _as_boxes(row.boxes_xyxy), (0, 1), margin_fraction=0.25, minimum_edge=8.0
                )
                roi_hash = object_digest(
                    {
                        "square_xyxy": square,
                        "canvas": [128, 128],
                        "event_x_offset_px": 5.0,
                        "source": "common_square_from_boxes_v4_t1_t2_margin025_min8",
                    }
                )
                for window_id, (start, end) in enumerate(windows):
                    access._record(
                        path,
                        f"exact timestamp search and boundary proof:{row.sample_token}:{window_id}",
                    )
                    first, stop = exact_window_bounds(
                        times, start, end, ms_to_idx=coarse, index_origin_us=origin
                    )
                    if (
                        (first and int(times[first - 1]) >= start)
                        or (first < len(times) and int(times[first]) < start)
                        or (stop < len(times) and int(times[stop]) < end)
                        or (stop > first and int(times[stop - 1]) >= end)
                    ):
                        raise ValueError("expansion half-open boundary proof failed")
                    record = RawWindowBinding(
                        sample_token=str(row.sample_token),
                        sequence_id=str(sequence),
                        track_id=str(row.track_id),
                        outer_fold=-1,
                        window_id=window_id,
                        parquet_row_identity=f"expansion:{row.sample_token}",
                        events_path_relative=f"{sequence}/events.h5",
                        h5_file_sha256=hashes[str(sequence)]["sha256"],
                        time_unit="us",
                        clock_origin_us=origin,
                        frame_to_event_clock_offset_us=clock.frame_to_event_offsets_us[window_id],
                        frame_to_event_clock_offset_drift_us=clock.frame_to_event_offsets_us[1]
                        - clock.frame_to_event_offsets_us[0],
                        window_start_us=start,
                        window_end_us=end,
                        observation_end_us=end,
                        target_anchor_us=int(row.timestamp_us),
                        target_anchor_event_clock_us=clock.label_anchor_event_us,
                        start_event_index=first,
                        stop_event_index=stop,
                        full_window_event_count=stop - first,
                        roi_event_count=None,
                        roi_transform_sha256=roi_hash,
                        roi_x0=square[0],
                        roi_y0=square[1],
                        roi_x1=square[2],
                        roi_y1=square[3],
                        event_x_offset_px=5.0,
                        polarity_encoding=polarity,
                        coordinate_system="eap_event_sensor_to_rgb_x_plus_5_then_common_roi_128",
                        read_probe_status="exact_timestamp_boundaries_verified_empty_windows_retained",
                        endpoint_delta01_us=windows[0][1] - windows[0][0],
                        endpoint_delta12_us=int(row.frame_timestamps_us[1])
                        - int(row.frame_timestamps_us[0]),
                    )
                    bound.append(asdict(record))
        current = path.stat()
        if (current.st_size, current.st_mtime_ns) != (
            hashes[str(sequence)]["bytes"],
            hashes[str(sequence)]["mtime_ns"],
        ):
            raise ValueError("raw source changed during expansion binding")
        partial = {"identity": identity_hash, "rows": bound}
        atomic_json(partial_path, {**partial, "sha256": object_digest(partial)})
        all_rows.extend(bound)
        print(f"expansion bindings:{sequence}, {len(bound) // 2} samples", flush=True)
    table = pd.DataFrame(all_rows).sort_values(["sample_token", "window_id"])
    if len(table) != 2 * len(metadata) or table.duplicated(["sample_token", "window_id"]).any():
        raise ValueError("incomplete expansion binding inventory")
    path = output / "RAW_BINDINGS.csv"
    table.to_csv(path, index=False)
    result = {
        "status": "EXPANSION_RAW_BOUND",
        "identity": identity,
        "identity_sha256": identity_hash,
        "binding": binding(path),
        "samples": len(metadata),
        "source_groups": len(sources),
        "zero_event_windows_retained": int(table.full_window_event_count.eq(0).sum()),
    }
    result["artifact_sha256"] = object_digest(result)
    atomic_json(output / "RAW_BINDING_MANIFEST.json", result)
    return result
