"""Check metadata clocks and pinned historical raw bindings on TRAIN only.

This audit does not assign fresh data roles or load any TTC annotation parquet.
Raw event probes are confined to hash-selected original development tokens.
"""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import h5py
import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, digest
from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from e_jepa_ttc.data.raw_event_binding import (
    EXPECTED_SEQUENCES,
    ReadOnlyTrainAccess,
    _reject_external_links,
    _stable_file_hashes,
    select_hash_probe_tokens,
)
from e_jepa_ttc.data.stage70_clock import context_clock

MEMBER = "run/frozen_audit/extracted_input/run/stage63/X3_RAW_BINDING_V2.csv"
COLUMNS = [
    "sequence_id",
    "sample_token",
    "track_id",
    "timestamp_us",
    "frame_timestamps_us",
    "events_path",
    "event_windows_us",
    "boxes_xyxy",
]


def main() -> None:
    """Emit a resumable metadata and raw-clock audit with explicit read scope."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--garl-root", type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    output = args.output_root.resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("audit output must be inside this worktree artifacts")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "LABEL_TIME_CHARTER.json").exists():
        raise ValueError("refusing to replace a completed time charter")
    source_path = (
        args.garl_root.resolve(strict=True) / "garl_ttc_benchmark/build_garlttc_dataset.py"
    )
    source_text = source_path.read_text(encoding="utf-8")
    for fragment in (
        "target = selected[-1]",
        "'timestamp_us': int(target['timestamp'])",
        "'ttc': float(target['ttc'])",
        "source_starts = [int(anno['meta']['corr_exposure_start_timestamp_us']) "
        "for anno in record]",
    ):
        if fragment not in source_text:
            raise ValueError("source anchor implementation differs from inspected source")
    historical = pd.read_csv(
        Path(paths["campaign_root"]) / "preflight/HISTORICAL_AUDIT_ROWS.csv",
        usecols=["sample_token", "sequence_id", "track_id", "outer_fold"],
        dtype=str,
    )
    archive_path = Path(paths["results_zip"])
    if digest(archive_path) != "1b43ef8410a787bb4af37674f84afc69c3642b81431f5d3f233821991bf04afb":
        raise ValueError("immutable results ZIP identity changed")
    with zipfile.ZipFile(archive_path) as archive:
        old = pd.read_csv(io.BytesIO(archive.read(MEMBER)))
    if len(old) != 16384 or old.duplicated(["sample_token", "window_id"]).any():
        raise ValueError("historical binding is not two unique windows for 8192 rows")
    access = ReadOnlyTrainAccess(
        Path(paths["raw_train_root"]),
        Path(paths["train_annotations"]),
        journal_path=output / "SOURCE_READS.jsonl",
    )
    source = pd.read_parquet(
        access.parquet_path("TRAIN metadata columns only: " + ",".join(COLUMNS)), columns=COLUMNS
    )
    if source.sample_token.duplicated().any():
        raise ValueError("TRAIN metadata sample IDs are not unique")
    counts = source.groupby("sequence_id").agg(
        rows=("sample_token", "size"), tracks=("track_id", "nunique")
    )
    counts.to_csv(output / "TRAIN_METADATA_COUNTS.csv")
    atomic_json(
        output / "TRAIN_METADATA_INVENTORY.json",
        {
            "source": binding(access.train_parquet),
            "columns_read": COLUMNS,
            "rows": len(source),
            "groups": len(counts),
            "labels_read": False,
            "roles_assigned": False,
            "acquisition_grouping": "NOT_YET_ESTABLISHED",
            "source_read_scope": (
                "Parquet projected metadata columns; no annotations/train.parquet read"
            ),
        },
    )
    joined = historical.merge(
        source, on=["sample_token", "sequence_id", "track_id"], validate="one_to_one"
    )
    if len(joined) != 8192 or set(joined.sequence_id) != EXPECTED_SEQUENCES:
        raise ValueError("original development universe is not exactly recovered")
    old_index = old.set_index(["sample_token", "window_id"], verify_integrity=True)
    clocks: dict[str, Any] = {}
    rows = []
    raw_paths: dict[str, Path] = {}
    for row in joined.itertuples(index=False):
        row = cast(Any, row)
        windows = tuple(tuple(int(v) for v in pair) for pair in row.event_windows_us)
        frames = tuple(int(v) for v in row.frame_timestamps_us)
        clock = context_clock(windows, frames, int(row.timestamp_us))
        square = common_square_from_boxes(
            row.boxes_xyxy, (0, 1), margin_fraction=0.25, minimum_edge=8
        )
        for i, (start, end) in enumerate(windows):
            prior = old_index.loc[(row.sample_token, i)]
            if (
                int(prior.window_start_us) != start
                or int(prior.window_end_us) != end
                or int(prior.target_anchor_us) != frames[1]
                or int(prior.target_anchor_event_clock_us) != clock.label_anchor_event_us
                or int(prior.outer_fold) != int(row.outer_fold)
                or str(prior.sequence_id) != row.sequence_id
                or str(prior.track_id) != row.track_id
                or float(prior.event_x_offset_px) != 5.0
            ):
                raise ValueError(f"historical clock/identity mismatch: {row.sample_token}")
            np.testing.assert_allclose(
                [prior.roi_x0, prior.roi_y0, prior.roi_x1, prior.roi_y1],
                square,
                rtol=0,
                atol=1e-10,
            )
        clocks[row.sample_token] = asdict(clock)
        rows.append(
            {
                "sample_token": row.sample_token,
                "sequence_id": row.sequence_id,
                "track_id": row.track_id,
                "outer_fold": int(row.outer_fold),
                "annotation_timestamp_us": int(row.timestamp_us),
                "label_anchor_event_us": clock.label_anchor_event_us,
                "gap_us": clock.gap_us,
                **{
                    f"midpoint_twice_relative_us_{i}": value
                    for i, value in enumerate(clock.midpoint_twice_relative_us)
                },
            }
        )
        if row.sequence_id not in raw_paths:
            raw_paths[row.sequence_id] = access.event_path(
                row.sequence_id, row.events_path, "full raw hash and original-token clock probes"
            )
    pd.DataFrame(rows).to_csv(output / "CONTEXT_CLOCK_ROWS.csv", index=False)
    print(
        f"Metadata/ROI/time identity verified for {len(rows)} old rows; "
        "hashing nine raw TRAIN files",
        flush=True,
    )
    hashes = _stable_file_hashes(raw_paths, output / "RAW_HASH_PROGRESS.json")
    for sequence, record in hashes.items():
        pinned = set(old.loc[old.sequence_id == sequence, "h5_file_sha256"])
        if pinned != {record["sha256"]}:
            raise ValueError(f"raw HDF5 differs from historical binding: {sequence}")
    tokens = select_hash_probe_tokens(old, count=32)
    probes = []
    for sequence, path in sorted(raw_paths.items()):
        with h5py.File(path, "r") as handle:
            # Check parent links before any descendant resolution.
            for key in ("events", "ms_to_idx", "t_offset"):
                link = handle.get(key, getlink=True)
                if isinstance(link, (h5py.ExternalLink, h5py.SoftLink)):
                    raise ValueError(f"indirect HDF5 source forbidden: {key}")
            _reject_external_links(handle, path)
            for key in ("events/t", "events/x", "events/y", "events/p", "ms_to_idx"):
                dataset = handle[key]
                if not isinstance(dataset, h5py.Dataset):
                    raise ValueError("required HDF5 object is not a dataset")
                if isinstance(handle.get(key, getlink=True), h5py.SoftLink) or dataset.is_virtual:
                    raise ValueError("indirect or virtual HDF5 data forbidden")
            t_offset = int(np.asarray(handle["t_offset"]).item()) if "t_offset" in handle else None
            timestamps = cast(h5py.Dataset, handle["events/t"])
            selected = old.loc[(old.sequence_id == sequence) & old.sample_token.isin(tokens)]
            for row in selected.itertuples(index=False):
                row = cast(Any, row)
                start, stop = int(row.start_event_index), int(row.stop_event_index)
                if not 0 <= start < stop <= len(timestamps):
                    raise ValueError("invalid pinned event indices")
                access.record_h5_slice(
                    path,
                    "32-token time-boundary probe",
                    max(0, start - 1),
                    min(stop + 1, len(timestamps)),
                )
                first, last = int(timestamps[start]), int(timestamps[stop - 1])
                before = int(timestamps[start - 1]) if start else None
                after = int(timestamps[stop]) if stop < len(timestamps) else None
                if not (row.window_start_us <= first <= last < row.window_end_us):
                    raise ValueError("event timestamps violate half-open window")
                if before is not None and before >= row.window_start_us:
                    raise ValueError("left boundary is not exact")
                if after is not None and after < row.window_end_us:
                    raise ValueError("right boundary is not exact")
                probes.append(
                    {
                        "sample_token": row.sample_token,
                        "window_id": int(row.window_id),
                        "first_observed_event_us": first,
                        "last_observed_event_us": last,
                        "previous_event_us": before,
                        "next_event_us": after,
                        "raw_t_offset_us": t_offset,
                        **clocks[row.sample_token],
                    }
                )
    frame = pd.DataFrame(rows)
    atomic_json(
        output / "LABEL_TIME_CHARTER.json",
        {
            "status": "ORIGINAL_8192_CLOCK_AND_ROI_VERIFIED",
            "labels_read": False,
            "source_semantics": (
                "TTC assigned to selected[-1]; data timestamp_us and label timestamp_us "
                "share that annotation"
            ),
            "source_builder": binding(source_path),
            "historical_binding_member": MEMBER,
            "raw_files": hashes,
            "metadata_source": binding(access.train_parquet),
            "checked_token_count": 32,
            "checked_window_count": len(probes),
            "examples": probes,
            "annotation_to_event_conversion": (
                "event_us = annotation_us - (frame_timestamp_us[i] - event_window_end_us[i]); "
                "per-window offsets checked against <=5us drift"
            ),
            "anchor": "later selected frame, mapped to second event-window endpoint",
            "raw_t_offset_handling": (
                "recorded as file metadata; no second offset added "
                "to already bound stored timestamps"
            ),
            "model_time_conversion": (
                "((context_start_us-anchor_us)+(context_end_us-anchor_us))/2000000; "
                "subtract integers before float conversion"
            ),
            "binning": "start + ((end-start)*i)//8, i=0..8; half-open",
            "common_roi": (
                "exact original common square; x offset +5px; new canvas128 does not change ROI"
            ),
            "strictly_ordered_midpoints_all_rows": True,
            "gap_us_min": int(frame.gap_us.min()),
            "gap_us_max": int(frame.gap_us.max()),
            "midpoint_approximation": "each accumulated context represents its physical midpoint",
            "limitation": (
                "Numeric binding and inspected release source support the anchor; "
                "source-build execution provenance is not an independent replay "
                "of original annotation acquisition"
            ),
            "access": access.audit_summary(),
            "optimizer_updates": 0,
        },
    )
    print("Completed 32-token/64-window raw clock probe", flush=True)


if __name__ == "__main__":
    main()
