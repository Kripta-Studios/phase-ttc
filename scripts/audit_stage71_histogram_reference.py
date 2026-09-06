"""Compare64 complete E128 TRAIN samples against an independent4-D histogram."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import h5py
import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding
from e_jepa_ttc.data.event_context128_reader import CountReader
from e_jepa_ttc.data.raw_event_binding import ReadOnlyTrainAccess, _reject_external_links


def main() -> None:
    """Read exact authorized slices; use histogramdd instead of production integer scatter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    campaign = Path(paths["campaign_root"]).resolve(strict=True)
    if not campaign.is_relative_to(ROOT / "artifacts"):
        raise ValueError("audit output must remain in this worktree")
    cache_root = campaign / "counts_probe64_v2"
    output = campaign / "histogram_reference64"
    if (output / "HISTOGRAM_PARITY.json").exists():
        raise ValueError("completed histogram audit exists; do not overwrite evidence")
    reader = CountReader(cache_root)
    if len(reader.metadata) != 64:
        raise ValueError("the prescribed reference probe must have64 complete samples")
    bounds = pd.read_csv(cache_root / "RAW_BINDINGS.csv").set_index(["sample_token", "window_id"])
    manifest = json.loads((cache_root / "CACHE_MANIFEST.json").read_text())
    records = []
    with ResourceOwner(output, remaining_bytes=4 * 1024**2, planned_ram_bytes=1024**3) as owner:
        access = ReadOnlyTrainAccess(
            Path(paths["raw_train_root"]),
            Path(paths["train_annotations"]),
            journal_path=output / "SOURCE_READS.jsonl",
            resource_check=owner.check,
        )
        for token in reader.metadata.sample_token:
            cached = reader.observation(str(token))
            for window_id in (0, 1):
                row = cast(pd.Series, bounds.loc[(token, window_id)])
                sequence = str(row.sequence_id)
                path = access.event_path(
                    sequence, row.events_path_relative, "independent histogram"
                )
                expected = manifest["identity"]["sources"][sequence]
                stat = path.stat()
                if (str(path), stat.st_size, stat.st_mtime_ns) != (
                    expected["path"],
                    expected["bytes"],
                    expected["mtime_ns"],
                ):
                    raise ValueError("raw source changed since its verified hash")
                start, end = int(row.window_start_us), int(row.window_end_us)
                edges_t = np.array([(end - start) * i // 8 for i in range(9)], dtype=np.int64)
                roi = [float(cast(Any, row[f"roi_{axis}"])) for axis in ("x0", "y0", "x1", "y1")]
                edges = [
                    edges_t,
                    np.array([-0.5, 0.5, 1.5]),
                    np.linspace(roi[0], roi[2], 129),
                    np.linspace(roi[1], roi[3], 129),
                ]
                counts = np.zeros((8, 2, 128, 128), dtype=np.uint64)
                raw_count = roi_count = 0
                with h5py.File(path, "r") as handle:
                    _reject_external_links(handle, path)
                    datasets = {key: cast(h5py.Dataset, handle[f"events/{key}"]) for key in "xytp"}
                    for first in range(
                        int(row.start_event_index), int(row.stop_event_index), 262144
                    ):
                        last = min(int(row.stop_event_index), first + 262144)
                        access.record_h5_slice(
                            path, "independent histogram physical read", first, last
                        )
                        x, y, t, p = [np.asarray(datasets[key][first:last]) for key in "xytp"]
                        if row.polarity_encoding != "zero_one" or not np.isin(p, [0, 1]).all():
                            raise ValueError("reference expects the verified original0/1 polarity")
                        if np.any(t < start) or np.any(t >= end):
                            raise ValueError("reference window is not half open")
                        raw_count += len(t)
                        keep = (x >= 0) & (x < 1280) & (y >= 0) & (y < 720)
                        calibrated_x = x.astype(np.float64) + 5
                        keep &= (
                            (calibrated_x >= roi[0])
                            & (calibrated_x < roi[2])
                            & (y >= roi[1])
                            & (y < roi[3])
                        )
                        roi_count += int(keep.sum())
                        histogram, _ = np.histogramdd(
                            (
                                t[keep].astype(np.int64) - start,
                                p[keep],
                                calibrated_x[keep],
                                y[keep],
                            ),
                            bins=edges,
                        )
                        counts += histogram.transpose(0, 1, 3, 2).astype(np.uint64)
                delta = counts.astype(np.int64) - cached.counts[window_id].astype(np.int64)
                record = {
                    "sample_token": str(token),
                    "window_id": window_id,
                    "raw_events": raw_count,
                    "roi_events": roi_count,
                    "max_absolute_difference": int(np.abs(delta).max()),
                    "different_cells": int(np.count_nonzero(delta)),
                    "count_conservation": int(counts.sum()) == roi_count,
                }
                records.append(record)
            atomic_json(
                output / "PROGRESS.json",
                {"complete_samples": len(records) // 2, "windows": records},
            )
            print(f"independent histogram {len(records) // 2}/64 samples", flush=True)
        result = {
            "status": "PASS"
            if all(r["different_cells"] == 0 and r["count_conservation"] for r in records)
            else "FAIL",
            "samples": 64,
            "windows": len(records),
            "optimizer_updates": 0,
            "cache_manifest": binding(cache_root / "CACHE_MANIFEST.json"),
            "reference_code": binding(Path(__file__)),
            "windows_evidence": records,
        }
        atomic_json(output / "HISTOGRAM_PARITY.json", result)
        if result["status"] != "PASS":
            raise RuntimeError("independent raw histogram does not match production counts")


if __name__ == "__main__":
    main()
