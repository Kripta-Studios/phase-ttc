"""Build the pre-registered query subset and its shared crop geometry (repository venv).

``python -m operational.literature_baselines.manifest --out <artifacts dir> --docs <docs dir>``
writes ``SUBSET_MANIFEST.npz`` (artifacts) and ``SUBSET.json`` (docs).  Only public TRAIN40
labels are read; they are used for band stratification and, later, for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from operational.phase_geometry.data import fold_of, sha256_file, verified_file
from operational.phase_geometry.metrics import BANDS, band_index, json_safe

from .common import (
    PER_SEQUENCE,
    SUBSET_SEED,
    box_scale,
    common_square,
    select_subset,
    truncated,
)
from .frames import safe_shard

COLUMNS = [
    "sequence_id",
    "sample_token",
    "track_id",
    "timestamp_us",
    "frame_timestamps_us",
    "rgb_shard_paths",
    "rgb_member_paths",
    "boxes_xyxy",
]


def build(root: Path, per_sequence: int = PER_SEQUENCE) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return manifest arrays (manifest order) and a JSON summary."""
    data = pd.read_parquet(verified_file("train_data", root), columns=COLUMNS)
    labels = pd.read_parquet(
        verified_file("train_annotations", root), columns=["sample_token", "ttc", "frame_ttc"]
    )
    frame = data.merge(labels, on="sample_token", how="inner", validate="one_to_one")
    if len(frame) != 88744 or frame["sequence_id"].nunique() != 40:
        raise ValueError("Expected the 88,744 public TRAIN40 rows in 40 sequences")
    frame = frame.sort_values("sample_token", kind="stable").reset_index(drop=True)
    ttc = frame["ttc"].to_numpy(np.float64)
    bands = band_index(ttc)
    chosen = select_subset(
        frame["sample_token"].tolist(),
        frame["sequence_id"].tolist(),
        bands,
        per_sequence=per_sequence,
        seed=SUBSET_SEED,
    )
    rows = frame.iloc[chosen].reset_index(drop=True)
    records: dict[str, list[Any]] = {
        k: []
        for k in (
            "box_previous",
            "box_current",
            "square",
            "delta_us",
            "shard_previous",
            "member_previous",
            "shard_current",
            "member_current",
            "truncated",
            "box_size",
            "frame_ttc_matches",
        )
    }
    for row in rows.itertuples(index=False):
        stamps = [int(v) for v in row.frame_timestamps_us]
        if len(stamps) != 2 or stamps[1] != int(row.timestamp_us):
            raise ValueError(f"{row.sample_token}: expected frames [t-delta, t]")
        boxes = [np.asarray(b, dtype=np.float64) for b in row.boxes_xyxy]
        sequence = str(row.sequence_id)
        records["box_previous"].append(boxes[0])
        records["box_current"].append(boxes[1])
        records["square"].append(np.asarray(common_square(boxes)))
        records["delta_us"].append(stamps[1] - stamps[0])
        records["shard_previous"].append(safe_shard(str(row.rgb_shard_paths[0]), sequence))
        records["shard_current"].append(safe_shard(str(row.rgb_shard_paths[1]), sequence))
        records["member_previous"].append(str(row.rgb_member_paths[0]))
        records["member_current"].append(str(row.rgb_member_paths[1]))
        records["truncated"].append(truncated(boxes[1]))
        records["box_size"].append(box_scale(boxes[1]))
        records["frame_ttc_matches"].append(
            bool(np.isclose(float(row.frame_ttc[1]), float(row.ttc), rtol=0, atol=1e-9))
        )
    groups = rows["sequence_id"].to_numpy(dtype=str)
    folds, blocks = fold_of(groups, root)
    arrays: dict[str, Any] = {
        "sample_token": rows["sample_token"].to_numpy(dtype=str),
        "sequence_id": groups,
        "track_id": rows["track_id"].to_numpy(dtype=str),
        "timestamp_us": rows["timestamp_us"].to_numpy(np.int64),
        "ttc": rows["ttc"].to_numpy(np.float64),
        "band": band_index(rows["ttc"].to_numpy(np.float64)),
        "fold": folds,
        "delta_us": np.asarray(records["delta_us"], dtype=np.int64),
        "box_previous": np.stack(records["box_previous"]),
        "box_current": np.stack(records["box_current"]),
        "square": np.stack(records["square"]),
        "shard_previous": np.asarray(records["shard_previous"]),
        "member_previous": np.asarray(records["member_previous"]),
        "shard_current": np.asarray(records["shard_current"]),
        "member_current": np.asarray(records["member_current"]),
        "truncated": np.asarray(records["truncated"], dtype=bool),
        "box_size": np.asarray(records["box_size"], dtype=np.float64),
    }
    token_digest = hashlib.sha256("\n".join(arrays["sample_token"]).encode()).hexdigest()
    per_band = Counter(BANDS[b] for b in arrays["band"])
    population = Counter(BANDS[b] for b in bands)
    summary = {
        "schema": "phase_literature_baselines_subset_v1",
        "seed": SUBSET_SEED,
        "per_sequence": per_sequence,
        "queries": len(rows),
        "sequences": int(len(set(groups))),
        "token_list_sha256": token_digest,
        "band_counts_subset": {b: per_band.get(b, 0) for b in BANDS},
        "band_counts_population": {b: population.get(b, 0) for b in BANDS},
        "per_sequence_counts": dict(sorted(Counter(groups.tolist()).items())),
        "delta_us_values": sorted({int(v) for v in arrays["delta_us"]}),
        "truncated_fraction": float(arrays["truncated"].mean()),
        "box_size_quantiles_5_25_50_75_95": np.quantile(
            arrays["box_size"], [0.05, 0.25, 0.5, 0.75, 0.95]
        ).tolist(),
        "frame_ttc_at_query_equals_ttc_fraction": float(np.mean(records["frame_ttc_matches"])),
        "blocks": blocks,
        "test12_private_labels_read": False,
    }
    return arrays, summary


def main() -> None:
    """Write the manifest NPZ and the subset summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--per-sequence", type=int, default=PER_SEQUENCE)
    args = parser.parse_args()
    arrays, summary = build(args.root.resolve(), args.per_sequence)
    args.out.mkdir(parents=True, exist_ok=True)
    args.docs.mkdir(parents=True, exist_ok=True)
    path = args.out / "SUBSET_MANIFEST.npz"
    np.savez_compressed(path, **arrays)
    summary["manifest_sha256"] = sha256_file(path)
    summary["manifest_source_sha256"] = sha256_file(Path(__file__))
    (args.docs / "SUBSET.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe({k: summary[k] for k in summary if k != "blocks"}), indent=1))


if __name__ == "__main__":
    main()
