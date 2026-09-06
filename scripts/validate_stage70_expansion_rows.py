"""Validate the frozen D row list without replacing rejected samples."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from e_jepa_ttc.data.raw_event_binding import _as_boxes
from e_jepa_ttc.data.stage70_clock import context_clock


def main() -> None:
    """Check TRAIN label/clock/ROI contracts after target-free sampling; no salvage draw."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if not campaign.is_relative_to(ROOT / "artifacts"):
        raise ValueError("expansion checks must remain inside this worktree")
    inventory_path = campaign / "expansion_inventory/EXPANSION_CANDIDATES.json"
    inventory = json.loads(inventory_path.read_text())
    signature = inventory.pop("artifact_sha256")
    if signature != object_digest(inventory):
        raise ValueError("prospective D candidate list changed")
    partitions = json.loads((campaign / "role_partitions/ROLE_PARTITIONS.json").read_text())
    signature = partitions.pop("artifact_sha256")
    if signature != object_digest(partitions):
        raise ValueError("role partition signature changed")
    output = campaign / "expansion_usability"
    if (output / "ROW_USABILITY.json").exists():
        raise ValueError("D row usability already recorded; no replacement sampling")
    with ResourceOwner(output, remaining_bytes=64 * 1024**2, planned_ram_bytes=1024**3):
        selected_path = verify(inventory["selected_metadata"])
        selected = pd.read_csv(selected_path)
        metadata_path = verify(partitions["outputs"]["train_metadata"])
        labels_path = verify(partitions["outputs"]["train_labels"])
        columns = [
            "sample_token",
            "sequence_id",
            "track_id",
            "timestamp_us",
            "frame_timestamps_us",
            "event_windows_us",
            "boxes_xyxy",
            "events_path",
        ]
        metadata = pd.read_parquet(metadata_path, columns=columns).set_index("sample_token")
        labels = pd.read_parquet(
            labels_path, columns=["sample_token", "sequence_id", "timestamp_us", "ttc", "frame_ttc"]
        ).set_index("sample_token")
        metadata, labels = metadata.loc[selected.sample_token], labels.loc[selected.sample_token]
        for column in ("sequence_id", "track_id", "timestamp_us", "events_path"):
            if not np.array_equal(metadata[column].to_numpy(), selected[column].to_numpy()):
                raise ValueError("selected D metadata identity changed")
        for column in ("sequence_id", "timestamp_us"):
            if not np.array_equal(metadata[column].to_numpy(), labels[column].to_numpy()):
                raise ValueError("D annotation identity differs from its metadata")
        rejected, usable = {}, []
        maximum_drift, maximum_gap = 0, 0
        for token, row in metadata.iterrows():
            target = float(labels.loc[token, "ttc"])
            if not np.isfinite(target) or not (target < 0 or target > 0.1):
                rejected[str(token)] = "invalid_signed_TTC_target_domain"
                continue
            frame_targets = np.asarray(labels.loc[token, "frame_ttc"], dtype=np.float64)
            if frame_targets.shape != (2,) or float(frame_targets[-1]) != target:
                rejected[str(token)] = "label_not_attached_to_later_frame"
                continue
            try:
                clock = context_clock(
                    row.event_windows_us, row.frame_timestamps_us, row.timestamp_us
                )
                common_square_from_boxes(
                    _as_boxes(row.boxes_xyxy), (0, 1), margin_fraction=0.25, minimum_edge=8.0
                )
            except ValueError as error:
                rejected[str(token)] = f"metadata_clock_or_ROI_invalid:{error}"
                continue
            maximum_drift = max(
                maximum_drift,
                abs(clock.frame_to_event_offsets_us[1] - clock.frame_to_event_offsets_us[0]),
            )
            maximum_gap = max(maximum_gap, abs(clock.gap_us))
            usable.append(str(token))
        usable_set = set(usable)
        train_metadata = metadata.loc[usable].reset_index()
        train_metadata["outer_fold"] = -1
        train_metadata.to_parquet(output / "USABLE_METADATA.parquet", index=False)
        train_labels = labels.loc[usable].reset_index()
        train_labels.to_parquet(output / "USABLE_TRAIN_LABELS.parquet", index=False)
        pd.DataFrame(sorted(rejected.items()), columns=["sample_token", "reason"]).to_csv(
            output / "REJECTED_ROWS.csv", index=False
        )
        folds = {
            fold: {
                "original_train_tokens": value["original_train_tokens"],
                "additional_tokens": [
                    token for token in value["additional_tokens"] if token in usable_set
                ],
            }
            for fold, value in inventory["folds"].items()
        }
        row_counts = {
            fold: len(value["original_train_tokens"]) + len(value["additional_tokens"])
            for fold, value in folds.items()
        }
        groups = train_metadata.sequence_id.nunique()
        result = {
            "status": "ROW_METADATA_USABLE_RAW_BINDING_PENDING"
            if groups >= 6 and min(row_counts.values()) >= 16384
            else "D_DATA_PREREQUISITE_BLOCKED",
            "frozen_candidate_inventory": binding(inventory_path),
            "sampling_replacements": 0,
            "usable_metadata": binding(output / "USABLE_METADATA.parquet"),
            "usable_labels": binding(output / "USABLE_TRAIN_LABELS.parquet"),
            "rejections": binding(output / "REJECTED_ROWS.csv"),
            "usable_additional_rows": len(usable),
            "additional_groups": int(groups),
            "fold_rows": row_counts,
            "folds": folds,
            "maximum_frame_event_offset_drift_us": maximum_drift,
            "maximum_absolute_context_gap_us": maximum_gap,
            "confirmation_labels_read": False,
            "raw_hdf5_verified": False,
        }
        result["artifact_sha256"] = object_digest(result)
        atomic_json(output / "ROW_USABILITY.json", result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "status",
                        "usable_additional_rows",
                        "additional_groups",
                        "fold_rows",
                    )
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
