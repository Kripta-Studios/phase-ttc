"""Freeze target-free D candidate sampling before raw/target usability checks."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner, disk_reservation
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.data.stage70_roles import SamplingRow, resolve_train_event, sample_expansion


def main() -> None:
    """Use only TRAIN metadata and path existence; never inspect a candidate TTC value."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    campaign = Path(paths["campaign_root"]).resolve(strict=True)
    if not campaign.is_relative_to(ROOT / "artifacts"):
        raise ValueError("expansion outputs must remain in this worktree")
    output = campaign / "expansion_inventory"
    if (output / "EXPANSION_CANDIDATES.json").exists():
        raise ValueError("prospective candidate identities already recorded; do not resample")
    partitions_path = campaign / "role_partitions/ROLE_PARTITIONS.json"
    partitions = json.loads(partitions_path.read_text())
    signature = partitions.pop("artifact_sha256")
    if signature != object_digest(partitions):
        raise ValueError("role partition signature mismatch")
    metadata_path = verify(partitions["outputs"]["train_metadata"])
    role_path = campaign / "data_roles/DATA_ROLES.json"
    roles = json.loads(role_path.read_text())["roles"]
    mapping = dict(roles["sequence_to_group"])
    allowed = frozenset(seq for seq, group in mapping.items() if group in roles["expansion"])
    original = pd.read_csv(campaign / "counts_original8192/metadata.csv")
    if len(original) != 8192 or set(original.outer_fold) != {0, 1, 2}:
        raise ValueError("original frozen endpoint universe changed")
    per_fold_old = {
        str(fold): sorted(original.loc[original.outer_fold != fold, "sample_token"])
        for fold in range(3)
    }
    limit = max(32768 - len(tokens) for tokens in per_fold_old.values())
    with ResourceOwner(output, remaining_bytes=16 * 1024**2, planned_ram_bytes=1024**3):
        metadata = pd.read_parquet(
            metadata_path,
            columns=["sample_token", "sequence_id", "track_id", "timestamp_us", "events_path"],
        )
        metadata = metadata.loc[metadata.sequence_id.isin(allowed)].copy()
        missing, sources = {}, {}
        for sequence, rows in metadata.groupby("sequence_id", sort=True):
            if rows.events_path.nunique() != 1:
                raise ValueError("expansion sequence maps to multiple physical source names")
            try:
                source = resolve_train_event(
                    Path(paths["raw_train_root"]),
                    str(sequence),
                    str(rows.events_path.iloc[0]),
                    allowed,
                )
            except FileNotFoundError:
                missing.update(
                    (str(token), "local_raw_media_missing") for token in rows.sample_token
                )
                continue
            stat = source.stat()
            sources[str(sequence)] = {
                "path": str(source),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        eligible = metadata.loc[~metadata.sample_token.isin(missing)]
        rows = [
            SamplingRow(
                str(row.sample_token),
                str(row.sequence_id),
                str(row.track_id),
                int(row.timestamp_us),
            )
            for row in eligible.itertuples(index=False)
        ]
        selected, rejected = sample_expansion(
            rows, sequence_to_group=mapping, limit=limit, prefix=roles["role_prefix"]
        )
        rejected.update(missing)
        selected_metadata = eligible.set_index("sample_token").loc[selected].reset_index()
        selected_metadata.to_csv(output / "SELECTED_METADATA.csv", index=False)
        pd.DataFrame(sorted(rejected.items()), columns=["sample_token", "reason"]).to_csv(
            output / "REJECTED_METADATA.csv", index=False
        )
        folds = {
            fold: {
                "original_train_tokens": old,
                "additional_tokens": selected[: 32768 - len(old)],
                "total_before_usability_checks": len(old) + min(len(selected), 32768 - len(old)),
            }
            for fold, old in per_fold_old.items()
        }
        group_counts = selected_metadata.sequence_id.map(mapping).value_counts().to_dict()
        estimated_bytes = len(selected) * (1048576 + 4096) + 4 * 1024**2
        result = {
            "status": "METADATA_CANDIDATES_RESERVED_CLOCK_AND_TARGET_USABILITY_PENDING",
            "sampling_prefix": roles["role_prefix"],
            "sampling_is_target_free": True,
            "no_label_based_replacement": True,
            "minimum_anchor_spacing_us": 20000,
            "source_metadata": binding(metadata_path),
            "fixed_roles": binding(role_path),
            "selected_metadata": binding(output / "SELECTED_METADATA.csv"),
            "rejected_metadata": binding(output / "REJECTED_METADATA.csv"),
            "physical_sources_stat_only": sources,
            "additional_groups": len(group_counts),
            "selected_group_row_counts": group_counts,
            "additional_rows": len(selected),
            "folds": folds,
            "uint16_cache_reservation_estimate": disk_reservation(output, estimated_bytes).record(),
            "not_yet_D_available": [
                "raw source hashes and binding",
                "common ROI and physical clock checks",
                "selected TRAIN target usability",
                "full execution resource reservation",
            ],
        }
        result["artifact_sha256"] = object_digest(result)
        atomic_json(output / "EXPANSION_CANDIDATES.json", result)
        print(
            json.dumps(
                {
                    "additional_groups": len(group_counts),
                    "additional_rows": len(selected),
                    "fold_rows_before_usability": {
                        fold: value["total_before_usability_checks"]
                        for fold, value in folds.items()
                    },
                    "cache_reservation_passes": result["uint16_cache_reservation_estimate"][
                        "passes"
                    ],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
