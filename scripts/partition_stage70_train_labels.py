"""Administrative public-TRAIN partitioning under document04; confirmation stays sealed."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify


def publish_partition(frame: pd.DataFrame, path: Path) -> None:
    """Close and sync each role-specific parquet before publication."""
    temporary = path.with_suffix(".parquet.tmp")
    with temporary.open("wb") as stream:
        frame.to_parquet(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    """Decode monolithic TRAIN labels administratively, emitting no target values or scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    campaign = Path(paths["campaign_root"]).resolve(strict=True)
    metadata_path = Path(paths["train_annotations"]).resolve(strict=True)
    labels_path = args.train_labels.resolve(strict=True)
    expected_labels = metadata_path.parent.parent / "annotations/train.parquet"
    if labels_path != expected_labels.resolve(strict=True):
        raise ValueError("only the resolved local public-TRAIN annotation file is authorized")
    if not campaign.is_relative_to(ROOT / "artifacts"):
        raise ValueError("partition output must remain inside this worktree")
    output = campaign / "role_partitions"
    manifest_path = output / "ROLE_PARTITIONS.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        signature = manifest.pop("artifact_sha256")
        if signature != object_digest(manifest):
            raise ValueError("completed role partition signature mismatch")
        for record in (*manifest["outputs"].values(), *manifest["sources"].values()):
            verify(record)
        print(
            "role partitions already published and verified; no label decode repeated", flush=True
        )
        return
    roles_path = campaign / "data_roles/DATA_ROLES.json"
    record = json.loads(roles_path.read_text())
    signature = record.pop("artifact_sha256")
    if object_digest(record) != signature:
        raise ValueError("fixed role signature changed")
    roles = record["roles"]
    mapping = dict(roles["sequence_to_group"])
    train_groups = set(roles["original"]) | set(roles["expansion"])
    confirmation_groups = set(roles["confirmation"])
    protected_groups = set(roles["protected"])
    if (
        train_groups & (confirmation_groups | protected_groups)
        or confirmation_groups & protected_groups
    ):
        raise ValueError("partition source roles overlap")
    with ResourceOwner(output, remaining_bytes=128 * 1024**2, planned_ram_bytes=1024**3):
        source_records = {
            "metadata": binding(metadata_path),
            "labels": binding(labels_path),
            "roles": binding(roles_path),
        }
        atomic_json(
            output / "ADMINISTRATOR_READ_INTENT.json",
            {
                "authority": "document04 monolithic public-TRAIN partition-builder paragraph",
                "sources": source_records,
                "columns": {"metadata": "all", "labels": "all"},
                "physical_scope": (
                    "one monolithic TRAIN row group; all role values decoded administratively"
                ),
                "training_and_selection_scope": (
                    "only original and EXPANSION_TRAIN rows emitted to the TRAIN view"
                ),
                "confirmation_scope": (
                    "copied to sealed partition; no label-derived diagnostics or values emitted"
                ),
                "protected_scope": (
                    "previously protected acquisition rows discarded; no protected view emitted"
                ),
                "exhaustive_physical_sandbox_claim": False,
            },
        )
        metadata = pd.read_parquet(metadata_path)
        labels = pd.read_parquet(labels_path)
        if (
            len(metadata) != 88744
            or len(labels) != 88744
            or metadata.sample_token.duplicated().any()
            or labels.sample_token.duplicated().any()
            or set(metadata.sample_token) != set(labels.sample_token)
        ):
            raise ValueError("public TRAIN row identity mismatch")
        labels = labels.set_index("sample_token").loc[metadata.sample_token].reset_index()
        for column in (
            "sample_token",
            "sequence_id",
            "track_id",
            "public_track_id",
            "timestamp_us",
        ):
            if not np.array_equal(metadata[column].to_numpy(), labels[column].to_numpy()):
                raise ValueError(f"public TRAIN metadata/annotation identity differs: {column}")
        if set(metadata.sequence_id) != set(mapping):
            raise ValueError("public TRAIN membership differs from fixed roles")
        groups = metadata.sequence_id.map(mapping)
        train = groups.isin(train_groups)
        confirmation = groups.isin(confirmation_groups)
        discarded = ~train & ~confirmation
        if not set(groups[discarded]) <= protected_groups:
            raise ValueError("unclassified source role cannot be silently discarded")
        # Only TRAIN targets enter checks. Confirmation values are merely copied below.
        train_labels = labels.loc[train].copy()
        previous = pd.read_csv(
            campaign / "preflight/HISTORICAL_AUDIT_ROWS.csv", usecols=["sample_token", "target_ttc"]
        )
        old_targets = (
            train_labels.set_index("sample_token").loc[previous.sample_token, "ttc"].to_numpy()
        )
        if not np.array_equal(
            old_targets.astype(np.float32), previous.target_ttc.to_numpy(dtype=np.float32)
        ):
            raise ValueError("original TRAIN TTC labels do not reproduce pinned float32 targets")
        views = {
            "train_metadata": metadata.loc[train].copy(),
            "train_labels": train_labels,
            "confirmation_metadata_SEALED": metadata.loc[confirmation].copy(),
            "confirmation_labels_SEALED": labels.loc[confirmation].copy(),
        }
        for name, frame in views.items():
            publish_partition(frame, output / f"{name}.parquet")
        seal = {
            "role": "TRAIN_CONFIRMATION_LOCKED",
            "groups": sorted(confirmation_groups),
            "permitted_pre_S76_operation": "byte hashing and membership metadata only",
            "requires": [
                "stable three-seed architecture replication",
                "verified acquisition and ancestor exclusions",
                "credible TRAIN-only Garl comparator",
                "final candidate decision lock",
                "three new-architecture and three Garl endpoints frozen",
                "single coordinated confirmation read",
            ],
            "not_a_physical_access_sandbox": True,
        }
        atomic_json(output / "CONFIRMATION_SEAL.json", seal)
        result = {
            "kind": "administrative_public_TRAIN_role_partitions",
            "sources": source_records,
            "outputs": {name: binding(output / f"{name}.parquet") for name in views},
            "confirmation_seal": binding(output / "CONFIRMATION_SEAL.json"),
            "metadata_counts": {
                "training_rows": int(train.sum()),
                "confirmation_rows": int(confirmation.sum()),
                "protected_rows_discarded": int(discarded.sum()),
                "training_groups": len(train_groups),
                "confirmation_groups": len(confirmation_groups),
            },
            "original8192_targets_float32_exact": True,
            "confirmation_targets_used_for_diagnostics_training_or_selection": False,
            "administrative_confirmation_label_decode": True,
        }
        result["artifact_sha256"] = object_digest(result)
        atomic_json(manifest_path, result)
        print(
            json.dumps({"status": "PARTITIONED_CONFIRMATION_SEALED", **result["metadata_counts"]}),
            flush=True,
        )


if __name__ == "__main__":
    main()
