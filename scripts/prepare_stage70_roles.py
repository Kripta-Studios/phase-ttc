"""Reserve public-TRAIN roles from metadata and prior cache membership evidence."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, digest, object_digest
from e_jepa_ttc.data.raw_event_binding import EXPECTED_SEQUENCES
from e_jepa_ttc.data.stage70_roles import assign_train_roles


def main() -> None:
    """Publish prospective locked identities without reading fresh TTC labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    output = args.output_root.resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("output must be local campaign artifacts")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "DATA_ROLES.json").exists():
        raise ValueError("refusing to regenerate locked role identities")
    historical = args.historical_root.resolve(strict=True)
    split = historical / "data/splits/eap_pilot12_v1.json"
    if digest(split) != "39bf3943c728a7455d025cf60ff7f1d6fa60d6c126faa4c8c79463813c24147f":
        raise ValueError("historical pilot membership bytes mismatch")
    assignment = json.loads(split.read_text(encoding="utf-8"))["assignments"]
    if set(assignment["train"]) != EXPECTED_SEQUENCES:
        raise ValueError("historical TRAIN sequence set changed")
    protected = set(assignment["validation"])
    exposed = set(EXPECTED_SEQUENCES) | protected
    entries: list[dict[str, Any]] = [
        {
            "source": binding(split),
            "consumed_fields": ["assignments"],
            "original_development": sorted(EXPECTED_SEQUENCES),
            "prior_validation_membership": sorted(protected),
        }
    ]
    for manifest in sorted((historical / "artifacts/cache").glob("*/manifest.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        selected = data.get("selection", {}).get(
            "selected_count_by_split_sequence_track_bucket", {}
        )
        membership: dict[str, set[str]] = {}
        for key, count in selected.items():
            parts = key.split("|")
            if len(parts) < 2 or int(count) < 0:
                raise ValueError("unrecognized historical selection membership schema")
            if int(count) > 0:
                membership.setdefault(parts[0], set()).add(parts[1])
        for role, sequences in membership.items():
            exposed.update(sequences)
            if role != "train":
                protected.update(sequences)
        entries.append(
            {
                "source": binding(manifest),
                "consumed_fields": [
                    "selection.selected_count_by_split_sequence_track_bucket membership",
                    "source_event_cache",
                    "sources.subset_manifest",
                ],
                "membership": {key: sorted(value) for key, value in membership.items()},
                "cache_parent": data.get("source_event_cache"),
                "subset_parent": data.get("sources", {}).get("subset_manifest"),
                "membership_known_directly": bool(membership),
            }
        )
    dynamics = historical / "artifacts/manifests/eap_level_dynamics_v1.json"
    if dynamics.exists():
        data = json.loads(dynamics.read_text(encoding="utf-8"))
        membership = data["split_assignments"]
        for role, sequences in membership.items():
            exposed.update(sequences)
            if role != "train":
                protected.update(sequences)
        entries.append(
            {
                "source": binding(dynamics),
                "consumed_fields": ["split_assignments"],
                "membership": membership,
            }
        )
    metadata_path = Path(paths["train_annotations"]).resolve(strict=True)
    metadata = pd.read_parquet(metadata_path, columns=["sequence_id", "seq_name", "sample_token"])
    if metadata.sample_token.duplicated().any() or len(metadata) != 88744:
        raise ValueError("public TRAIN inventory identity changed")
    if metadata.seq_name.fillna("").ne("").any():
        raise ValueError("named sequence metadata requires acquisition-group audit")
    sequences = set(metadata.sequence_id.astype(str))
    if not exposed <= sequences:
        raise ValueError("historical exposure membership is not contained in public TRAIN")
    resolution_path = ROOT / "configs/protocol/stage70_role_hash_resolution.json"
    resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
    roles = assign_train_roles(
        {seq: seq for seq in sequences},
        original_sequences=sorted(EXPECTED_SEQUENCES),
        historically_exposed_sequences=sorted(exposed),
        protected_sequences=sorted(protected),
        role_prefix=resolution["document04_role_hash_prefix"],
    )
    ledger = {
        "kind": "historical_membership_evidence",
        "entries": entries,
        "documented_exposed_sequences": sorted(exposed),
        "protected_prior_validation_sequences": sorted(protected),
        "read_scope": (
            "Historical metadata manifests deserialized; membership and source-reference fields "
            "consumed. No protected prediction arrays or fresh TTC labels read."
        ),
        "exhaustive_process_access_audit": False,
        "remaining_confirmation_requirement": (
            "Verify complete fit/selection/normalization/pretext ancestor lineage and "
            "acquisition exclusions before S76 may open confirmation."
        ),
    }
    ledger["artifact_sha256"] = object_digest(ledger)
    atomic_json(output / "EXPOSURE_LEDGER.json", ledger)
    record = {
        "kind": "fixed_public_TRAIN_roles",
        "roles": asdict(roles),
        "role_hash_resolution": binding(resolution_path),
        "metadata_source": binding(metadata_path),
        "metadata_columns_read": ["sequence_id", "seq_name", "sample_token"],
        "exposure_ledger": binding(output / "EXPOSURE_LEDGER.json"),
        "grouping": (
            "sequence_id fallback; no acquisition identifier established by available metadata"
        ),
        "claim_limit": "sequence-level generalization; independent routes are not established",
        "confirmation_status": "RESERVED_AND_CLOSED_PENDING_COMPLETE_S76_ANCESTRY_PROOF"
        if roles.confirmation
        else "FRESH_CONFIRMATION_GROUPS_UNPROVEN_OR_INSUFFICIENT",
        "D_availability": "PENDING_USABLE_ROWS_AND_RAW_RESOURCE_CHECKS",
        "optimizer_updates_at_assignment": 0,
        "fresh_labels_read": False,
    }
    record["artifact_sha256"] = object_digest(record)
    atomic_json(output / "DATA_ROLES.json", record)
    print(
        json.dumps(
            {
                "original_groups": len(roles.original),
                "historical_exposed_groups": len(roles.historical_exposed),
                "confirmation_reserved": roles.confirmation,
                "expansion_groups": len(roles.expansion),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
