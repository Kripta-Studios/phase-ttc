"""Prepare the capacity cohort from the predecessor's verified TRAIN metadata only."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import (
    AllowlistedSources,
    atomic_json,
    binding,
    object_digest,
    verify,
)
from e_jepa_ttc.data.capacity_cohort import CohortRecipe, build_cohort
from e_jepa_ttc.data.stage70_roles import SamplingRow, resolve_train_event


def signed(path: Path) -> dict:
    """Verify a local immutable manifest without broadening its data roles."""
    record = json.loads(path.read_text(encoding="utf-8"))
    signature = record.pop("artifact_sha256")
    if object_digest(record) != signature:
        raise ValueError(f"manifest signature changed: {path.name}")
    return {**record, "artifact_sha256": signature}


def main() -> None:
    """Select a shared temporal sample and preserve the already reserved confirmation IDs."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--predecessor-root", type=Path, required=True)
    cli.add_argument("--local-paths", type=Path, required=True)
    cli.add_argument("--output-root", type=Path, required=True)
    args = cli.parse_args()
    predecessor = args.predecessor_root.resolve(strict=True)
    output = args.output_root.resolve()
    if output != ROOT / "artifacts/stage70_76_capacity/cohort":
        raise ValueError("capacity preparation must use its separate owned output directory")
    if (output / "COHORT.json").exists():
        raise ValueError("prospective cohort already exists; do not reselect its identities")
    lock = signed(predecessor / "TRAINING_LOCK.json")
    if lock.get("kind") != "STAGE70_76_TRAINING_LOCK" or lock.get("campaign_root") != str(
        predecessor
    ):
        raise ValueError("predecessor is not the original frozen architecture campaign")
    if binding(args.local_paths) != lock["local_paths"]:
        raise ValueError("operational path map differs from the predecessor's frozen binding")
    local = json.loads(verify(lock["local_paths"]).read_text(encoding="utf-8"))
    partitions_path = predecessor / "role_partitions/ROLE_PARTITIONS.json"
    partitions = signed(partitions_path)
    role_path = verify(lock["data_roles"])
    roles = signed(role_path)["roles"]
    mapping = dict(roles["sequence_to_group"])
    confirmation = sorted(s for s, g in mapping.items() if g in roles["confirmation"])
    protected = sorted(s for s, g in mapping.items() if g in roles["protected"])
    permitted = frozenset(mapping.keys() - set(confirmation) - set(protected))
    metadata_binding = partitions["outputs"]["train_metadata"]
    if (
        Path(metadata_binding["path"]).resolve()
        != predecessor / "role_partitions/train_metadata.parquet"
    ):
        raise ValueError("source binding escaped the administrative TRAIN metadata partition")
    with ResourceOwner(output, remaining_bytes=32 * 1024**2, planned_ram_bytes=1024**3):
        sources = AllowlistedSources([metadata_binding], output / "SOURCE_READS.jsonl")
        metadata = pd.read_parquet(
            sources.access(Path(metadata_binding["path"])),
            columns=["sample_token", "sequence_id", "track_id", "timestamp_us", "events_path"],
        )
        if not set(metadata.sequence_id) <= permitted:
            raise ValueError("TRAIN metadata partition unexpectedly contains a reserved role")
        available = set()
        missing = {}
        physical_sources = {}
        for sequence, rows in metadata.groupby("sequence_id", sort=True):
            if rows.events_path.nunique() != 1:
                raise ValueError("sequence maps to more than one physical raw source")
            try:
                path = resolve_train_event(
                    Path(local["raw_train_root"]),
                    str(sequence),
                    str(rows.events_path.iloc[0]),
                    permitted,
                )
            except FileNotFoundError:
                missing[str(sequence)] = "RAW_TRAIN_FILE_MISSING"
                continue
            available.add(str(sequence))
            stat = path.stat()
            physical_sources[str(sequence)] = {
                "path": str(path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "read_scope": "path_resolution_and_stat_only",
            }
        rows = [
            SamplingRow(
                str(row.sample_token),
                str(row.sequence_id),
                str(row.track_id),
                int(row.timestamp_us),
            )
            for row in metadata.loc[metadata.sequence_id.isin(available)].itertuples(index=False)
        ]
        cohort = build_cohort(
            rows,
            sequence_to_group=mapping,
            confirmation_sequences=confirmation,
            protected_sequences=protected,
            recipe=CohortRecipe(),
        )
        selected = metadata.set_index("sample_token").loc[cohort["ordered_tokens"]].reset_index()
        selected.to_csv(output / "SELECTED_METADATA.csv", index=False)
        atomic_json(output / "COHORT.json", cohort)
        report = {
            "status": "METADATA_SELECTED_CLOCK_TARGET_AND_CACHE_USABILITY_PENDING",
            "cohort": binding(output / "COHORT.json"),
            "selected_metadata": binding(output / "SELECTED_METADATA.csv"),
            "predecessor_lock": binding(predecessor / "TRAINING_LOCK.json"),
            "fixed_original_roles": binding(role_path),
            "source_partitions": binding(partitions_path),
            "source_metadata": metadata_binding,
            "physical_sources": physical_sources,
            "missing_sequences": missing,
            "selected_sequences": cohort["actual_sequence_count"],
            "selected_rows": len(selected),
            "external_sequences": len(cohort["external_sequences"]),
            "D_group_minimum": 6,
            "D_status": (
                "GROUP_COUNT_BLOCKED"
                if len({mapping[s] for s in cohort["external_sequences"]}) < 6
                else "ROW_AND_REPRESENTATION_PREREQUISITES_PENDING"
            ),
            "acquisition_independence": (
                "inherited audit; sequence fallback is not verified independence"
            ),
            "confirmation_metadata_or_targets_read": False,
            "TTC_targets_read": False,
            "optimizer_updates": 0,
            "source_reads": binding(output / "SOURCE_READS.jsonl"),
        }
        atomic_json(
            output / "PREPARATION.json", {**report, "artifact_sha256": object_digest(report)}
        )
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "status",
                        "selected_sequences",
                        "selected_rows",
                        "external_sequences",
                        "D_status",
                        "optimizer_updates",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
