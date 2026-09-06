"""Physical integration of a fixed capacity cohort through existing TRAIN adapters."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner, disk_reservation
from e_jepa_ttc.artifacts.risk_geometry_v10 import (
    AllowlistedSources,
    atomic_json,
    binding,
    object_digest,
    verify,
)
from e_jepa_ttc.data.event_context128_cache import build_count_cache
from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from e_jepa_ttc.data.raw_event_binding import _as_boxes
from e_jepa_ttc.data.stage70_clock import context_clock
from e_jepa_ttc.data.stage70_expansion_binding import ExpansionTrainAccess, expansion_bindings


def _signed(source: dict[str, Any]) -> dict[str, Any]:
    record = json.loads(verify(source).read_text(encoding="utf-8"))
    unsigned = {k: v for k, v in record.items() if k != "artifact_sha256"}
    if object_digest(unsigned) != record.get("artifact_sha256"):
        raise ValueError("capacity source signature changed")
    return record


def validate_capacity_rows(
    preparation_source: dict[str, Any], output: Path, check: Callable[[], None]
) -> dict[str, Any]:
    """Verify the fixed token universe; invalid rows block it rather than trigger replacement."""
    preparation = _signed(preparation_source)
    cohort = _signed(preparation["cohort"])
    partitions = _signed(preparation["source_partitions"])
    roles = _signed(preparation["fixed_original_roles"])["roles"]
    mapping = dict(roles["sequence_to_group"])
    forbidden = set(roles["confirmation"]) | set(roles["protected"])
    if (
        cohort.get("kind") != "CAPACITY_DEVELOPMENT_COHORT"
        or not set(cohort["selected_sequences"]) <= mapping.keys()
        or {mapping[s] for s in cohort["selected_sequences"]} & forbidden
    ):
        raise ValueError("cohort contains an unresolved or protected source role")
    parent = Path(preparation["source_partitions"]["path"]).resolve().parent
    sources = partitions["outputs"]
    for key in ("train_metadata", "train_labels"):
        if Path(sources[key]["path"]).resolve() != parent / f"{key}.parquet":
            raise ValueError("capacity source escaped the administrative TRAIN partition")
    identity = {
        "preparation": preparation_source,
        "cohort": preparation["cohort"],
        "source_metadata": sources["train_metadata"],
        "source_labels": sources["train_labels"],
        "producer_code": binding(Path(__file__)),
    }
    result_path = output / "ROW_USABILITY.json"
    if result_path.exists():
        previous = _signed(binding(result_path))
        if previous["identity"] != identity:
            raise ValueError("row validation recipe changed; retain the earlier attempt")
        for key in ("metadata", "labels", "source_reads"):
            if previous.get(key):
                verify(previous[key])
        return previous
    output.mkdir(parents=True, exist_ok=True)
    access = AllowlistedSources(
        [sources["train_metadata"], sources["train_labels"]], output / "SOURCE_READS.jsonl"
    )
    check()
    metadata = pd.read_parquet(
        access.access(Path(sources["train_metadata"]["path"])),
        columns=[
            "sample_token",
            "sequence_id",
            "track_id",
            "timestamp_us",
            "frame_timestamps_us",
            "event_windows_us",
            "boxes_xyxy",
            "events_path",
        ],
    ).set_index("sample_token")
    labels = pd.read_parquet(
        access.access(Path(sources["train_labels"]["path"])),
        columns=["sample_token", "sequence_id", "timestamp_us", "ttc", "frame_ttc"],
    ).set_index("sample_token")
    tokens = cohort["ordered_tokens"]
    decoded_label_rows = len(labels)
    if metadata.index.duplicated().any() or labels.index.duplicated().any():
        raise ValueError("duplicate administrative TRAIN token")
    metadata, labels = metadata.loc[tokens].copy(), labels.loc[tokens].copy()
    if set(metadata.sequence_id) != set(cohort["selected_sequences"]):
        raise ValueError("selected metadata acquisition identities changed")
    for column in ("sequence_id", "timestamp_us"):
        if not np.array_equal(metadata[column].to_numpy(), labels[column].to_numpy()):
            raise ValueError("annotation identity differs from fixed metadata")
    rejected = {}
    max_drift = max_gap = 0
    for index, (token, row) in enumerate(metadata.iterrows()):
        if index % 64 == 0:
            check()
        target = float(labels.loc[token, "ttc"])
        frame_targets = np.asarray(labels.loc[token, "frame_ttc"], dtype=np.float64)
        if not np.isfinite(target) or not (target < 0 or target > 0.1):
            rejected[str(token)] = "invalid_signed_TTC_target"
            continue
        if frame_targets.shape != (2,) or float(frame_targets[-1]) != target:
            rejected[str(token)] = "target_not_attached_to_later_annotation"
            continue
        try:
            clock = context_clock(row.event_windows_us, row.frame_timestamps_us, row.timestamp_us)
            common_square_from_boxes(
                _as_boxes(row.boxes_xyxy), (0, 1), margin_fraction=0.25, minimum_edge=8.0
            )
        except ValueError as error:
            rejected[str(token)] = f"clock_or_ROI_contract:{error}"
            continue
        max_drift = max(
            max_drift, abs(clock.frame_to_event_offsets_us[1] - clock.frame_to_event_offsets_us[0])
        )
        max_gap = max(max_gap, abs(clock.gap_us))
    result: dict[str, Any] = {
        "kind": "CAPACITY_COHORT_ROW_USABILITY",
        "identity": identity,
        "status": "DATA_PREREQUISITE_BLOCKED" if rejected else "ALL_SELECTED_ROWS_VALID",
        "selected_rows": len(tokens),
        "rejected": rejected,
        "replacement_rows": 0,
        "maximum_offset_drift_us": max_drift,
        "maximum_gap_us": max_gap,
        "confirmation_targets_read": False,
        "TTC_targets_read_for_contract_validation": True,
        "administrative_TRAIN_label_rows_decoded": decoded_label_rows,
        "selected_target_rows_used": len(tokens),
        "optimizer_updates": 0,
    }
    if not rejected:
        metadata["outer_fold"] = metadata.sequence_id.map(cohort["sequence_to_outer_fold"]).astype(
            int
        )
        metadata.reset_index().to_parquet(output / "METADATA.parquet", index=False)
        labels.reset_index().to_parquet(output / "LABELS.parquet", index=False)
        result["metadata"] = binding(output / "METADATA.parquet")
        result["labels"] = binding(output / "LABELS.parquet")
    result["source_reads"] = binding(output / "SOURCE_READS.jsonl")
    result["artifact_sha256"] = object_digest(result)
    atomic_json(result_path, result)
    return result


def materialize_capacity_counts(
    usability_source: dict[str, Any], output: Path, *, build_cache: bool
) -> dict[str, Any]:
    """Reuse bounded raw binding/count-cache builders; no encoder or normalization fit."""
    usability = _signed(usability_source)
    if usability["status"] != "ALL_SELECTED_ROWS_VALID":
        raise ValueError("selected cohort has an unresolved clock/target/ROI prerequisite")
    preparation = _signed(usability["identity"]["preparation"])
    cohort = _signed(preparation["cohort"])
    predecessor = _signed(preparation["predecessor_lock"])
    local = json.loads(verify(predecessor["local_paths"]).read_text(encoding="utf-8"))
    metadata = verify(usability["metadata"])
    allowed = frozenset(cohort["selected_sequences"])
    output.mkdir(parents=True, exist_ok=True)
    # Include the predecessor's largest remaining per-fit allocation in admission.
    predecessor_reserve = max(
        value["disk_bytes"] for value in predecessor["execution_reservations"].values()
    )
    remaining_cache_bytes = usability["selected_rows"] * (1048576 + 4096) + 4 * 1024**2
    reservation = disk_reservation(output, remaining_cache_bytes + predecessor_reserve)
    if not reservation.passes:
        raise RuntimeError("RESOURCE_BLOCKED: combined cache/training reservations leave under40GB")
    bound_root = output / "raw_bindings"
    with ResourceOwner(
        bound_root, remaining_bytes=512 * 1024**2, planned_ram_bytes=2 * 1024**3
    ) as owner:
        progress = bound_root / "RAW_HASH_PROGRESS.json"
        if not progress.exists():
            hashes = {}
            manifests = []
            for source in predecessor["count_caches"].values():
                cache = _signed(source)
                manifests.append(source)
                for sequence, record in cache["identity"]["sources"].items():
                    if sequence in allowed:
                        if sequence in hashes and hashes[sequence] != record:
                            raise ValueError("predecessor count sources disagree on raw identity")
                        hashes[sequence] = record
            atomic_json(progress, hashes)
            atomic_json(
                bound_root / "HASH_REUSE_SOURCES.json",
                {
                    "verified_cache_manifests": manifests,
                    "initial_hash_records": hashes,
                    "policy": (
                        "builder rechecks exact path/size/mtime; "
                        "changed files require a full hash read"
                    ),
                },
            )
        access = ExpansionTrainAccess(
            Path(local["raw_train_root"]),
            metadata,
            allowed_sequences=allowed,
            journal_path=bound_root / "SOURCE_READS.jsonl",
            resource_check=owner.check,
        )
        raw = expansion_bindings(metadata, access, bound_root)
        by_hash: dict[str, set[str]] = {}
        for sequence, source in raw["identity"]["sources"].items():
            by_hash.setdefault(source["sha256"], set()).add(cohort["sequence_to_group"][sequence])
        if any(len(groups) > 1 for groups in by_hash.values()):
            raise ValueError("duplicate raw bytes assigned to supposedly different acquisitions")
    cache_source = None
    if build_cache:
        cache_root = output / "counts"
        access = ExpansionTrainAccess(
            Path(local["raw_train_root"]),
            metadata,
            allowed_sequences=allowed,
            journal_path=cache_root / "SOURCE_READS.jsonl",
            resource_check=lambda: None,
        )
        build_count_cache(verify(raw["binding"]), raw["identity"]["sources"], access, cache_root)
        cache_source = binding(cache_root / "CACHE_MANIFEST.json")
    result = {
        "kind": "CAPACITY_PHYSICAL_REPRESENTATION",
        "status": "COUNTS_COMPLETE" if build_cache else "RAW_BOUND_CACHE_PENDING",
        "usability": usability_source,
        "cohort": preparation["cohort"],
        "raw_binding_manifest": binding(bound_root / "RAW_BINDING_MANIFEST.json"),
        "count_cache": cache_source,
        "combined_reservation": reservation.record(),
        "normalization_fitted": False,
        "optimizer_updates": 0,
    }
    result["artifact_sha256"] = object_digest(result)
    destination = output / ("COUNTS_COMPLETE.json" if build_cache else "RAW_BOUND.json")
    if destination.exists():
        previous = _signed(binding(destination))
        for key in ("usability", "cohort", "raw_binding_manifest", "count_cache", "status"):
            if previous[key] != result[key]:
                raise ValueError("completed physical integration identity changed")
        return previous
    atomic_json(destination, result)
    return result
