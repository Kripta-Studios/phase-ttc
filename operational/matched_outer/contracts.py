"""Frozen protocol, causal query and learned-component lineage guards."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """A run cannot satisfy the frozen matched-outer contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_protocol(path: Path, receipt_path: Path) -> dict[str, Any]:
    """Load a protocol only when its bytes still match the freeze receipt."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    actual = _sha256(path)
    if actual != receipt.get("protocol_sha256"):
        raise ContractError(
            f"protocol differs from frozen SHA-256: expected {receipt.get('protocol_sha256')}, "
            f"got {actual}"
        )
    protocol = json.loads(path.read_text(encoding="utf-8"))
    design = protocol.get("outer_design", {})
    folds = int(design.get("folds", 0))
    held_out = int(design.get("held_out_sequences_per_fold", 0))
    train = int(design.get("training_sequences_per_fold", 0))
    if folds < 2 or held_out < 1 or train != held_out * (folds - 1):
        raise ContractError("outer fold cardinalities are inconsistent")
    blocks = [list(map(str, design.get(f"fold_{fold}", []))) for fold in range(folds)]
    if any(len(block) != held_out for block in blocks):
        raise ContractError("an outer fold has the wrong sequence count")
    flattened = [sequence for block in blocks for sequence in block]
    if len(set(flattened)) != len(flattened):
        raise ContractError("outer folds overlap on sequence_id")
    if not design.get("seeds"):
        raise ContractError("outer protocol has no seeds")
    protocol["protocol_sha256"] = actual
    return protocol


def validate_query_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate identities, alias equivalence and all causal clocks."""
    required = {
        "physical_query_id",
        "official_alias",
        "sequence_id",
        "track_id",
        "region_id",
        "query_timestamp_us",
        "capture_start_us",
        "capture_end_us",
        "box_available_us",
        "raw_observation_start_us",
        "raw_observation_end_us",
        "signed_ttc_target",
        "history_missing_reason",
        "model_fields",
    }
    materialized = list(rows)
    physical: dict[str, tuple[str, str, float]] = {}
    aliases: set[str] = set()
    for index, row in enumerate(materialized):
        missing = sorted(required - set(row))
        if missing:
            raise ContractError(f"query row {index} misses fields: {missing}")
        query = int(row["query_timestamp_us"])
        capture_start = int(row["capture_start_us"])
        capture_end = int(row["capture_end_us"])
        available = int(row["box_available_us"])
        raw_start = int(row["raw_observation_start_us"])
        raw_end = int(row["raw_observation_end_us"])
        if available > query:
            raise ContractError(f"future box in query row {index}")
        if not capture_start <= capture_end <= available:
            raise ContractError(f"capture/availability clocks are not causal in row {index}")
        if not raw_start <= raw_end <= query:
            raise ContractError(f"raw observation clocks are not causal in row {index}")
        alias = str(row["official_alias"])
        if alias in aliases:
            raise ContractError(f"duplicate official alias {alias}")
        aliases.add(alias)
        key = str(row["physical_query_id"])
        identity = (str(row["sequence_id"]), str(row["track_id"]), float(row["signed_ttc_target"]))
        previous = physical.setdefault(key, identity)
        if previous[:2] != identity[:2]:
            raise ContractError(f"alias identity differs for physical query {key}")
        if previous[2] != identity[2]:
            raise ContractError(f"alias target differs for physical query {key}")
    return {
        "status": "pass",
        "rows": len(materialized),
        "physical_queries": len(physical),
        "aliases": len(materialized) - len(physical),
    }


def _outer_sequences(protocol: Mapping[str, Any], fold: int) -> set[str]:
    design = protocol["outer_design"]
    if fold < 0 or fold >= int(design["folds"]):
        raise ContractError(f"unknown outer fold {fold}")
    return set(map(str, design[f"fold_{fold}"]))


def _all_sequences(protocol: Mapping[str, Any]) -> set[str]:
    design = protocol["outer_design"]
    return {
        str(sequence) for fold in range(int(design["folds"])) for sequence in design[f"fold_{fold}"]
    }


def validate_lineage(
    records: Iterable[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject outer exposure in every learned producer/normalizer/refiner/head."""
    materialized = list(records)
    seen: set[tuple[str, int, int]] = set()
    all_sequences = _all_sequences(protocol)
    expected_train = int(protocol["outer_design"]["training_sequences_per_fold"])
    for record in materialized:
        component = str(record.get("component", ""))
        level = str(record.get("level", ""))
        fold = int(record.get("outer_fold", -1))
        seed = int(record.get("seed", -1))
        identity = (component, fold, seed)
        if not component or not level or identity in seen:
            raise ContractError(f"invalid or duplicate lineage identity {identity}")
        seen.add(identity)
        training = set(map(str, record.get("training_sequences", [])))
        unknown = training - all_sequences
        if unknown:
            raise ContractError(f"unknown training sequences for {component}: {sorted(unknown)}")
        if bool(record.get("learned")):
            overlap = training & _outer_sequences(protocol, fold)
            if overlap:
                raise ContractError(f"outer exposure at {level} {component}: {sorted(overlap)}")
            if len(training) != expected_train:
                raise ContractError(
                    f"learned {level} {component} saw {len(training)} sequences, "
                    f"expected {expected_train}"
                )
            digest = str(record.get("weight_sha256", ""))
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ContractError(f"learned {level} {component} has no valid weight SHA-256")
    return {"status": "pass", "components": len(materialized)}


def cache_key(
    protocol_sha256: str,
    outer_fold: int,
    inner_fold_or_refit: str,
    seed: int,
    producer_weight_sha256: str,
    modality: str,
    history_budget_ms: int,
) -> str:
    """Content key required for any reusable temporal cache."""
    payload = {
        "history_budget_ms": int(history_budget_ms),
        "inner_fold_or_refit": str(inner_fold_or_refit),
        "modality": str(modality),
        "outer_fold": int(outer_fold),
        "producer_weight_sha256": str(producer_weight_sha256),
        "protocol_sha256": str(protocol_sha256),
        "seed": int(seed),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ContractError",
    "cache_key",
    "load_frozen_protocol",
    "validate_lineage",
    "validate_query_rows",
]
