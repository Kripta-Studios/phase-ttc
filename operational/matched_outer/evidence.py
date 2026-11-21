"""Build the per-query causal contract and explicit all-level lineage audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import load_frozen_protocol, validate_lineage, validate_query_rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_query_contract(
    data: pd.DataFrame, annotations: pd.DataFrame, *, history_us: int = 600_000
) -> pd.DataFrame:
    """Resolve the earliest causal availability of every box in each query window."""
    keys = ["sequence_id", "sample_token", "track_id", "timestamp_us"]
    required_data = {*keys, "frame_timestamps_us", "boxes_xyxy"}
    required_annotations = {"sample_token", "ttc"}
    if required_data - set(data) or required_annotations - set(annotations):
        raise ValueError("source tables lack query-contract columns")
    target = annotations[["sample_token", "ttc"]]
    if target["sample_token"].duplicated().any() or data["sample_token"].duplicated().any():
        raise ValueError("sample_token must be one-to-one in training tables")
    joined = data.merge(target, on="sample_token", how="left", validate="one_to_one")
    if joined["ttc"].isna().any():
        raise ValueError("a physical query has no signed TTC target")
    records: list[dict[str, Any]] = []
    for (_, _), track in joined.groupby(["sequence_id", "track_id"], sort=False):
        ordered = track.sort_values(["timestamp_us", "sample_token"], kind="mergesort")
        availability: dict[int, int] = {}
        for row in ordered.itertuples(index=False):
            query = int(row.timestamp_us)
            for frame in np.asarray(row.frame_timestamps_us, dtype=np.int64).reshape(-1):
                if int(frame) > query:
                    raise ValueError(f"future frame in source query {row.sample_token}")
                availability.setdefault(int(frame), query)
        pool_times = np.asarray(sorted(availability), dtype=np.int64)
        pool_available = np.asarray(
            [availability[int(frame)] for frame in pool_times], dtype=np.int64
        )
        for row in ordered.itertuples(index=False):
            query = int(row.timestamp_us)
            keep = (
                (pool_times >= query - history_us)
                & (pool_times <= query)
                & (pool_available <= query)
            )
            frames = pool_times[keep]
            available = pool_available[keep]
            if len(frames):
                capture_start = int(frames.min())
                capture_end = int(frames.max())
                box_available = int(available.max())
                span = capture_end - capture_start
                missing_reason = "" if span >= history_us else "history_shorter_than_600ms"
            else:
                capture_start = capture_end = box_available = query
                missing_reason = "no_causal_box_history"
            records.append(
                {
                    "physical_query_id": str(row.sample_token),
                    "official_alias": str(row.sample_token),
                    "sequence_id": str(row.sequence_id),
                    "track_id": str(row.track_id),
                    "region_id": str(row.track_id),
                    "query_timestamp_us": query,
                    "capture_start_us": capture_start,
                    "capture_end_us": capture_end,
                    "box_available_us": box_available,
                    "raw_observation_start_us": capture_start,
                    "raw_observation_end_us": capture_end,
                    "signed_ttc_target": float(row.ttc),
                    "history_missing_reason": missing_reason,
                    "model_fields": "causal_boxes|box_availability_clock",
                }
            )
    result = pd.DataFrame(records).sort_values(
        ["sequence_id", "query_timestamp_us", "track_id", "physical_query_id"],
        kind="mergesort",
    )
    validate_query_rows(result.to_dict(orient="records"))
    return result.reset_index(drop=True)


def expand_level_lineage(weights: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Name every possible learned boundary, including explicitly absent ones."""
    output: list[dict[str, Any]] = []
    for weight in weights:
        row = dict(weight)
        common = {
            "outer_fold": int(row["outer_fold"]),
            "seed": int(row["seed"]),
            "training_sequences": list(row["training_sequences"]),
            "outer_sequences": list(row.get("outer_sequences", [])),
        }
        output.extend(
            [
                {
                    **common,
                    "component": "public_box_rows",
                    "level": "source",
                    "learned": False,
                    "training_sequences": [],
                    "weight_sha256": "none",
                },
                {
                    **common,
                    "component": "causal_box_geometry",
                    "level": "producer",
                    "learned": False,
                    "training_sequences": [],
                    "weight_sha256": "none",
                },
                {
                    **common,
                    "component": "fixed_geometry_scaling",
                    "level": "normalizer",
                    "learned": False,
                    "training_sequences": [],
                    "weight_sha256": "none",
                },
                {
                    **common,
                    "component": "no_refiner",
                    "level": "refiner",
                    "learned": False,
                    "training_sequences": [],
                    "weight_sha256": "none",
                },
                row,
            ]
        )
    return output


def _read_weights(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            row["outer_fold"] = int(row["outer_fold"])
            row["seed"] = int(row["seed"])
            row["learned"] = row["learned"].lower() == "true"
            row["training_sequences"] = row["training_sequences"].split("|")
            row["outer_sequences"] = row["outer_sequences"].split("|")
            records.append(row)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_frozen_protocol(args.protocol, args.freeze_receipt)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query = build_query_contract(pd.read_parquet(args.data), pd.read_parquet(args.annotations))
    query_path = args.output_dir / "QUERY_CONTRACT.parquet"
    query.to_parquet(query_path, index=False)
    query.head(32).to_csv(args.output_dir / "QUERY_CONTRACT_SAMPLE.csv", index=False)
    weights = _read_weights(args.weights)
    validate_lineage(weights, protocol)
    levels = expand_level_lineage(weights)
    # Non-learned rows intentionally repeat per weight; learned identities remain unique.
    for row in levels:
        row["training_sequences"] = "|".join(row["training_sequences"])
        row["outer_sequences"] = "|".join(row["outer_sequences"])
    pd.DataFrame(levels).to_csv(args.output_dir / "LEVEL_LINEAGE.csv", index=False)
    summary = {
        "artifact_type": "phase_ttc_matched_outer_contract_evidence_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "query_contract_path": query_path.as_posix(),
        "query_contract_sha256": _sha256(query_path),
        "physical_queries": int(len(query)),
        "official_aliases": int(query["official_alias"].nunique()),
        "sequences": int(query["sequence_id"].nunique()),
        "future_box_rows": int((query["box_available_us"] > query["query_timestamp_us"]).sum()),
        "future_observation_rows": int(
            (query["raw_observation_end_us"] > query["query_timestamp_us"]).sum()
        ),
        "missing_history_counts": query["history_missing_reason"].value_counts().to_dict(),
        "learned_weights": len(weights),
        "levels_demonstrated": sorted({row["level"] for row in levels}),
        "lineage_status": "pass",
    }
    (args.output_dir / "CONTRACT_EVIDENCE.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["build_query_contract", "expand_level_lineage"]
