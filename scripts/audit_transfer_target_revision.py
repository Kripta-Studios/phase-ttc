"""Reassociate sealed EvTTC transfer queries without changing predictions or clocks."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPAIRED = {
    "CCRs-1-low-100-overlap-100": (
        "CCRs1-low",
        "1c5adba8667acb8096eb80c26e9a9229ca3a7d76d0a6d998085c2d9b4e820fed",
    ),
    "CCRs-1-medium-100-overlap-100": (
        "CCRs1-medium",
        "780ee30ff9775da8452102bbbb9f5120ae903055007348096ecbc24b594b8c76",
    ),
}


def digest(path: Path) -> str:
    """Hash original bytes, including their preserved line endings."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_points(path: Path) -> list[tuple[int, float]]:
    """Read five-column public TTC without fitting an offset or interpolating."""
    points = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.replace(",", " ").split()
        if len(fields) != 5:
            raise ValueError(f"Expected five-column public TTC: {path.name}")
        timestamp = Decimal(fields[1]) * 1_000_000
        if timestamp != timestamp.to_integral_value():
            raise ValueError("Public timestamp is not an integer microsecond")
        points.append((int(timestamp), float(fields[4])))
    return points


def nearest_target(points: list[tuple[int, float]], anchor_us: int) -> dict[str, Any]:
    """Match the historical 50 ms rule with earlier ties and explicit exclusions."""
    times = [point[0] for point in points]
    if not times or any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("Target times must be nonempty and strictly increasing")
    insertion = bisect.bisect_left(times, anchor_us)
    before, after = max(0, insertion - 1), min(len(times) - 1, insertion)
    selected = min({before, after}, key=lambda i: (abs(times[i] - anchor_us), i))
    timestamp, target = points[selected]
    distance = abs(timestamp - anchor_us)
    reason = ""
    if distance > 50_000:
        reason = "TTC_TIMESTAMP_MISMATCH"
    elif not math.isfinite(target) or target == 0:
        reason = "TTC_INVALID_OR_ZERO"
    return {
        "target_ttc": target if not reason else "",
        "target_exclusion": reason,
        "target_timestamp_us": timestamp,
        "target_delta_us": distance,
        "neighbor_before_us": points[before][0],
        "neighbor_before_ttc": points[before][1],
        "neighbor_after_us": points[after][0],
        "neighbor_after_ttc": points[after][1],
    }


def revise_candidates(
    rows: list[dict[str, Any]],
    curves: dict[str, list[tuple[int, float]]],
    hashes: dict[str, str],
) -> list[dict[str, Any]]:
    """Recompute all eligibility decisions while retaining every prediction field."""
    seen: set[str] = set()
    result = []
    for row in rows:
        identity = row["query_id"]
        if identity in seen:
            raise ValueError(f"Duplicate query: {identity}")
        seen.add(identity)
        group = row["group_id"]
        anchor_us = int(identity.rsplit(":", 1)[1])
        result.append(
            {
                **row,
                "legacy_target_ttc": row["target_ttc"],
                "legacy_target_exclusion": row["target_exclusion"],
                "anchor_us": anchor_us,
                **nearest_target(curves[group], anchor_us),
                "target_source_sha256": hashes[group],
            }
        )
    return result


def main() -> None:
    """Validate original assignment, then write an additive corrected projection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=ROOT / "paper/from_motion_to_time_20261102")
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=ROOT / "artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json",
    )
    parser.add_argument(
        "--original-results",
        type=Path,
        default=ROOT / "docs/final_train40_transfer_20261027/RESULTS.json",
    )
    parser.add_argument(
        "--identity",
        type=Path,
        default=ROOT / "docs/superpowers/plans/evidence/2026-10-30-evttc-audit.json",
    )
    parser.add_argument(
        "--public-sources",
        type=Path,
        default=ROOT / "output/phase_safe_fusion/evaluator_audit/sources",
    )
    args = parser.parse_args()
    original = json.loads(args.original_results.read_text(encoding="utf-8"))
    manifest = json.loads(args.query_manifest.read_text(encoding="utf-8"))
    if digest(args.query_manifest) != original["provenance"]["manifest_sha256"]:
        raise ValueError("Sealed query manifest identity changed")
    metadata = {row["query_id"]: row for row in manifest["rows"]}
    input_path = args.paper / "data/transfer_queries.csv"
    with input_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if set(metadata) != {r["query_id"] for r in rows} or len(rows) != 554:
        raise ValueError("Expected all 554 sealed transfer candidates")
    paths = {row["physical_sequence_id"]: Path(row["ttc_csv_path"]) for row in manifest["rows"]}
    old_curves, curves, old_hashes, hashes = {}, {}, {}, {}
    for group, path in paths.items():
        expected = original["label_sha256"][str(path)]
        if digest(path) != expected:
            raise ValueError(f"Original GT identity changed: {group}")
        old_hashes[group] = expected
        old_curves[group] = read_points(path)
        source = path
        if group in REPAIRED:
            name, expected = REPAIRED[group]
            source = args.public_sources / (name + ".csv")
        if digest(source) != expected:
            raise ValueError(f"Corrected public source identity mismatch: {group}")
        hashes[group] = expected
        curves[group] = read_points(source)
    # An unchanged matching policy must first reproduce every historical target.
    for row in revise_candidates(rows, old_curves, old_hashes):
        old = row["legacy_target_ttc"]
        if row["target_exclusion"] != row["legacy_target_exclusion"] or (
            old != "" and float(old) != row["target_ttc"]
        ):
            raise ValueError(f"Historical matching was not reproduced: {row['query_id']}")
        if metadata[row["query_id"]]["anchor_us"] != row["anchor_us"]:
            raise ValueError("Query identity and manifest clocks differ")
    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    matches = identity["evidence"]["recording_overlap"]["sequences"]
    clock_evidence = {}
    for group, (name, _) in REPAIRED.items():
        entry = next(item for item in matches if item["competition_sequence"] == name)
        candidates = entry["verified_sample_matches"]
        if len(candidates) != 1 or candidates[0]["sequence_id"] != group:
            raise ValueError("Native recording identity is ambiguous")
        match = candidates[0]
        frames = match["all_rgb_frames"]
        if not match["all_query_rgb_timestamps_match"] or not all(
            f["equal"] and f["competition_sha256"] == f["original_sha256"] for f in frames
        ):
            raise ValueError("Native RGB identity or clock verification failed")
        clock_evidence[group] = {
            "competition_sequence": name,
            "verified_frames": len(frames),
            "competition_to_native_offset_us": match["timestamp_offset_us"],
            "query_clock": "native microseconds; no additional offset or fitted clock",
        }
    revised = revise_candidates(rows, curves, hashes)
    output = args.paper / "audit_20261104"
    output.mkdir(exist_ok=True)
    with (output / "corrected_transfer_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(revised[0]))
        writer.writeheader()
        writer.writerows(revised)
    counts = {}
    for cohort in ("historical46", "extended_new"):
        part = [r for r in revised if r["cohort"] == cohort]
        counts[cohort] = {
            "candidates": len(part),
            "historical_eligible": sum(
                not r["target_exclusion"] for r in rows if r["cohort"] == cohort
            ),
            "corrected_eligible": sum(not r["target_exclusion"] for r in part),
        }
    receipt = {
        "schema": "phase_ttc_transfer_target_revision_v1",
        "counts": counts,
        "prediction_policy": (
            "All saved predictions and candidate anchors unchanged; no inference or fitting"
        ),
        "target_policy": (
            "Nearest public TTC within inclusive 50000us; earlier timestamp breaks ties"
        ),
        "verified_historical_targets": len(rows),
        "historical_projection_sha256": digest(input_path),
        "query_manifest_sha256": digest(args.query_manifest),
        "original_results_sha256": digest(args.original_results),
        "identity_evidence_sha256": digest(args.identity),
        "source_script_sha256": digest(Path(__file__)),
        "original_target_source_sha256": old_hashes,
        "corrected_target_source_sha256": hashes,
        "clock_identity": clock_evidence,
        "corrected_projection_sha256": digest(output / "corrected_transfer_candidates.csv"),
        "source_scope": (
            "Bounded derived target/neighbor projection; original raw sensor data not distributed"
        ),
    }
    (output / "transfer_reassociation.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
