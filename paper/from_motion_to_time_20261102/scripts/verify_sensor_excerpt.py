"""Verify real sensor illustrations against the website's extraction provenance."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCERPT = ROOT / "data/real_sensor_excerpt"
QUERY_INDICES = (0, 7, 15)


def digest(path: Path) -> str:
    """Return the SHA-256 of the distributed bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Check asset identity, query timestamps, and saved-result provenance."""
    provenance = json.loads((EXCERPT / "source_provenance.json").read_text(encoding="utf-8"))
    replay = json.loads((EXCERPT / "replay-data.json").read_text(encoding="utf-8"))
    assets = {"figures/evttc_ccrm_poster.jpg": "poster.jpg"}
    for index in QUERY_INDICES:
        for kind in ("rgb.jpg", "events.png"):
            name = f"query-{index:02d}-{kind}"
            assets[f"figures/evttc_{name}"] = name
    assets["figures/evttc_query-07-channels.png"] = "query-07-channels.png"
    assets["data/real_sensor_excerpt/ATTRIBUTION.txt"] = "ATTRIBUTION.txt"
    assets["data/real_sensor_excerpt/replay-data.json"] = "replay-data.json"
    checked = {}
    for relative, original in assets.items():
        path = ROOT / relative
        expected = provenance["assets"][original]
        actual = digest(path)
        if actual != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Website sensor asset identity mismatch: {relative}")
        checked[relative] = actual
    if digest(EXCERPT / "extract-real-replay.py") != provenance["extraction_code_sha256"]:
        raise ValueError("Website extraction script identity mismatch")
    result_path = ROOT / "data/transfer_queries.csv"
    if digest(result_path) != provenance["query_results_sha256"]:
        raise ValueError("Sensor excerpt does not match the distributed transfer results")
    with result_path.open(encoding="utf-8", newline="") as handle:
        results = {row["query_id"]: row for row in csv.DictReader(handle)}
    if replay["sequence_id"] != provenance["sequence_id"]:
        raise ValueError("Excerpt sequence identities differ")
    selected = []
    for index in QUERY_INDICES:
        query = next(row for row in replay["queries"] if row["index"] == index)
        anchor = query["timestamp_us"]
        endpoints = [anchor - 200_000, anchor - 100_000, anchor]
        expected_windows = [[end - 100_000, end] for end in endpoints]
        if query["event_windows_us"] != expected_windows:
            raise ValueError(f"Unexpected event windows: query {index}")
        residuals = [t - end for t, end in zip(query["rgb_timestamps_us"], endpoints, strict=True)]
        if len(residuals) != 3 or max(map(abs, residuals)) > 7:
            raise ValueError(f"RGB/event timing differs from the figure caption: query {index}")
        if float(results[query["id"]]["target_ttc"]) != query["gt_ttc"]:
            raise ValueError(f"Query target does not match archived results: query {index}")
        selected.append(
            {
                "index": index,
                "query_id": query["id"],
                "anchor_us": anchor,
                "rgb_minus_event_endpoint_us": residuals,
            }
        )
    if replay["frames"][0]["timestamp_us"] != 14_350_374:
        raise ValueError("Full-scene poster timestamp differs from the figure caption")
    if len(replay["channel_names"]) != 12:
        raise ValueError("Expected twelve event input planes")
    report = {
        "scope": "Byte-identical real sensor illustrations; no inference or new evaluation",
        "sequence_id": replay["sequence_id"],
        "verified_assets": checked,
        "extraction_code_sha256": provenance["extraction_code_sha256"],
        "query_results_sha256": provenance["query_results_sha256"],
        "queries": selected,
    }
    output = ROOT / "updates/review/real_sensor_excerpt_validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified_assets": len(checked), "verified_queries": len(selected)}))


if __name__ == "__main__":
    main()
