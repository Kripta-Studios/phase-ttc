"""Materialize exact completed, failed and pending test IDs from a closed QA attempt."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify


def summarize(audit: Path) -> dict[str, object]:
    """A killed test is pending even if its setup passed; retain all observed failures."""
    receipt_path = audit / "CLOSED_ACCOUNTING.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    signature = receipt.pop("sha256")
    if object_digest(receipt) != signature or not receipt["all_processes_closed"]:
        raise ValueError("test disposition requires verified closed-process accounting")
    verify(receipt["database"])
    verify(receipt["stdout"])
    capture = audit / "capture"
    collected_path = capture / "COLLECTED_NODEIDS.json"
    collected = json.loads(collected_path.read_text(encoding="utf-8"))
    reports_path = capture / "REPORTS.jsonl"
    reports = [json.loads(line) for line in reports_path.read_text(encoding="utf-8").splitlines()]
    completed = {row["nodeid"] for row in reports if row["phase"] == "teardown"}
    failures = {row["nodeid"] for row in reports if row["outcome"] == "failed"}
    skipped = {row["nodeid"] for row in reports if row["outcome"] == "skipped"}
    passed = {
        row["nodeid"] for row in reports if row["phase"] == "call" and row["outcome"] == "passed"
    }
    pending = set(collected) - completed
    result_path = capture / "RESULT.json"
    finished = result_path.exists()
    return {
        "kind": "PYTEST_EXACT_NODE_DISPOSITION",
        "complete_suite": finished and not pending and receipt["interruption"] is None,
        "collected_count": len(collected),
        "completed_count": len(completed),
        "passed_ids": sorted((passed & completed) - failures),
        "failure_ids": sorted(failures),
        "skipped_ids": sorted(skipped & completed),
        "completed_ids": sorted(completed),
        "pending_ids": sorted(pending),
        "pending_count": len(pending),
        "interruption": receipt["interruption"],
        "pytest_exit_code": receipt["pytest_exit_code"],
        "charged_optimizer_updates": receipt["charged"],
        "confirmed_optimizer_updates": receipt["confirmed"],
        "uncertain_optimizer_attempts": receipt["uncertain"],
        "source_bindings": [
            binding(path)
            for path in (
                receipt_path,
                collected_path,
                reports_path,
                audit / "RECIPE.json",
            )
        ],
        "result_binding": binding(result_path) if finished else None,
    }


def main() -> None:
    """Write one immutable disposition next to an owned QA attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    audit = args.audit.resolve(strict=True)
    if not audit.is_relative_to(ROOT / "artifacts/stage70_76_architecture/baseline_qa"):
        raise ValueError("audit must be within this campaign's QA output")
    output = audit / "TEST_DISPOSITION.json"
    if output.exists():
        raise ValueError("immutable test disposition already exists")
    result = summarize(audit)
    atomic_json(output, {**result, "sha256": object_digest(result)})
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "complete_suite",
                    "collected_count",
                    "completed_count",
                    "pending_count",
                    "confirmed_optimizer_updates",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
