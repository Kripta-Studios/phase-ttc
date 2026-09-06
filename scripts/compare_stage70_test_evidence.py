"""Compare complete baseline/current node coverage, preserving explicit environment rechecks."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify


def signed(path: Path) -> dict[str, Any]:
    """Verify a closed evidence record and its direct physical predecessors."""
    record = json.loads(path.read_text(encoding="utf-8"))
    signature = record.pop("sha256")
    if object_digest(record) != signature:
        raise ValueError("test evidence signature differs")
    for source in record.get("source_bindings", []) + record.get("parts", []):
        verify(source)
    return record


def outcomes(record: dict[str, Any]) -> dict[str, str]:
    """Require a unique terminal disposition for every recorded test."""
    result: dict[str, str] = {}
    for field, state in (
        ("passed_ids", "passed"),
        ("failure_ids", "failed"),
        ("skipped_ids", "skipped"),
    ):
        for node in record[field]:
            if node in result:
                raise ValueError("ambiguous test disposition")
            result[node] = state
    expected = record.get("total_tests", record.get("collected_count"))
    if len(result) != expected:
        raise ValueError("test coverage is incomplete")
    return result


def main() -> None:
    """Materialize the fixed, source-bound full-suite comparison with no waived failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if campaign != ROOT / "artifacts/stage70_76_architecture":
        raise ValueError("comparison must remain in this campaign")
    qa = campaign / "baseline_qa"
    output = qa / "BASELINE_VERSUS_CURRENT_4A8B545.json"
    if output.exists():
        raise ValueError("immutable comparison already exists")
    baseline_path = qa / "BASELINE_COMPLETE_COVERAGE.json"
    current_path = qa / "current_4a8b545_full/TEST_DISPOSITION.json"
    baseline, current = signed(baseline_path), signed(current_path)
    if not baseline["complete_node_coverage"] or not current["complete_suite"]:
        raise ValueError("full baseline/current coverage required")
    tables = {"baseline": outcomes(baseline), "current": outcomes(current)}
    changes = []
    references = [binding(baseline_path), binding(current_path)]
    for side, audit_name, reason in (
        ("baseline", "baseline_source57b_uv_environment", "bound uv to installed offline runtime"),
        ("baseline", "baseline_source57b_python_imports", "resolved Windows base-Python imports"),
        ("current", "current_0777326_python_imports", "resolved Windows base-Python imports"),
    ):
        path = qa / audit_name / "TEST_DISPOSITION.json"
        recheck = signed(path)
        if not recheck["complete_suite"] or recheck["confirmed_optimizer_updates"] != 0:
            raise ValueError("environment recheck must complete without optimizer work")
        replacements = outcomes(recheck)
        if not set(replacements) <= tables[side].keys():
            raise ValueError("environment recheck changed the source test universe")
        for node, state in replacements.items():
            changes.append(
                {
                    "side": side,
                    "nodeid": node,
                    "before": tables[side][node],
                    "after": state,
                    "reason": reason,
                }
            )
            tables[side][node] = state
        references.append(binding(path))
    allowed_qa_changes = {
        "scripts/run_stage70_test_audit.py",
        "scripts/qa_support/sitecustomize.py",
        "pyright-stage70-audit.json",
    }
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", "4a8b545", "0777326"], cwd=ROOT, text=True
        ).splitlines()
    )
    if not changed <= allowed_qa_changes:
        raise ValueError("focused environment recheck revision changed non-QA code")
    failed = {
        side: {node for node, state in table.items() if state == "failed"}
        for side, table in tables.items()
    }
    shared = tables["baseline"].keys() & tables["current"].keys()
    result = {
        "kind": "BASELINE_VERSUS_CURRENT_EXACT_FAILURE_IDS",
        "baseline_source_commit": baseline["source_commit"],
        "current_full_suite_commit": subprocess.check_output(
            ["git", "rev-parse", "4a8b545"], cwd=ROOT, text=True
        ).strip(),
        "current_environment_recheck_commit": subprocess.check_output(
            ["git", "rev-parse", "0777326"], cwd=ROOT, text=True
        ).strip(),
        "counts": {
            side: {
                state: sum(value == state for value in table.values())
                for state in ("passed", "failed", "skipped")
            }
            for side, table in tables.items()
        },
        "baseline_failure_ids": sorted(failed["baseline"]),
        "current_failure_ids": sorted(failed["current"]),
        "new_failure_ids": sorted(failed["current"] - failed["baseline"]),
        "resolved_failure_ids": sorted(failed["baseline"] - failed["current"]),
        "added_test_ids": sorted(tables["current"].keys() - shared),
        "removed_test_ids": sorted(tables["baseline"].keys() - shared),
        "environment_rechecks": changes,
        "source_bindings": references,
        "initial_new_failure_corrected": (
            "test_v41_learns_a_pure_temporal_event_signal; reference-test thread setting "
            "moved from collection-time global state to a restoring module fixture"
        ),
        "earlier_current_attempt": binding(qa / "current_a01c6b7_full/TEST_DISPOSITION.json"),
        "all_old_failure_observations_preserved": True,
        "scientific_training_lock_ready": False,
        "scope_note": (
            "Coverage is for the named source revisions. Remaining orchestration and "
            "diagnostic implementation still needs its own required tests before scientific freeze."
        ),
    }
    atomic_json(output, {**result, "sha256": object_digest(result)})
    print(json.dumps({"counts": result["counts"], "new_failure_ids": result["new_failure_ids"]}))


if __name__ == "__main__":
    main()
