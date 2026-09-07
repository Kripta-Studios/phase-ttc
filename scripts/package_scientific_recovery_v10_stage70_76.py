"""Report and package closed Stage70–76 evidence using the existing verified bundler."""

# ruff: noqa: E402, E501 -- repository bootstrap and report prose
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psutil

from e_jepa_ttc.artifacts.essential_bundle_v10 import create_bundle
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import verify_initial_barrier
from e_jepa_ttc.training.stage72_jobs import require_scientific_lock


def package(root: Path) -> dict[str, Any]:
    """Build reports from actual stage receipts, preserving missing and failed evidence."""
    root = root.resolve(strict=True)
    if root not in {
        ROOT / "artifacts/stage70_76_architecture",
        ROOT / "artifacts/stage70_76_capacity",
    }:
        raise ValueError("package root must be this owned campaign")
    owner_path = root / "training_writer/ACTIVE_OWNER.json"
    if owner_path.exists():
        owner = json.loads(owner_path.read_text())
        if (
            psutil.pid_exists(owner["pid"])
            and psutil.Process(owner["pid"]).create_time() == owner["created"]
        ):
            raise ValueError("stop and flush the training writer before packaging")
    execution = json.loads((root / "CAMPAIGN_EXECUTION.json").read_text(encoding="utf-8"))
    if execution["status"] not in ("AUTHORIZED_GRAPH_TERMINAL", "INTERRUPTED"):
        raise ValueError("running or unexecuted graph cannot be packaged as final evidence")
    lock = json.loads((root / "TRAINING_LOCK.json").read_text(encoding="utf-8"))
    require_scientific_lock(root / "TRAINING_LOCK.json", lock["artifact_sha256"])
    if ("capacity_cohort" in lock) != (root.name == "stage70_76_capacity"):
        raise ValueError("package root differs from the frozen cohort identity")
    if execution["training_lock_sha256"] != lock["artifact_sha256"]:
        raise ValueError("execution belongs to a different scientific lock")
    if execution["status"] == "AUTHORIZED_GRAPH_TERMINAL":
        if set(execution["stages"]) != {"S72", "S73", "S74", "S75", "S76"}:
            raise ValueError(
                "terminal graph omitted an independent or conditional stage disposition"
            )
        for stage in ("S72", "S73"):
            verify_initial_barrier(
                Path(lock["stage_barrier_root"]) / f"{stage}_ALL_ENDPOINTS_FROZEN.json",
                expected_stage=stage,
                expected_lock=lock["artifact_sha256"],
            )
    comparison = None
    if "capacity_cohort" in lock and execution["status"] == "AUTHORIZED_GRAPH_TERMINAL":
        from e_jepa_ttc.evaluation.initial_campaign_evidence_v11 import capacity_comparison_receipt

        _, comparison = capacity_comparison_receipt(lock)
    ledger = json.loads((root / "UPDATE_LEDGER.json").read_text(encoding="utf-8"))
    ledger_signature = ledger.pop("sha256")
    if object_digest(ledger) != ledger_signature:
        raise ValueError("optimizer accounting changed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    scientific = [job for job in ledger["jobs"].values() if not job["spec"]["technical"]]
    completed = [job for job in scientific if job["status"] == "COMPLETE"]
    decision = {
        "campaign_status": execution["status"],
        "training_commit": lock["code_commit"],
        "analysis_commit": commit,
        "canonical_candidates": lock["canonical_candidates"],
        "scientific_optimizer_updates": sum(job["confirmed"] for job in scientific),
        "scientific_optimizer_fits_started": sum(job["charged"] > 0 for job in scientific),
        "scientific_optimizer_endpoints_complete": len(completed),
        "scientific_pretext_endpoints_complete": sum(
            job["spec"]["arm"] == "CORRESPONDENCE" for job in completed
        ),
        "total_accounting": execution["optimizer_totals"],
        "stages": execution["stages"],
        "expanded_comparison": comparison,
        "independent_confirmation": False,
        "protected_evaluations": "CLOSED",
        "active_training_writer": False,
        "next_action": "RESUME_EXACT_PENDING_GRAPH"
        if execution["status"] == "INTERRUPTED"
        else "REVIEW_SCIENTIFIC_EVIDENCE_AND_SCOPED_PREREQUISITES",
        "interruption": execution.get("error"),
        "historical_gates_modified": False,
        "limitations": [
            "development groups reused; bootstrap fractions are not Bayesian posterior probabilities",
            "foreground truth unavailable; geometry masks measure observability, not object truth",
            "sequence grouping is an acquisition fallback, not verified independent acquisitions",
            "TRAIN mechanism panels are nonselectable diagnostics, not independent confirmation",
            "raw latency uses warm filesystem caches and fixed random TRAIN order",
            "full deployed expert cost must include every old and new producer; standalone parameter count is not system size",
            "missing FLOP or historical full-stack timing measurements remain explicitly unmeasured",
        ],
    }
    atomic_json(root / "NEXT_DECISION_V4.json", decision)
    lines = [
        "# E-JEPA-TTC Stage70–76 final report",
        "",
        f"Campaign status: **{execution['status']}**.",
        "",
        f"Scientific optimizer endpoints: {len(completed)}; confirmed scientific updates: {decision['scientific_optimizer_updates']}.",
        f"Training commit: `{lock['code_commit']}`. Analysis commit: `{commit}`.",
        "",
        "S72 PPS-TRAJ remains the primary hypothesis. Candidate identities were resolved before any scientific fit.",
        "S73 is independent of S72 scientific success. Replication retains all prescribed seeds; routing uses genuinely nested new producers.",
        "Independent confirmation did not run under the frozen S76 prerequisite block. This is not a scientific negative.",
        "",
        "| Treatment / seed | MiD | Finite rows | Total rows |",
        "|---|---:|---:|---:|",
    ]
    for field in ("treatment_export_root", "replication_export_root"):
        for path in sorted(Path(lock[field]).rglob("TREATMENT_EXPORT.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            summary = result["summary"]
            lines.append(
                f"| {result['treatment']} / {result['seed']} | {summary['score']} | {summary['finite_rows']} | {summary['rows']} |"
            )
    lines.extend(
        [
            "",
            "Stage decisions, grouped uncertainty and all conditional reasons are retained verbatim in NEXT_DECISION_V4.json and their source-bound analysis receipts.",
            "",
            "Utility, mechanism, optimization replication and independent confirmation are separate claims. The old selector oracle is a lower bound for that exact selector class, not an information-theoretic data floor. The learned scale model is a tested hypothesis, not a universal event-motion law.",
            "",
            "Limitations:",
            "",
        ]
    )
    lines.extend("- " + text for text in decision["limitations"])
    lines.extend(
        [
            "",
            "Actual accounting and stage dispositions:",
            "",
            "```json",
            json.dumps(decision, indent=2),
            "```",
            "",
        ]
    )
    (root / "CODEX_STAGE70_76_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    files: dict[str, Path] = {}
    omitted = []
    allowed = {".json", ".jsonl", ".csv", ".npz", ".xml", ".md", ".txt", ".log"}
    directories = {
        Path(lock[key])
        for key in (
            "endpoint_root",
            "outer_inference_root",
            "inner_inference_root",
            "treatment_export_root",
            "replication_export_root",
            "initial_evidence_root",
            "replication_evidence_root",
            "stage_barrier_root",
            "stage_decision_root",
            "nested_source_root",
            "nested_export_root",
            "router_root",
        )
    }
    directories.update(
        root / name
        for name in (
            "preflight",
            "data_roles",
            "resolved_views_original",
            "resolved_views_expanded",
            "prospective_nested_source_recipes",
            "historical_nested_tables",
            "historical_simplex_replay",
        )
    )
    if "capacity_cohort" in lock:
        directories.add(Path(lock["capacity_comparison_output"]))
        # Keep the closed prediction products and lineage; native TRAIN caches stay local.
        for job in lock["comparator_jobs"].values():
            producer = Path(job["output_root"])
            if producer.resolve() != producer or not producer.is_relative_to(root):
                raise ValueError("comparator package source escaped its owned campaign")
            for path in sorted((producer / "prediction").glob("*")):
                if path.is_symlink() or path.resolve() != path.absolute():
                    raise ValueError("comparator package source was redirected")
                if path.is_file() and path.suffix in allowed:
                    files["run/" + path.relative_to(root).as_posix()] = path
    for directory in sorted(directories):
        if directory.resolve().parent != root:
            raise ValueError("bundle directory escaped the owned campaign")
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if path.resolve() != path.absolute() or path.is_symlink():
                raise ValueError("bundle source was redirected")
            if path.suffix in allowed:
                files["run/" + path.relative_to(root).as_posix()] = path
            else:
                omitted.append(binding(path))
    for path in root.glob("*.json"):
        files["run/" + path.name] = path
    for pattern in ("*.xml", "*DISPOSITION.json", "BASELINE_VERSUS_CURRENT*.json"):
        for path in (root / "baseline_qa").rglob(pattern):
            files["run/" + path.relative_to(root).as_posix()] = path
    for name in ("CODEX_STAGE70_76_FINAL_REPORT.md", "NEXT_DECISION_V4.json"):
        files[name] = root / name
    # Only manifests for heavy counts, not count arrays or recursively nested old ZIPs.
    for name, source in lock["count_caches"].items():
        files[f"local_manifests/{name}_CACHE_MANIFEST.json"] = Path(source["path"])
    atomic_json(
        root / "OMITTED_LOCAL_PAYLOADS.json",
        {
            "files": omitted,
            "raw_and_cache": lock["count_caches"],
            "immutable_input_references": lock["preflight_receipts"]["handoff_integrity"],
        },
    )
    files["run/OMITTED_LOCAL_PAYLOADS.json"] = root / "OMITTED_LOCAL_PAYLOADS.json"
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for name in tracked:
        if name and (
            name.startswith(("src/", "scripts/", "configs/", "tests/", "docs/"))
            or name in {"pyproject.toml", "uv.lock", "AGENTS.md", "pyright-stage70-audit.json"}
        ):
            files["code/" + name] = ROOT / name
    # Verify the closed payload once; the final archive is independently reverified
    # by the same utility after adding this non-circular payload verification receipt.
    files.pop("run/BUNDLE_VERIFICATION.json", None)
    payload = root / f"PAYLOAD_VERIFICATION_{commit[:12]}.zip"
    check = create_bundle(files, payload)
    atomic_json(
        root / "BUNDLE_VERIFICATION.json",
        {
            "payload_verification": check,
            "all_writers_closed": True,
            "final_zip_verification": "see adjacent .verification.json and .zip.sha256; not a self-referential hash",
        },
    )
    files["BUNDLE_VERIFICATION.json"] = root / "BUNDLE_VERIFICATION.json"
    output = root / f"E_JEPA_TTC_STAGE70_76_ESSENTIAL_RESULTS_{commit[:12]}.zip"
    return create_bundle(files, output)


def main() -> int:
    """Package a closed campaign; never start a fit or open raw/confirmation data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
