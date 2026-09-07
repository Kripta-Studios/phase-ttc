"""Freeze and execute the authorized Stage70–76 graph using its existing runtimes."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.training.stage70_update_ledger import UpdateLedger
from e_jepa_ttc.training.stage72_73_runtime_v11 import InitialArchitectureRuntime
from e_jepa_ttc.training.stage72_jobs import require_scientific_lock
from e_jepa_ttc.training.stage73_factorial import Availability, initial_jobs, stage_update_total
from e_jepa_ttc.training.stage74_runtime_v11 import run_replication
from e_jepa_ttc.training.stage75_router_runtime_v11 import run_nested_routers
from e_jepa_ttc.training.stage75_runtime_v11 import run_new_nested_producers
from e_jepa_ttc.training.stage76_confirmation import publish_blocked_confirmation


def signed(path: Path) -> dict[str, Any]:
    """Read a content-addressed local manifest without granting new source roles."""
    record = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in record.items() if key != "artifact_sha256"}
    if object_digest(unsigned) != record.get("artifact_sha256"):
        raise ValueError(f"manifest signature changed: {path.name}")
    return record


def freeze(root: Path, campaign: Path, local_paths: Path) -> tuple[Path, str]:
    """Compile the lock only from a clean commit and the completed finite QA closure."""
    path = root / "TRAINING_LOCK.json"
    if path.exists():
        record = signed(path)
        require_scientific_lock(path, record["artifact_sha256"])
        if record["protocol"] != binding(campaign) or record["local_paths"] != binding(local_paths):
            raise ValueError("resume protocol or resolved path map differs from the frozen inputs")
        return path, record["artifact_sha256"]
    closure = signed(root / "preflight/FREEZE_CLOSURE.json")
    if closure["status"] != "PASS" or set(closure["closed_blockers"]) != {
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
    }:
        raise ValueError("first fit is blocked by the finite mandatory closure list")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tracked = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    )
    if tracked or closure["code_commit"] != commit:
        raise ValueError("final QA must identify the current clean implementation commit")
    draft = signed(verify(closure["resolved_lock_inputs"]))
    if draft["campaign_root"] != str(root):
        raise ValueError("resolved lock inputs belong to a different output root")
    for source in closure["preflight_receipts"].values():
        verify(source)
    for source in closure["scientific_files"]:
        verify(source)
    ledger = json.loads((root / "UPDATE_LEDGER.json").read_text(encoding="utf-8"))
    ledger_signature = ledger.pop("sha256")
    if object_digest(ledger) != ledger_signature:
        raise ValueError("optimizer ledger signature changed before freeze")
    if any(job["charged"] for job in ledger["jobs"].values() if not job["spec"]["technical"]):
        raise ValueError("cannot create a first-fit lock after scientific optimizer exposure")
    if ledger.get("pending") is not None:
        raise ValueError("resolve the uncertain optimizer intent before freeze")
    record = {
        **{key: value for key, value in draft.items() if key != "artifact_sha256"},
        "kind": "STAGE70_76_TRAINING_LOCK",
        "code_commit": commit,
        "tracked_worktree_clean_at_freeze": True,
        "implementation_complete_before_first_fit": True,
        "scientific_updates_at_freeze": 0,
        "protocol": binding(campaign),
        "local_paths": binding(local_paths),
        "preflight_receipts": closure["preflight_receipts"],
        "scientific_files": closure["scientific_files"],
        "closure": binding(root / "preflight/FREEZE_CLOSURE.json"),
    }
    record["artifact_sha256"] = object_digest(record)
    atomic_json(path, record)
    require_scientific_lock(path, record["artifact_sha256"])
    return path, record["artifact_sha256"]


def execute_capacity_comparator_graph(
    lock_path: Path,
    lock_sha: str,
    owner: ResourceOwner,
    ledger: UpdateLedger,
    reserve_resources: Callable[[str], None],
) -> dict[str, Any]:
    """Finish all encoder/head fits before exporting the shared nested comparison."""
    from e_jepa_ttc.training.stage70_update_ledger import FitSpec
    from scripts.build_scientific_recovery_v9_stage61_reference import execute_capacity_pair
    from scripts.run_scientific_recovery_v9_stage65 import publish_capacity_risk17
    from scripts.train_scientific_recovery_v8_router_expert import execute_capacity_comparator

    lock = require_scientific_lock(lock_path, lock_sha)
    registry = {}
    expected = {(f, i, a) for f in range(3) for i in (None, 0, 1, 2) for a in ("A5", "C2F", "PAIR")}
    for job_id, job in lock["comparator_jobs"].items():
        spec = FitSpec(**job["spec"])
        protocol = json.loads(verify(job["protocol"]).read_text(encoding="utf-8"))
        key = (spec.fold, protocol["inner_fold"], spec.arm)
        if (
            spec.job_id != job_id
            or spec.technical
            or spec.seed != 7
            or spec.stage != "COMPARATOR"
            or key not in expected
            or key in registry
        ):
            raise ValueError("expanded comparator graph has a substituted or duplicate producer")
        registry[key] = job_id
    if set(registry) != expected:
        raise ValueError("expanded comparator graph requires24 encoders and12 PAIR heads")
    for (fold, inner, arm), job_id in registry.items():
        if (
            arm == "PAIR"
            and lock["comparator_jobs"][job_id]["a5_job_id"] != registry[(fold, inner, "A5")]
        ):
            raise ValueError("PAIR graph crosses parent encoder producers")
    for job in lock["comparator_jobs"].values():
        ledger.reserve(FitSpec(**job["spec"]))
    progress_path = Path(lock["campaign_root"]) / "COMPARATOR_GRAPH_PROGRESS.json"
    for fit_only in (True, False):
        for arms, runner in (
            (("A5", "C2F"), execute_capacity_comparator),
            (("PAIR",), execute_capacity_pair),
        ):
            for key, job_id in registry.items():
                if key[2] not in arms:
                    continue
                reserve_resources(job_id if fit_only else job_id + "_export")
                atomic_json(
                    progress_path,
                    {
                        "training_lock_sha256": lock_sha,
                        "active_job": job_id,
                        "operation": "FIT" if fit_only else "EXPORT",
                        "prescribed_jobs": list(registry.values()),
                        "pending_endpoints": [
                            j
                            for j in registry.values()
                            if ledger.state["jobs"][j]["status"] != "COMPLETE"
                        ],
                        "optimizer_totals": ledger.totals(),
                    },
                )
                runner(
                    job_id,
                    lock_path=lock_path,
                    lock_sha256=lock_sha,
                    ledger=ledger,
                    owner=owner,
                    device=torch.device("cuda"),
                    fit_only=fit_only,
                )
    reserve_resources("CAPACITY_RISK17_PUBLICATION")
    result = publish_capacity_risk17(lock_path, lock_sha, ledger, owner)
    atomic_json(
        progress_path,
        {
            "training_lock_sha256": lock_sha,
            "operation": "COMPLETE",
            "pending_endpoints": [],
            "optimizer_totals": ledger.totals(),
            "comparison": binding(
                Path(lock["capacity_comparison_output"]) / "COMPARISON_COMPLETE.json"
            ),
        },
    )
    return result


def execute(root: Path, lock_path: Path, lock_sha: str) -> dict[str, Any]:
    """Run every independent stage and each triggered branch synchronously."""
    lock = {**require_scientific_lock(lock_path, lock_sha), "artifact_sha256": lock_sha}
    result_path = root / "CAMPAIGN_EXECUTION.json"
    result: dict[str, Any] = {
        "training_lock_sha256": lock_sha,
        "status": "RUNNING",
        "stages": {},
        "confirmation_labels_read": False,
        "active_operation": "INITIAL_FITS",
    }
    with ResourceOwner(
        root / "training_writer",
        remaining_bytes=lock["startup_reservation"]["disk_bytes"],
        planned_ram_bytes=lock["startup_reservation"]["ram_bytes"],
    ) as owner:
        ledger = UpdateLedger(root / "UPDATE_LEDGER.json", owner.root)
        try:
            if "capacity_cohort" in lock:
                ledger.bind_capacity_envelope(lock["shared_budget_envelope"])
                ledger.inherit_closed_campaign(lock["predecessor_execution"])
                ledger.carry_preparation_accounting(lock["preparation_accounting"])
            runtime = InitialArchitectureRuntime(lock_path, lock_sha, owner, ledger)
            atomic_json(result_path, result)
            barriers = runtime.fit_initial_stages()
            result["stages"].update(
                {
                    stage: {"status": "ENDPOINTS_FROZEN", "barrier": source}
                    for stage, source in barriers.items()
                }
            )
            if "capacity_cohort" in lock:
                result["active_operation"] = "CAPACITY_COMPARATORS"
                atomic_json(result_path, result)
                execute_capacity_comparator_graph(
                    lock_path, lock_sha, owner, ledger, runtime._reserve_resources
                )
                result["capacity_comparators"] = binding(
                    Path(lock["capacity_comparison_output"]) / "COMPARISON_COMPLETE.json"
                )
            result["active_operation"] = "INITIAL_INFERENCE_AND_ANALYSIS"
            atomic_json(result_path, result)
            exports = runtime.infer_and_export(barriers)
            decisions = runtime.analyze_initial_stages(exports, barriers)
            for stage in ("S72", "S73"):
                result["stages"][stage]["status"] = "COMPLETE"
                result["stages"][stage]["exports"] = {
                    key: value for key, value in exports.items() if key.startswith(stage)
                }
            result["active_operation"] = "S74"
            atomic_json(result_path, result)
            replication = run_replication(runtime, decisions["S74"])
            result["stages"]["S74"] = replication
            result["active_operation"] = "S75_NEW_INNER_PRODUCERS"
            atomic_json(result_path, result)
            producers = run_new_nested_producers(runtime, decisions["S75"])
            result["active_operation"] = "S75_ROUTING"
            atomic_json(result_path, result)
            routers = run_nested_routers(runtime, decisions["S75"])
            result["stages"]["S75"] = {"producers": producers, "routers": routers}
            result["active_operation"] = "S76_DISPOSITION"
            atomic_json(result_path, result)
            result["stages"]["S76"] = publish_blocked_confirmation(
                lock, replication, root / "S76_DISPOSITION.json"
            )
            result["status"] = "AUTHORIZED_GRAPH_TERMINAL"
            result["active_operation"] = None
        except BaseException as error:
            result["status"] = "INTERRUPTED"
            result["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            raise
        finally:
            result["optimizer_totals"] = ledger.totals()
            result["update_ledger"] = binding(ledger.path)
            atomic_json(result_path, result)
    return result


def parser() -> argparse.ArgumentParser:
    """Expose explicit mutually exclusive modes; execution defaults to the entire graph."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--campaign", type=Path, required=True, help="Immutable handoff protocol JSON")
    cli.add_argument(
        "--capacity",
        action="store_true",
        help="Use the separate v11 capacity campaign; all freeze and predecessor gates still apply",
    )
    cli.add_argument(
        "--local-paths", type=Path, required=True, help="Resolved local operational path map"
    )
    cli.add_argument(
        "--output-root", type=Path, required=True, help="Owned Stage70–76 output directory"
    )
    mode = cli.add_mutually_exclusive_group(required=True)
    for flag in ("preflight-only", "validate-plan", "execute-authorized", "package-only"):
        mode.add_argument("--" + flag, action="store_true")
    cli.add_argument(
        "--resume",
        action="store_true",
        help="Recover exact existing state, never select an earlier checkpoint",
    )
    cli.add_argument(
        "--stage",
        choices=("ALL", "S72", "S73", "S74", "S75", "S76"),
        default="ALL",
        help=(
            "Plan inspection filter; scientific execution requires ALL "
            "to preserve independent branches"
        ),
    )
    return cli


def main() -> int:
    """Validate source bindings, execute the graph, stop writers, then package evidence."""
    cli = parser()
    args = cli.parse_args()
    if args.resume and not args.execute_authorized:
        cli.error("--resume requires --execute-authorized")
    if args.stage != "ALL" and not args.validate_plan:
        cli.error("--stage is a plan inspection filter; execute the full authorized graph")
    root = args.output_root.resolve(strict=True)
    expected_root = ROOT / (
        "artifacts/stage70_76_capacity" if args.capacity else "artifacts/stage70_76_architecture"
    )
    if root != expected_root:
        raise ValueError("launcher cannot mutate another campaign or historical output")
    campaign, local_paths = (
        args.campaign.resolve(strict=True),
        args.local_paths.resolve(strict=True),
    )
    verification = json.loads(
        (root / "preflight/INPUT_VERIFICATION.json").read_text(encoding="utf-8")
    )
    if binding(campaign) not in verification["handoff_files"]:
        raise ValueError("campaign protocol is absent from the verified immutable handoff")
    if args.validate_plan or args.preflight_only:
        graph = signed(root / "resolved_initial_source_graph/INITIAL_SOURCE_GRAPH.json")
        plans = initial_jobs(Availability(**graph["availability"]))
        print(
            json.dumps(
                {
                    "initial_jobs": [
                        plan.job_id
                        for plan in plans
                        if args.stage == "ALL" or plan.stage == args.stage
                    ],
                    "initial_updates": stage_update_total(plans),
                    "canonical_candidates": graph["canonical_candidates"],
                    "freeze_closure_present": (root / "preflight/FREEZE_CLOSURE.json").exists(),
                    "scientific_fit_started": False,
                },
                indent=2,
            )
        )
        return 0
    if args.execute_authorized:
        if (root / "TRAINING_LOCK.json").exists() and not args.resume:
            cli.error("existing scientific lock requires explicit --resume")
        lock_path, lock_sha = freeze(root, campaign, local_paths)
        locked = require_scientific_lock(lock_path, lock_sha)
        if ("capacity_cohort" in locked) != args.capacity:
            raise ValueError("launcher mode differs from the frozen campaign identity")
        execute(root, lock_path, lock_sha)
    from package_scientific_recovery_v10_stage70_76 import package

    package(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
