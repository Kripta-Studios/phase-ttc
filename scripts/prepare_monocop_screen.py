"""Bind the tested MonoCoP recipe, observed controls and bounded quality-screen code."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.monocop_campaign import MonoCoPScreenConfig, verify_screen_inputs
from e_jepa_ttc.detection.range_campaign import digest, write_json


def main() -> None:
    """Register the first competitor screen before any quality optimization steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--technical-config", required=True, type=Path)
    parser.add_argument("--technical-gate", required=True, type=Path)
    parser.add_argument("--population", required=True, type=Path)
    parser.add_argument("--control-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new plan directory")
    cfg = MonoCoPScreenConfig(
        "phase_monocop_screen_v1",
        "monocop-all-range300-seed7-screen-v1",
        args.technical_config.as_posix(),
        digest(args.technical_config),
        args.technical_gate.as_posix(),
        digest(args.technical_gate),
        args.population.as_posix(),
        digest(args.population),
        2048,
        [0, 512, 1024, 1536, 2048],
        128,
        7200.0,
        4.0,
    )
    technical, _ = verify_screen_inputs(cfg)
    control = json.loads(args.control_report.read_text())
    if control["schema"] != "phase_completed_pretrained_screen_v1":
        raise ValueError("Verified matched-panel controls are required")
    if (
        control["incumbent_panel"]["samples"] != 512
        or control["incumbent_panel"]["profile_sha256"] != technical.evaluation_profile_sha256
        or control["runs"]["dfull300"]["status"]["status"] != "COMPLETE"
        or control["runs"]["kitti"]["status"]["status"] != "COMPLETE"
    ):
        raise ValueError("Control profile or population differs")
    references = {
        "incumbent_ranked": float(control["incumbent_panel"]["mAP"]),
        "dfull300_best": float(control["runs"]["dfull300"]["best"]),
    }
    # Freeze the full original result; numerical comparisons are rebuilt from it.
    args.output.mkdir(parents=True)
    write_json(args.output / "screen.json", asdict(cfg))
    sources = [
        "scripts/prepare_monocop_screen.py",
        "scripts/train_detection_monocop.py",
        "scripts/run_monocop_gpu_smoke.py",
        "scripts/prepare_monocop_portable.py",
        "scripts/audit_monocop_cpu_port.py",
        *[
            f"src/e_jepa_ttc/detection/{name}.py"
            for name in (
                "monocop_campaign",
                "monocop_smoke",
                "monocop_initialization",
                "range_campaign",
                "benchmark_evaluation",
                "campaign_training",
                "compact_training_cache",
                "resident_input_cache",
                "eap_rgb_data",
                "indexed_rgb_data",
                "monodgp_eap_adapter",
                "geometry_contract",
                "development_metrics",
                "portable_attention",
                "evaluation_protocol",
            )
        ],
    ]
    for name in sources:
        destination = args.output / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    document = ROOT / "docs/detection_sota_20261029/MONOCOP_SCREEN_PLAN_20261101.md"
    shutil.copyfile(document, args.output / document.name)
    plan = {
        "schema": "phase_monocop_screen_plan_v1",
        "decision": "ADMITTED_LOCAL_MONOCOP_SCREEN_ONLY",
        "config_sha256": cfg.sha256,
        "config_file_sha256": digest(args.output / "screen.json"),
        "plan_document_sha256": digest(document),
        "git_commit_at_preparation": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "execution_tree_note": "Archived source hashes identify the exact producer bytes.",
        "sources": {name: digest(ROOT / name) for name in sources},
        "controls": {
            "report": args.control_report.as_posix(),
            "report_sha256": digest(args.control_report),
        },
        "primary": "Eight-class aligned raw-GT mAP on the fixed 512-frame selection panel",
        "reference_panel_mAP": references,
        "promotion": "After 2048 updates, gain >=0.005 over both incumbent and best "
        "Dfull300 panel before full selection confirmation. Full gain >=0.005 over "
        "incumbent and review any per-class AP drop >0.01. No automatic promotion.",
        "maximum_seconds": 7200,
        "maximum_cuda_bytes": 9 * 2**30,
        "new_cloud_spend": False,
        "official_submission": False,
        "automatic_followup": False,
    }
    write_json(args.output / "PLAN.json", plan)
    print(
        json.dumps(
            {
                "status": plan["decision"],
                "config_sha256": cfg.sha256,
                "plan_sha256": digest(args.output / "PLAN.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
