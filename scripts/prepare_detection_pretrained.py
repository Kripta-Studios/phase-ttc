"""Register one initialization comparison before its first GPU measurements."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.long_range_campaign import LongRangeConfig  # noqa: E402
from e_jepa_ttc.detection.pretrained_campaign import (  # noqa: E402
    PretrainedConfig,
    verify_pretrained_artifacts,
)
from e_jepa_ttc.detection.pretrained_initialization import POLICY  # noqa: E402
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def main() -> None:
    """Keep the admitted 300m population, recipe and exposure budget unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-plan", required=True, type=Path)
    parser.add_argument("--conversion", required=True, type=Path)
    parser.add_argument("--cpu-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    control_path = args.control_plan / "range300.json"
    control = LongRangeConfig.load(control_path)
    conversion = json.loads(args.conversion.read_text())
    audit = json.loads(args.cpu_audit.read_text())
    if (
        audit["status"] != "PASS_CPU_ONLY"
        or audit["cuda_initialized"]
        or audit["conversion_sha256"] != digest(args.conversion)
        or audit["transfer_policy"]["policy"] != POLICY
    ):
        raise ValueError("External transfer did not pass its bound real-frame CPU audit")
    values: dict[str, Any] = control.__dict__ | {
        "schema": "phase_eap_pretrained_campaign_v1",
        "experiment_id": "kitti-init-range300-seed7-screen-v1",
        "initialization": POLICY,
    }
    cfg = PretrainedConfig(
        **values,
        pretrained_weights=str((args.conversion.parent / "model_state.pt").resolve()),
        pretrained_weights_sha256=conversion["model_state_sha256"],
        conversion_receipt=str(args.conversion.resolve()),
        conversion_receipt_sha256=digest(args.conversion),
    )
    verify_pretrained_artifacts(cfg)
    if digest(Path(cfg.population)) != cfg.population_sha256:
        raise ValueError("Control population changed")
    smoke_control = LongRangeConfig.load(args.control_plan / "smoke300.json")
    if digest(Path(smoke_control.population)) != smoke_control.population_sha256:
        raise ValueError("Control smoke population changed")
    population = json.loads(Path(smoke_control.population).read_text())
    control_plan = json.loads((args.control_plan / "PLAN.json").read_text())
    far = control_plan["smoke_far_target"]["token"]
    if far not in population["roles"]["train"]:
        raise ValueError("Far target missing from smoke")
    # A repeated two-frame TRAIN batch checks learning; it is never a quality result.
    train = [far, next(t for t in population["roles"]["train"] if t != far)]
    selection = population["roles"]["selection"]
    smoke_population = {
        **population,
        "sampling": (
            "Two fixed TRAIN frames, including the admitted far target; numerical overfit only"
        ),
        "roles": {"train": train, "selection": selection},
        "media": {t: population["media"][t] for t in train + selection},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    pop_path = args.output.resolve() / "smoke_population.json"
    write_json(pop_path, smoke_population)
    smoke = replace(
        cfg,
        experiment_id="kitti-init-range300-seed7-smoke-v1",
        population=str(pop_path),
        population_sha256=digest(pop_path),
        updates=32,
        evaluation_updates=[0, 16, 32],
        checkpoint_interval=16,
        maximum_seconds=600.0,
    )
    smoke.validate()
    write_json(args.output / "screen.json", asdict(cfg))
    write_json(args.output / "smoke.json", asdict(smoke))
    write_json(
        args.output / "PLAN.json",
        {
            "schema": "phase_pretrained_initialization_plan_v1",
            "status": "PREPARED_PENDING_GPU_SMOKE",
            "hypothesis": (
                "A supervised monocular 3D geometry initialization may improve "
                "the corrected eAP recipe"
            ),
            "control_config": str(control_path.resolve()),
            "control_config_sha256": digest(control_path),
            "screen_config_sha256": digest(args.output / "screen.json"),
            "smoke_config_sha256": digest(args.output / "smoke.json"),
            "cpu_audit_sha256": digest(args.cpu_audit),
            "conversion_receipt_sha256": digest(args.conversion),
            "smoke_far_target": control_plan["smoke_far_target"],
            "paired_settings": (
                "Same 4096 TRAIN/512 selection, seed7, 300m, 640x384, batch 2, "
                "FP32, LR 5e-5, warmup cosine, 2048 updates and evaluations "
                "0/512/1024/1536/2048"
            ),
            "changed_factor": (
                "Initialization lineage and mandatory class/depth/velocity head "
                "remapping; no upstream optimizer state"
            ),
            "smoke_gate": (
                "32 repeated-batch updates, finite "
                "outputs/loss/gradients/checkpoint, final 8 mean loss below "
                "first 8, model weights reload, no inference annotations"
            ),
            "primary": "Eight-class aligned raw-GT mAP",
            "screen_budget_seconds": cfg.maximum_seconds,
            "promotion": (
                "Confirm selected checkpoint on all 23958 public selection "
                "frames only if panel gain>=0.005 over matched 300m control and "
                "incumbent; full gain>=0.005 over both with review of any class "
                "drop>0.01"
            ),
            "limits": [
                (
                    "Equal new exposures do not equalize previous training: "
                    "Dfull already consumed 281116 eAP exposures; KITTI upstream "
                    "used Car-only supervision."
                ),
                (
                    "A negative 2048-update screen cannot rule out longer "
                    "external pretraining transfer."
                ),
                (
                    "Public selection is already exposed; this is exploratory "
                    "development, not independent test."
                ),
                "No automatic extra epochs, seeds, cloud jobs, official submission or promotion.",
            ],
            "resource_scope": (
                "Local GPU, one GPU process at a time, after the registered 300m queue"
            ),
            "test_labels_read": False,
            "sources": {
                p.relative_to(ROOT).as_posix(): digest(p)
                for p in (
                    Path(__file__),
                    ROOT / "scripts/train_detection_pretrained.py",
                    ROOT / "src/e_jepa_ttc/detection/pretrained_campaign.py",
                    ROOT / "src/e_jepa_ttc/detection/pretrained_initialization.py",
                )
            },
        },
    )


if __name__ == "__main__":
    main()
