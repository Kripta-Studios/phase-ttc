"""Register the 300m extension and a real far-target smoke before further GPU work."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.eap_rgb_data import finite_matrix  # noqa: E402
from e_jepa_ttc.detection.long_range_campaign import LongRangeConfig  # noqa: E402
from e_jepa_ttc.detection.range_campaign import RangeConfig, digest, write_json  # noqa: E402


def main() -> None:
    """Reuse the exact screening panel and budget; only range admission changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-config", required=True, type=Path)
    parser.add_argument("--census", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    control = RangeConfig.load(args.control_config)
    if control.depth_max_m != 150.0:
        raise ValueError("Expected 150m comparison arm")
    census = json.loads(args.census.read_text())
    if (
        census["status"] != "VERIFIED"
        or census["test_labels_read"] is not False
        or census["raw_center_depth_extrema_m"][1] >= 300
        or census["capacity"]["admitted400"]["over_50"]
    ):
        raise ValueError("Census does not justify the registered all-observed-target range")
    if digest(Path(control.population)) != control.population_sha256:
        raise ValueError("Population changed")
    population = json.loads(Path(control.population).read_text())
    if census["media_admission_sha256"] != population["admission_sha256"]:
        raise ValueError("Different admitted media")
    release = Path(control.release_root)
    media = json.loads((release / "detector/code/data/admission/media_admission.json").read_text())
    tokens = population["roles"]["train"]
    if any(media["final_media"][t]["role"] != "train" for t in tokens):
        raise ValueError("Non-TRAIN input")
    dataset = Path(control.dataset_root)
    metadata = pd.read_parquet(
        dataset / "data/train.parquet",
        filters=[("sample_token", "in", tokens)],
        columns=["sample_token", "T_event_ego"],
    )
    transforms = {
        str(t): finite_matrix(raw, (4, 4)) for t, raw in metadata.itertuples(index=False, name=None)
    }
    farthest = (-float("inf"), "")
    for sequence in sorted({t.split(":")[0] for t in tokens}):
        name = f"data/train/{sequence}/labels.parquet"
        if digest(dataset / name) != census["label_sha256"][name]:
            raise ValueError("TRAIN labels differ")
        rows = pd.read_parquet(
            dataset / name,
            filters=[("sample_token", "in", tokens)],
            columns=["sample_token", "bbox_3d_ego"],
        )
        for token, box in rows.itertuples(index=False, name=None):
            transform = transforms[token]
            center = transform[:3, :3] @ np.asarray(box[:3]) + transform[:3, 3]
            if float(center[2]) > farthest[0]:
                farthest = (float(center[2]), str(token))
    if not 200 < farthest[0] < 300:
        raise ValueError("Smoke must exercise a real target beyond 200m")
    args.output.mkdir(parents=True, exist_ok=False)
    values = {
        **control.__dict__,
        "schema": "phase_eap_long_range_campaign_v1",
        "depth_max_m": 300.0,
        "experiment_id": "range300-seed7-screen-v1",
    }
    cfg = LongRangeConfig(**values)
    cfg.validate()
    write_json(args.output / "range300.json", asdict(cfg))
    smoke_tokens = [farthest[1], *[t for t in tokens if t != farthest[1]][:15]]
    smoke_selection = population["roles"]["selection"][:16]
    smoke_population = {
        **population,
        "sampling": "TRAIN farthest optical-depth frame plus first 15 other panel tokens; "
        "first 16 frozen selection tokens, for numerical smoke only",
        "roles": {"train": smoke_tokens, "selection": smoke_selection},
        "media": {t: population["media"][t] for t in smoke_tokens + smoke_selection},
    }
    pop_path = args.output.resolve() / "smoke_population.json"
    write_json(pop_path, smoke_population)
    smoke = replace(
        cfg,
        experiment_id="range300-seed7-smoke-v1",
        population=str(pop_path),
        population_sha256=digest(pop_path),
        updates=8,
        evaluation_updates=[0, 8],
        checkpoint_interval=4,
        maximum_seconds=600.0,
    )
    smoke.validate()
    write_json(args.output / "smoke300.json", asdict(smoke))
    write_json(
        args.output / "PLAN.json",
        {
            "schema": "phase_eap_long_range_plan_v1",
            "status": "PREPARED_PENDING_GPU_SMOKE",
            "hypothesis": (
                "Supervise all geometrically admissible TRAIN objects instead of cutting at 150m"
            ),
            "range_m": 300,
            "why_300": "Census maximum below 250m; 300m provides declared margin",
            "control": str(args.control_config.resolve()),
            "control_config_sha256": digest(args.control_config),
            "census": str(args.census.resolve()),
            "census_sha256": digest(args.census),
            "screen_config_sha256": digest(args.output / "range300.json"),
            "smoke_config_sha256": digest(args.output / "smoke300.json"),
            "paired_settings": "Same Dfull initialization policy, 4096 TRAIN/512 selection panel, "
            "seed7, 2048 updates, batch2, FP32, LR/scheduler, validation schedule",
            "smoke_far_target": {"token": farthest[1], "optical_depth_m": farthest[0]},
            "screen_maximum_seconds": cfg.maximum_seconds,
            "resource_scope": "Local GPU only, one GPU process at a time, no cloud spending",
            "primary": "Eight-class aligned raw-GT mAP",
            "promotion": "Full-selection improvement >=0.005 over incumbent and valid control; "
            "review every class decline >0.01; no promotion from this panel alone",
            "selection_exposure": "Exploratory extension after seeing partial 60/150 panel curves; "
            "not independent confirmation",
            "smoke_gate": "Finite loss/gradient/checkpoint, far target admitted, exact initial "
            "label-free predictions versus same frozen reference tokens",
            "test_labels_read": False,
            "producer_source_sha256": digest(ROOT / "scripts/train_detection_long_range.py"),
            "config_source_sha256": digest(
                ROOT / "src/e_jepa_ttc/detection/long_range_campaign.py"
            ),
        },
    )


if __name__ == "__main__":
    main()
