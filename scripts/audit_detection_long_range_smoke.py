"""Gate the new 300m producer on real far targets, numerical health and initial parity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_detection_range_smoke import bin_function  # noqa: E402

from e_jepa_ttc.detection.eap_rgb_data import RawRGBSample, finite_matrix  # noqa: E402
from e_jepa_ttc.detection.long_range_campaign import LongRangeConfig  # noqa: E402
from e_jepa_ttc.detection.monodgp_eap_adapter import prepare_sample, resized_intrinsic  # noqa: E402
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def main() -> None:
    """Fail before screening if any asserted smoke evidence is absent or mismatched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan = json.loads((args.plan_dir / "PLAN.json").read_text())
    cfg = LongRangeConfig.load(args.plan_dir / "smoke300.json")
    contract = json.loads((args.run / "CONTRACT.json").read_text())
    if cfg.sha256 != contract["config_sha256"]:
        raise ValueError("Smoke contract differs")
    status = json.loads((args.run / "STATUS.json").read_text())
    if status["status"] != "COMPLETE" or status["update"] != 8:
        raise ValueError("Smoke did not complete eight updates")
    metrics = [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines()]
    if len(metrics) != 8 or any(
        not math.isfinite(row[key]) for row in metrics for key in ("loss", "gradient_norm")
    ):
        raise ValueError("Incomplete or nonfinite smoke")
    initial = json.loads((args.run / "evaluation/000000.json").read_text())
    reference = json.loads(args.reference.read_text())
    population = json.loads(Path(cfg.population).read_text())
    tokens = population["roles"]["selection"]
    expected = {t: reference["predictions"][t] for t in tokens}
    if initial["predictions"] != expected or initial["label_free_forward"] is not True:
        raise ValueError("Initial label-free prediction parity differs")
    token = plan["smoke_far_target"]["token"]
    data = Path(cfg.dataset_root)
    row = pd.read_parquet(
        data / "data/train.parquet", filters=[("sample_token", "==", token)]
    ).iloc[0]
    annotations = pd.read_parquet(
        data / f"data/train/{token.split(':')[0]}/labels.parquet",
        filters=[("sample_token", "==", token)],
    ).to_dict("records")
    intrinsic = resized_intrinsic(
        finite_matrix(row["K_event"], (3, 3)),
        tuple(population["media"][token]["image_wh"]),
        (640, 384),
    )
    sample = RawRGBSample(
        token,
        token.split(":")[0],
        np.zeros((384, 640, 3), dtype=np.uint8),
        annotations,
        intrinsic,
        finite_matrix(row["T_event_ego"], (4, 4)),
        0,
        0,
        {},
        "geometry_only",
    )
    prepared = prepare_sample(sample, max_depth=300.0, include_velocity=True)
    maximum_depth = float(prepared.target["depth"].max())
    if maximum_depth <= 200 or not any(token in r["sample_tokens"] for r in metrics):
        raise ValueError("The smoke did not train on an admitted far target")
    ddn = args.run / "portable_source/lib/models/monodgp/depth_predictor/ddn_loss/ddn_loss.py"
    probe = torch.tensor([60.0, 150.0, maximum_depth, 299.0])
    bins = bin_function(ddn)(
        SimpleNamespace(range_min=0.001, range_max=300.0, range_bins=80),
        probe,
        mode="LID",
        target=True,
    )
    if not torch.all((bins >= 0) & (bins < 80)):
        raise ValueError("Depth classifier treats an in-range target as background")
    checkpoint = args.run / "checkpoints/last.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if saved["contract"] != contract or saved["update"] != 8:
        raise ValueError("Saved checkpoint lineage differs")
    if not all(torch.isfinite(v).all() for v in saved["model"].values()):
        raise ValueError("Nonfinite saved model")
    write_json(
        args.output,
        {
            "schema": "phase_long_range_smoke_gate_v1",
            "status": "PASS",
            "initial_prediction_exact_parity_tokens": len(tokens),
            "completed_updates": len(metrics),
            "maximum_admitted_depth_m": maximum_depth,
            "far_frame_seen_by_training": True,
            "lid_bins_for_probe_depths": bins.tolist(),
            "probe_depths_m": probe.tolist(),
            "checkpoint_sha256": digest(checkpoint),
            "contract_sha256": digest(args.run / "CONTRACT.json"),
            "reference_sha256": digest(args.reference),
            "source_sha256": digest(Path(__file__)),
        },
    )


if __name__ == "__main__":
    main()
