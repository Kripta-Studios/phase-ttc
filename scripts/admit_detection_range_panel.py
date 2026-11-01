"""Check target capacity and counts on a frozen panel before GPU training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.eap_rgb_data import RawRGBSample, finite_matrix  # noqa: E402
from e_jepa_ttc.detection.monodgp_eap_adapter import prepare_sample, resized_intrinsic  # noqa: E402
from e_jepa_ttc.detection.range_campaign import RangeConfig, digest, write_json  # noqa: E402


def main() -> None:
    """Use exact geometry with blank images, without loading RGB or using selection labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = RangeConfig.load(args.config)
    if digest(Path(cfg.population)) != cfg.population_sha256:
        raise ValueError("Population changed")
    population = json.loads(Path(cfg.population).read_text())
    tokens = population["roles"]["train"]
    media = pd.read_parquet(
        Path(cfg.dataset_root) / "data/train.parquet", filters=[("sample_token", "in", tokens)]
    )
    if len(media) != len(tokens):
        raise ValueError("Incomplete metadata")
    labels = {}
    for sequence in sorted({t.split(":")[0] for t in tokens}):
        name = f"data/train/{sequence}/labels.parquet"
        if digest(Path(cfg.dataset_root) / name) != population["label_sha256"][name]:
            raise ValueError("Labels changed")
        table = pd.read_parquet(
            Path(cfg.dataset_root) / name, filters=[("sample_token", "in", tokens)]
        )
        labels.update(
            {
                str(t): rows.to_dict("records")
                for t, rows in table.groupby("sample_token", sort=False)
            }
        )
    torch.set_num_threads(2)
    blank = np.zeros((384, 640, 3), dtype=np.uint8)
    counts = {str(depth): Counter() for depth in (60, 150)}
    maxima = dict.fromkeys(counts, 0)
    overflow = []
    start = time.perf_counter()
    for row in media.to_dict("records"):
        token = row["sample_token"]
        k = resized_intrinsic(
            finite_matrix(row["K_event"], (3, 3)),
            tuple(population["media"][token]["image_wh"]),
            (640, 384),
        )
        raw = RawRGBSample(
            token,
            token.split(":")[0],
            blank,
            labels.get(token, []),
            k,
            finite_matrix(row["T_event_ego"], (4, 4)),
            0,
            0,
            {},
            "admission_geometry_only",
        )
        for depth in (60, 150):
            prepared = prepare_sample(raw, max_depth=float(depth), max_objects=1000000)
            n = len(prepared.native_ground_truth)
            counts[str(depth)].update(str(b["class_name"]) for b in prepared.native_ground_truth)
            maxima[str(depth)] = max(maxima[str(depth)], n)
            if n > 50:
                overflow.append({"token": token, "depth": depth, "targets": n})
    write_json(
        args.output,
        {
            "schema": "phase_range_panel_admission_v1",
            "population_sha256": cfg.population_sha256,
            "frames": len(tokens),
            "counts": {k: dict(v) for k, v in counts.items()},
            "maximum_targets_per_frame": maxima,
            "over_capacity": overflow,
            "status": "PASS" if not overflow else "FAIL",
            "elapsed_seconds": time.perf_counter() - start,
            "script_sha256": digest(Path(__file__)),
        },
    )
    if overflow:
        raise ValueError("Panel exceeds query capacity")


if __name__ == "__main__":
    main()
