"""Pre-registered checks that each baseline runs correctly (inside its own venv).

1. Synthetic zoom with a known answer on real eAP crops (first 12 manifest queries).
2. The authors' KITTI sample pair shipped in the CSCV clone, forward order, at the authors'
   KITTI input size (OEXP with its KITTI settings ``maxdisp=512, fac=2``; SFPP 6 iterations).

``python -m operational.literature_baselines.sanity --method OEXP --manifest <npz> --out <dir>``
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from operational.phase_geometry.metrics import json_safe

from .adapters import baselines_root, load, load_oexp
from .common import SYNTHETIC_SIGMAS, box_in_crop, edge_box, zoom_square
from .frames import TarFrameReader, crop, eap_root
from .run import aggregate_output

KITTI_PAIR = ("src/CSCV/000019_10.png", "src/CSCV/000019_11.png")
KITTI_SIZE = (1280, 384)
SYNTHETIC_QUERIES = 12


def synthetic_zoom(method: str, manifest: dict[str, np.ndarray], device: torch.device) -> dict:
    """Median rho_hat per sigma over the first manifest queries versus ``-log(sigma)``."""
    predictor = load(method, device)
    reader = TarFrameReader(eap_root())
    table: dict[str, list[float]] = {f"{s:.2f}": [] for s in SYNTHETIC_SIGMAS}
    for index in range(SYNTHETIC_QUERIES):
        square = tuple(float(v) for v in manifest["square"][index])
        frame = reader.read(
            str(manifest["shard_current"][index]), str(manifest["member_current"][index])
        )
        x0, y0, x1, y1 = edge_box(manifest["box_current"][index])
        centre = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
        image1 = crop(frame, square)
        crop_box = box_in_crop(manifest["box_current"][index], square)
        for sigma in SYNTHETIC_SIGMAS:
            image2 = crop(frame, zoom_square(square, centre, sigma))
            stats = aggregate_output(predictor.run(image1, image2), crop_box)
            table[f"{sigma:.2f}"].append(stats["median"])
    rows = []
    for sigma in SYNTHETIC_SIGMAS:
        values = np.asarray(table[f"{sigma:.2f}"])
        truth = -math.log(sigma)
        rows.append(
            {
                "sigma": sigma,
                "rho_true": truth,
                "rho_hat_median": float(np.nanmedian(values)),
                "abs_error_median": float(np.nanmedian(np.abs(values - truth))),
                "sign_correct_fraction": (
                    float(np.mean(np.sign(values) == np.sign(truth))) if sigma != 1.0 else None
                ),
                "non_finite": int((~np.isfinite(values)).sum()),
            }
        )
    medians = [r["rho_hat_median"] for r in rows]
    monotone = all(a > b for a, b in zip(medians, medians[1:], strict=False))
    signs = all(
        math.copysign(1.0, r["rho_hat_median"]) == math.copysign(1.0, r["rho_true"])
        for r in rows
        if r["sigma"] != 1.0
    )
    return {
        "queries": SYNTHETIC_QUERIES,
        "rows": rows,
        "monotone_decreasing": monotone,
        "median_signs_correct": signs,
        "passed": bool(monotone and signs),
    }


def _kitti_images() -> list[np.ndarray]:
    images = []
    for relative in KITTI_PAIR:
        with Image.open(baselines_root() / relative) as image:
            resized = image.convert("RGB").resize(KITTI_SIZE, Image.Resampling.BILINEAR)
            images.append(np.asarray(resized, dtype=np.uint8))
    return images


def kitti_sample(method: str, device: torch.device, out: Path) -> dict[str, Any]:
    """Forward-order log(Z2/Z1) on the authors' KITTI pair (static scene should expand)."""
    first, second = _kitti_images()
    if method == "OEXP":
        predictor = load_oexp(
            device, size=KITTI_SIZE[0], height=KITTI_SIZE[1], md=(8, 4, 4, 4, 4), fac=2.0
        )
    else:
        predictor = load(method, device)
    output = predictor.run(first, second)
    if "rho" not in output:
        return {"skipped": "flow-only method; protocol checks OEXP and SFPP only"}
    log_depth_ratio = output["rho"]  # forward order: log(Z2/Z1), image 2 later
    finite = log_depth_ratio[np.isfinite(log_depth_ratio)]
    np.save(out / f"{method}_KITTI_LOGMID.npy", log_depth_ratio.astype(np.float32))
    return {
        "input_size": list(KITTI_SIZE),
        "log_z2_over_z1_median": float(np.median(finite)),
        "fraction_negative": float(np.mean(finite < 0)),
        "quantiles_5_50_95": np.quantile(finite, [0.05, 0.5, 0.95]).tolist(),
        "expected": "median < 0 (camera moving forward)",
        "passed": bool(np.median(finite) < 0),
    }


def main() -> None:
    """Run both checks for one method and write ``<METHOD>_SANITY.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("OEXP", "SFPP", "RAFT_AFF"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    with np.load(args.manifest, allow_pickle=False) as archive:
        manifest = {k: archive[k] for k in archive.files}
    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "phase_literature_baselines_sanity_v1",
        "method": args.method,
        "synthetic_zoom": synthetic_zoom(args.method, manifest, device),
        "kitti_sample": kitti_sample(args.method, device, args.out),
    }
    (args.out / f"{args.method}_SANITY.json").write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(result), indent=1))


if __name__ == "__main__":
    main()
