"""Extract plain tensor weights from the pinned official MonoDGP KITTI checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy._core.multiarray import scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402

REVISION = "de9f0ad875450f41f9d74edb4fabff8dc911960b"
CHECKPOINT_SHA256 = "1d5f30b34b8bef49638079a8b07f05ebf11bb5f85d6a9a11c7b028c69396f05d"
README_SHA256 = "0ffcad1bae20a8d2e5002b3ebf1967d52247aba20348bd82d9f0910d51402bef"
LOG_SHA256 = "dc223016f4a630fe76cdd6cf2636a970a10d10dbd27e593729d69b22e8b46b5a"


def main() -> None:
    """Keep weights_only enabled; permit only the observed NumPy scalar metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.download_root
    checkpoint = root / "train_val/Mod_AP40_22_71/checkpoint_best.pth"
    bound = {
        checkpoint: CHECKPOINT_SHA256,
        root / "README.md": README_SHA256,
        root / "train_val/Mod_AP40_22_71/22_7108.log": LOG_SHA256,
    }
    for path, expected in bound.items():
        if digest(path) != expected:
            raise ValueError(f"Downloaded official source differs: {path.name}")
    metadata = root / (
        ".cache/huggingface/download/train_val/Mod_AP40_22_71/checkpoint_best.pth.metadata"
    )
    if metadata.read_text().splitlines()[:2] != [REVISION, CHECKPOINT_SHA256]:
        raise ValueError("HF revision/LFS etag receipt differs")
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint))
    if unsafe != {"numpy.core.multiarray.scalar", "numpy.dtype"}:
        raise ValueError(f"Unexpected serialized globals: {sorted(unsafe)}")
    allowed: list[Any] = [
        (scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype(np.float64)),
    ]
    torch.set_num_threads(2)
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(payload) != {"epoch", "model_state", "optimizer_state", "best_result", "best_epoch"}:
        raise ValueError("Unexpected checkpoint structure")
    state = payload["model_state"]
    if not isinstance(state, dict) or len(state) != 701:
        raise ValueError("Unexpected model dictionary")
    for key, value in state.items():
        if not isinstance(key, str) or type(value) is not torch.Tensor:
            raise ValueError("Plain tensor state required")
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite pretrained tensor: {key}")
    args.output.mkdir(parents=True, exist_ok=False)
    output_path = args.output / "model_state.pt"
    temporary = args.output / ".model_state.tmp"
    torch.save(state, temporary)
    os.replace(temporary, output_path)
    if torch.serialization.get_unsafe_globals_in_checkpoint(output_path):
        raise ValueError("Converted weights still contain nonstandard globals")
    restored = torch.load(output_path, map_location="cpu", weights_only=True)
    if state.keys() != restored.keys() or not all(
        torch.equal(value, restored[key]) for key, value in state.items()
    ):
        raise ValueError("Tensor round trip differs")
    receipt = {
        "schema": "phase_monodgp_pretrained_conversion_v1",
        "status": "PASS",
        "repo": "FanqiPu/MonoDGP",
        "revision": REVISION,
        "license_card": "mit",
        "upstream_checkpoint_sha256": CHECKPOINT_SHA256,
        "readme_sha256": README_SHA256,
        "training_log_sha256": LOG_SHA256,
        "model_state_sha256": digest(output_path),
        "tensor_count": len(state),
        "tensor_elements": sum(v.numel() for v in state.values()),
        "tensor_shapes": {key: list(value.shape) for key, value in state.items()},
        "epoch": int(payload["epoch"]),
        "best_epoch": int(payload["best_epoch"]),
        "reported_upstream_kitti_car_AP_R40_percent": float(payload["best_result"]),
        "load_policy": "weights_only=True; observed NumPy scalar/dtype/Float64DType allowlist",
        "round_trip_exact": True,
        "optimizer_transferred": False,
        "training_admission": "PENDING_SEMANTIC_TRANSFER_AND_GPU_SMOKE",
        "limitations": [
            "Published source/config support Car-only training despite a three-class output shape.",
            "Original training config and code hashes are absent from the upstream checkpoint.",
            "Published KITTI AP is not an eAP result or a prediction of transfer performance.",
        ],
        "documentation": "https://docs.pytorch.org/docs/2.11/notes/serialization.html",
        "source_sha256": digest(Path(__file__)),
    }
    write_json(args.output / "CONVERSION.json", receipt)
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in ("status", "tensor_count", "model_state_sha256", "round_trip_exact")
            }
        )
    )


if __name__ == "__main__":
    main()
