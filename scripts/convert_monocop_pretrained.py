"""Extract tensor-only weights from the pinned official three-class MonoCoP model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from numpy._core.multiarray import scalar

REVISION = "91e1119b4463a83a9f64b7c2838be4bc5314834f"
CHECKPOINT_SHA256 = "f574a65bfd22ebcbb80b77fd18a09b4812d89ae5cdc21e6a556a081b7b755321"
SOURCE_REVISION = "7d3932444f7ea94be564ae61a0b334fc6673213b"
SOURCE_RECEIPT_SHA256 = "90f69f32b8e2f63ea56cc168afdb1fe571f106ac1c8f54dd1b43b3e3a8a342c8"
LOG_SHA256 = "85d7060452afec56f78eb7a20f6c28d6faa5f104c1b3fa2b4257b4ffb662e311"


def digest(path: Path) -> str:
    """Read a SHA-256 without holding a checkpoint's bytes in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Validate provenance and retain weights_only=True with observed scalar metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--source-intake", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.download_root / "MonoCoP_All.pth"
    receipt_path = args.source_intake / "SOURCE_RECEIPT.json"
    if (
        digest(checkpoint) != CHECKPOINT_SHA256
        or digest(receipt_path) != SOURCE_RECEIPT_SHA256
        or digest(args.training_log) != LOG_SHA256
    ):
        raise ValueError("Pinned checkpoint, source receipt or author log differs")
    metadata = args.download_root / ".cache/huggingface/download/MonoCoP_All.pth.metadata"
    if metadata.read_text().splitlines()[:2] != [REVISION, CHECKPOINT_SHA256]:
        raise ValueError("Hugging Face revision or LFS etag differs")
    source = json.loads(receipt_path.read_text(encoding="utf-8"))
    semantic_files = ("config/monocop.yaml", "lib/datasets/kitti/kitti_dataset.py")
    for name in semantic_files:
        if digest(args.source_intake / "source" / name) != source["files"][name]["sha256"]:
            raise ValueError(f"Class/size source differs: {name}")
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint))
    if unsafe != {"numpy.dtype", "numpy._core.multiarray.scalar"}:
        raise ValueError(f"Unexpected checkpoint globals: {sorted(unsafe)}")
    torch.set_num_threads(2)
    with torch.serialization.safe_globals([scalar, np.dtype, type(np.dtype(np.float64))]):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(payload) != {"epoch", "model_state", "optimizer_state", "best_result", "best_epoch"}:
        raise ValueError("Unknown upstream checkpoint structure")
    state = payload["model_state"]
    if not isinstance(state, dict) or len(state) != 694:
        raise ValueError("Unexpected tensor dictionary")
    for name, value in state.items():
        if not isinstance(name, str) or type(value) is not torch.Tensor:
            raise ValueError("Checkpoint must contain plain named tensors")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
    epoch = int(payload["epoch"])
    best_epoch = int(payload["best_epoch"])
    best_result = float(payload["best_result"])
    if f"Best Result:{best_result}, epoch:{best_epoch}" not in args.training_log.read_text():
        raise ValueError("Checkpoint metadata does not match the pinned author log")
    args.output.mkdir(parents=True, exist_ok=False)
    temporary = args.output / ".model_state.tmp"
    pure = args.output / "model_state.pt"
    torch.save(state, temporary)
    os.replace(temporary, pure)
    if torch.serialization.get_unsafe_globals_in_checkpoint(pure):
        raise ValueError("Converted weights still require extra globals")
    restored = torch.load(pure, map_location="cpu", weights_only=True)
    if state.keys() != restored.keys() or not all(
        torch.equal(value, restored[name]) for name, value in state.items()
    ):
        raise ValueError("Tensor round trip differs")
    if torch.cuda.is_initialized():
        raise ValueError("This conversion must not initialize CUDA")
    shutil.copyfile(__file__, args.output / "executed_converter.py")
    result = {
        "schema": "phase_monocop_pretrained_conversion_v1",
        "status": "PASS_TENSOR_CONVERSION_ONLY",
        "repo": "zhihao406/MonoCoP",
        "revision": REVISION,
        "filename": checkpoint.name,
        "upstream_checkpoint_sha256": CHECKPOINT_SHA256,
        "upstream_checkpoint_bytes": checkpoint.stat().st_size,
        "hf_metadata_sha256": digest(metadata),
        "source_revision": SOURCE_REVISION,
        "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
        "semantic_source_files": {name: source["files"][name]["sha256"] for name in semantic_files},
        "training_log_sha256": LOG_SHA256,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "author_reported_kitti_car_moderate_3d_AP_R40_percent": best_result,
        "author_metric_is_our_eap_metric": False,
        "tensor_count": len(state),
        "tensor_elements_with_state_aliases": sum(v.numel() for v in state.values()),
        "tensor_shapes": {name: list(value.shape) for name, value in state.items()},
        "model_state_sha256": digest(pure),
        "round_trip_exact": True,
        "optimizer_transferred": False,
        "cuda_initialized": False,
        "load_policy": "weights_only=True; observed NumPy scalar/dtype/Float64DType allowlist",
        "published_class_order": ["Pedestrian", "Car", "Cyclist"],
        "planned_eap_mapping": {
            "Pedestrian_0": "pedestrian_3",
            "Car_1": "car_0",
            "Cyclist_2": "bike_1",
        },
        "limits": [
            "Published config and log support all three classes; "
            "checkpoint omits its original config/code hash.",
            "Author best-result metadata matches the log; it is a KITTI metric, not an eAP score.",
            "Cyclist versus eAP bike and source versus target box conventions "
            "need explicit transfer review.",
            "Source MIT and a model-card badge do not certify weight license "
            "or external-data benchmark admission.",
            "CPU conversion does not admit another GPU experiment or establish detection quality.",
        ],
        "source_sha256": digest(Path(__file__)),
        "serialization_documentation": "https://docs.pytorch.org/docs/2.11/notes/serialization.html",
    }
    (args.output / "CONVERSION.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({key: result[key] for key in ("status", "model_state_sha256", "tensor_count")})
    )


if __name__ == "__main__":
    main()
