"""One-step, no-precision smoke of released Garl branches on fold-separated real inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .contracts import load_frozen_protocol

VARIANTS = ("visual_only", "rgbe_late_fusion")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_probe_records(
    records: Sequence[Mapping[str, Any]], outer_sequences: set[str]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split smoke rows without allowing an outer sequence into optimizer input."""
    train = [row for row in records if str(row["sequence_id"]) not in outer_sequences]
    outer = [row for row in records if str(row["sequence_id"]) in outer_sequences]
    if not train or not outer:
        raise ValueError("smoke shard must contain both train and outer sequence rows")
    if {str(row["sequence_id"]) for row in train} & outer_sequences:
        raise ValueError("outer sequence entered smoke optimizer rows")
    return train, outer


def _tensor(records: Sequence[Mapping[str, Any]], key: str) -> torch.Tensor:
    return torch.as_tensor(np.stack([np.asarray(row[key]) for row in records]), dtype=torch.float32)


def _input(records: Sequence[Mapping[str, Any]], variant: str) -> torch.Tensor:
    rgb = _tensor(records, "garl_rgb_pair").reshape(len(records), 6, 128, 128)
    if variant == "visual_only":
        return rgb
    event = _tensor(records, "garl_event_roi")
    return torch.cat((rgb, event), dim=1)


def _config(path: Path, output_dir: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["dataset"]["root"] = "unused_by_tensor_smoke"
    payload["dirs"]["output"] = output_dir.as_posix()
    for key in ("pretrained_ckpt_rgb", "pretrained_ckpt_event"):
        payload["model"].pop(key, None)
    payload["training_settings"]["total_epochs"] = 1
    payload["training_settings"]["batch_size"] = 2
    payload["training_settings"]["num_threads"] = 0
    return payload


def run_variant(
    *,
    release_root: Path,
    variant: str,
    config_path: Path,
    train_records: Sequence[Mapping[str, Any]],
    outer_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Perform one real forward/backward/update and an unscored outer forward."""
    sys.path.insert(0, str(release_root))
    from garl_ttc.models import TTCNetwork

    torch.manual_seed(7)
    config = _config(config_path, output_dir)
    model = TTCNetwork(config, is_train=True).float().cpu()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    batch = list(train_records[:2])
    data = _input(batch, variant)
    target = _tensor(batch, "ttc_s").reshape(-1)
    heights = _tensor(batch, "garl_visible_heights_px")
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, _, losses, _ = model.forward_train(
        data,
        target,
        visible_height_target=heights,
        mask_target=None,
        use_mask_supervison=[False] * len(batch),
        epoch_idx=1,
    )
    total = torch.stack(list(losses.values())).sum()
    if not torch.isfinite(total):
        raise FloatingPointError(f"{variant} smoke loss is non-finite")
    total.backward()
    gradient_tensors = sum(
        int(parameter.grad is not None and torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    if gradient_tensors == 0:
        raise RuntimeError(f"{variant} smoke produced no finite gradients")
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        raw = model(_input(list(outer_records[:2]), variant))[0]
    if not torch.isfinite(raw).all():
        raise FloatingPointError(f"{variant} outer forward is non-finite")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{variant}_one_step.pt"
    torch.save(model.state_dict(), checkpoint)
    outer_output = output_dir / f"{variant}_outer_forward.npz"
    np.savez_compressed(
        outer_output,
        sample_token=np.asarray([row["sample_token"] for row in outer_records[:2]]),
        raw_output=raw.detach().cpu().numpy(),
    )
    return {
        "variant": variant,
        "status": "completed_one_step_no_precision",
        "train_rows_used": len(batch),
        "outer_rows_forwarded": min(2, len(outer_records)),
        "train_sequences": sorted({str(row["sequence_id"]) for row in batch}),
        "outer_sequences": sorted({str(row["sequence_id"]) for row in outer_records[:2]}),
        "input_shape": list(data.shape),
        "loss": float(total.detach()),
        "finite_gradient_tensors": gradient_tensors,
        "checkpoint_path": checkpoint.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "outer_output_path": outer_output.as_posix(),
        "outer_output_sha256": _sha256(outer_output),
        "config_path": config_path.as_posix(),
        "config_sha256": _sha256(config_path),
        "wall_seconds": time.perf_counter() - started,
        "device": "cpu",
        "precision_metric_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--visual-config", type=Path, required=True)
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    args = parser.parse_args()
    protocol = load_frozen_protocol(args.protocol, args.freeze_receipt)
    outer_sequences = set(protocol["outer_design"][f"fold_{args.outer_fold}"])
    with gzip.open(args.shard, "rb") as stream:
        records = torch.load(stream, map_location="cpu", weights_only=False)
    train, outer = split_probe_records(records, outer_sequences)
    config_by_variant = {
        "visual_only": args.visual_config,
        "rgbe_late_fusion": args.full_config,
    }
    results = [
        run_variant(
            release_root=args.release_root,
            variant=variant,
            config_path=config_by_variant[variant],
            train_records=train,
            outer_records=outer,
            output_dir=args.output_dir,
        )
        for variant in VARIANTS
    ]
    phase_checkpoint = args.output_dir.parent / "fold0/box_head_seed7.state.pt"
    receipt = {
        "artifact_type": "phase_ttc_matched_outer_architecture_smoke_v1",
        "status": "completed_partial_architecture_smoke",
        "protocol_sha256": protocol["protocol_sha256"],
        "outer_fold": args.outer_fold,
        "source_shard_sha256": _sha256(args.shard),
        "garl": results,
        "phase": {
            "box_head_forward_backward_checkpoint": phase_checkpoint.is_file(),
            "box_head_checkpoint_sha256": (
                _sha256(phase_checkpoint) if phase_checkpoint.is_file() else None
            ),
            "rgb_producer_forward_backward": False,
            "reason": (
                "no fold-aware PHASE RGB producer trainer is integrated; "
                "precomputed PHASE20 is second-order OOF only"
            ),
        },
        "precision_metric_computed": False,
        "gpu_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "SMOKE_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": receipt["status"], "variants": len(results)}))


if __name__ == "__main__":
    main()


__all__ = ["run_variant", "split_probe_records"]
