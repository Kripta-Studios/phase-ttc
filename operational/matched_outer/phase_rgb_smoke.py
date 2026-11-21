"""Fold-aware one-update smoke for the historical PHASE R_A5 RGB producer."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.losses.causal_scale_ttc import (
    CausalScaleTTCLossConfig,
    causal_scale_ttc_loss,
)
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from operational.independent_refit.fitter import state_tensor_sha256

from .contracts import load_frozen_protocol


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_fold_records(
    records: Sequence[Mapping[str, Any]], outer_sequences: set[str]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Separate optimizer and outer-forward rows before any tensor is built."""
    train = [row for row in records if str(row["sequence_id"]) not in outer_sequences]
    outer = [row for row in records if str(row["sequence_id"]) in outer_sequences]
    if not train or not outer:
        raise ValueError("RGB probe must contain both train and outer sequence rows")
    if {str(row["sequence_id"]) for row in train} & outer_sequences:
        raise ValueError("outer sequence entered PHASE RGB optimizer rows")
    return train, outer


def _historical_recipe(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = json.loads(config_path.read_text(encoding="utf-8"))
    if document.get("schema") != "rgb_port_producers_v1":
        raise ValueError("historical RGB recipe schema differs")
    recipe = document["producers"]["R_A5"]
    model = recipe["model_config"]
    training = recipe["training_config"]
    if (
        model.get("modality") != "rgb"
        or model.get("in_channels") != 3
        or model.get("transport_mode") != "legacy"
        or training.get("seed") != 7
        or training.get("learning_rate") != 3e-4
        or training.get("weight_decay") != 1e-4
    ):
        raise ValueError("historical R_A5 recipe differs")
    return recipe, document


def _tensor(records: Sequence[Mapping[str, Any]], key: str) -> torch.Tensor:
    return torch.as_tensor(np.stack([np.asarray(row[key]) for row in records])).float()


def run_phase_rgb_smoke(
    *,
    records: Sequence[Mapping[str, Any]],
    outer_sequences: set[str],
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one CPU update and one label-free outer forward, without scoring."""
    train, outer = select_fold_records(records, outer_sequences)
    recipe, _ = _historical_recipe(config_path)
    torch.manual_seed(int(recipe["training_config"]["seed"]))
    torch.set_num_threads(1)
    model = CausalScaleTTC(CausalScaleTTCConfig(**recipe["model_config"])).float().cpu()
    initial_state_sha256 = state_tensor_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe["training_config"]["learning_rate"]),
        weight_decay=float(recipe["training_config"]["weight_decay"]),
    )

    batch = list(train[:2])
    rgb = _tensor(batch, "garl_rgb_pair")
    delta = _tensor(batch, "garl_delta_t_s").reshape(-1, 1)
    target = _tensor(batch, "ttc_s").reshape(-1)
    if rgb.ndim != 5 or tuple(rgb.shape[1:3]) != (2, 3):
        raise ValueError("probe RGB must be genuine two-endpoint [B,2,3,H,W]")
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(rgb, delta, return_dense_features=False)
    result = causal_scale_ttc_loss(
        output,
        target_ttc_seconds=target,
        delta_t_s=delta,
        risk_thresholds_s=model.config.risk_thresholds_s,
        target_valid=torch.isfinite(target) & (target != 0),
        config=CausalScaleTTCLossConfig(**recipe["loss_config"]),
    )
    if not bool(torch.isfinite(result.total)):
        raise FloatingPointError("PHASE R_A5 smoke loss is non-finite")
    result.total.backward()
    finite_gradient_tensors = sum(
        int(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()))
        for parameter in model.parameters()
    )
    if finite_gradient_tensors == 0:
        raise RuntimeError("PHASE R_A5 smoke produced no finite gradients")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    outer_batch = list(outer[:2])
    outer_rgb = _tensor(outer_batch, "garl_rgb_pair")
    outer_delta = _tensor(outer_batch, "garl_delta_t_s").reshape(-1, 1)
    model.eval()
    with torch.inference_mode():
        outer_output = model(outer_rgb, outer_delta, return_dense_features=False)
    if not bool(torch.isfinite(outer_output.ttc_mean_seconds).all()):
        raise FloatingPointError("PHASE R_A5 outer forward is non-finite")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "phase_r_a5_one_step.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "initial_state_sha256": initial_state_sha256,
            "train_sample_tokens": [str(row["sample_token"]) for row in batch],
        },
        checkpoint,
    )
    forward_path = output_dir / "phase_r_a5_outer_forward.npz"
    np.savez_compressed(
        forward_path,
        sample_token=np.asarray([str(row["sample_token"]) for row in outer_batch]),
        raw_ttc_output=outer_output.ttc_mean_seconds.detach().cpu().numpy(),
    )
    return {
        "schema": "phase_ttc_matched_outer_r_a5_smoke_v1",
        "status": "completed_one_update_and_unscored_outer_forward",
        "historical_architecture": "e_jepa_ttc.models.causal_scale_ttc.CausalScaleTTC",
        "historical_fit": "R_A5",
        "device": "cpu",
        "input_shape": list(rgb.shape),
        "train_rows_used": len(batch),
        "train_sequences": sorted({str(row["sequence_id"]) for row in batch}),
        "outer_rows_forwarded": len(outer_batch),
        "outer_sequences": sorted({str(row["sequence_id"]) for row in outer_batch}),
        "finite_gradient_tensors": finite_gradient_tensors,
        "loss": float(result.total.detach()),
        "loss_support_counts": result.counts,
        "checkpoint_path": checkpoint.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "model_state_sha256": state_tensor_sha256(model.state_dict()),
        "initial_state_sha256": initial_state_sha256,
        "outer_output_path": forward_path.as_posix(),
        "outer_output_sha256": _sha256(forward_path),
        "recipe_path": config_path.as_posix(),
        "recipe_sha256": _sha256(config_path),
        "wall_seconds": time.perf_counter() - started,
        "precision_metric_computed": False,
        "outer_labels_read_by_forward": False,
        "scope": {
            "proven": "real RGB forward/backward/update/checkpoint and fold isolation",
            "not_proven": "complete R_A5 fit or matched-outer OOF predictions",
            "omitted_from_historical_objective": [
                "bbox_geometry (probe shard stores visible heights but not full boxes)",
                "DINOv3 relational target (not stored in bounded probe shard)",
            ],
            "historical_full_runner_block": (
                "run_e_a5 is event-only and hard-gated to CUDA, 35048 rows, 18 epochs; "
                "the R_A5 raw source additionally requires frozen P/H/V manifests, raw tar "
                "members and a nominal-teacher cache. This smoke reuses its exact R_A5 model, "
                "AdamW settings and causal-scale loss on the available target terms only."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    args = parser.parse_args()
    protocol = load_frozen_protocol(args.protocol, args.freeze_receipt)
    outer_sequences = set(protocol["outer_design"][f"fold_{args.outer_fold}"])
    with gzip.open(args.shard, "rb") as stream:
        records = torch.load(stream, map_location="cpu", weights_only=False)
    receipt = run_phase_rgb_smoke(
        records=records,
        outer_sequences=outer_sequences,
        config_path=args.recipe,
        output_dir=args.output_dir,
    )
    receipt["protocol_sha256"] = protocol["protocol_sha256"]
    receipt["outer_fold"] = args.outer_fold
    receipt["source_shard_sha256"] = _sha256(args.shard)
    (args.output_dir / "PHASE_RGB_SMOKE_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": receipt["status"], "loss": receipt["loss"]}))


if __name__ == "__main__":
    main()


__all__ = ["run_phase_rgb_smoke", "select_fold_records"]
