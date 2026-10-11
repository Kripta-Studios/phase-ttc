"""Replay corrected temporal TTC uncertainty from frozen pair outputs.

Example::

    python -m operational.uncertainty_revision.replay \
      --input pair_outputs.npz --output revised_uncertainty.npz \
      --current-pair-weight 0.75 --covariance conservative_upper_bound

The input archive must contain ``pair_log_ratio``, ``pair_log_variance``,
``pair_known``, and ``pair_delta_t_s``, each shaped ``[batch, pairs]``.
This command never loads or modifies a checkpoint and refuses to overwrite an
existing result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.models.temporal_uncertainty import propagate_temporal_uncertainty


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--current-pair-weight", type=float, required=True)
    parser.add_argument(
        "--covariance",
        choices=("independent", "conservative_upper_bound"),
        default="conservative_upper_bound",
    )
    parser.add_argument("--ttc-clip-seconds", type=float, default=60.0)
    parser.add_argument("--risk-thresholds-s", type=float, nargs="+", default=(0.5, 1, 2, 4))
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output}")
    with np.load(args.input, allow_pickle=False) as archive:
        required = {
            "pair_log_ratio",
            "pair_log_variance",
            "pair_known",
            "pair_delta_t_s",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"input archive is missing keys: {sorted(missing)}")
        ratio_covariance = (
            torch.from_numpy(archive["ratio_covariance"])
            if "ratio_covariance" in archive.files
            else None
        )
        revision = propagate_temporal_uncertainty(
            torch.from_numpy(archive["pair_log_ratio"]),
            torch.from_numpy(archive["pair_log_variance"]),
            torch.from_numpy(archive["pair_known"]).bool(),
            torch.from_numpy(archive["pair_delta_t_s"]),
            current_pair_weight=args.current_pair_weight,
            risk_thresholds_s=tuple(args.risk_thresholds_s),
            ttc_clip_seconds=args.ttc_clip_seconds,
            covariance_assumption=args.covariance,
            ratio_covariance=ratio_covariance,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        inverse_ttc_mean=revision.inverse_ttc_mean.detach().cpu().numpy(),
        inverse_ttc_log_variance=revision.inverse_ttc_log_variance.detach().cpu().numpy(),
        ttc_mean_seconds=revision.ttc_mean_seconds.detach().cpu().numpy(),
        ttc_log_variance=revision.ttc_log_variance.detach().cpu().numpy(),
        collision_logits=revision.collision_logits.detach().cpu().numpy(),
        known_mask=revision.known_mask.detach().cpu().numpy(),
        temporal_blend_used=revision.temporal_blend_used.detach().cpu().numpy(),
        uncertainty_valid_mask=revision.uncertainty_valid_mask.detach().cpu().numpy(),
        ttc_uncertainty_valid_mask=(revision.ttc_uncertainty_valid_mask.detach().cpu().numpy()),
        ratio_saturated_mask=revision.ratio_saturated_mask.detach().cpu().numpy(),
    )
    provenance = {
        "schema": revision.version,
        "inference_only": revision.inference_only,
        "covariance_assumption": revision.covariance_assumption,
        "current_pair_weight": args.current_pair_weight,
        "compatible_with_legacy_calibrated_heads": (
            revision.compatible_with_legacy_calibrated_heads
        ),
        "input": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
