#!/usr/bin/env python
"""Build the three-step common-coordinate Object Event TTC v4 cache."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.artifacts.risk_geometry_v10 import binding, verify  # noqa: E402
from e_jepa_ttc.data.capacity_sources import _signed  # noqa: E402
from e_jepa_ttc.data.garlttc_lhr_cache import (  # noqa: E402
    GarlTTCLHRCacheConfig,
    materialize_garlttc_lhr_cache,
)
from scripts.materialize_dinov3_relational_teacher import _capacity_teacher_scope  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--garlttc-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("screen", "full"), default="screen")
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--margin-fraction", type=float, default=0.25)
    parser.add_argument("--bins-per-polarity", type=int, default=5)
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Materialize only public train rows; required by the V7 T20 ablation.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--capacity-preparation", type=Path)
    parser.add_argument("--capacity-preparation-sha256")
    args = parser.parse_args()

    maximum = args.max_samples_per_split
    exact_tokens = None
    if (args.capacity_preparation is None) != (args.capacity_preparation_sha256 is None):
        raise ValueError("Capacity cache requires preparation path and exact SHA256")
    if args.capacity_preparation is not None:
        if not args.train_only or maximum is not None:
            raise ValueError("Capacity cache requires --train-only without a new sample cap")
        preparation_source = binding(args.capacity_preparation)
        if preparation_source["sha256"] != args.capacity_preparation_sha256:
            raise ValueError("Capacity preparation hash changed")
        preparation = _signed(preparation_source)
        partitions = _signed(preparation["source_partitions"])
        metadata = verify(partitions["outputs"]["train_metadata"])
        _, sequences, _ = _capacity_teacher_scope(
            args.capacity_preparation, args.capacity_preparation_sha256, metadata
        )
        split = json.loads(args.split.read_text(encoding="utf-8"))
        if set(split.get("assignments", {}).get("train", [])) != sequences:
            raise ValueError("Capacity cache TRAIN split differs from the registered cohort")
        exact_tokens = tuple(_signed(preparation["cohort"])["ordered_tokens"])
    if maximum is None and args.profile == "screen" and exact_tokens is None:
        maximum = 2048
    if args.bins_per_polarity == 10:
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(args.output_dir.parent).free
        minimum_free_bytes = 25 * 1024**3
        if free_bytes < minimum_free_bytes:
            raise RuntimeError(
                "T20 materialization requires at least 25 GiB free before writing; "
                f"found {free_bytes / 1024**3:.2f} GiB"
            )
    config = GarlTTCLHRCacheConfig(
        roi_size=args.roi_size,
        shard_size=args.shard_size,
        store_full_frame_events=False,
        store_garl_event_roi=False,
        store_jepa_event_roi=False,
        store_event_v4_common_roi=True,
        event_v4_margin_fraction=args.margin_fraction,
        event_v4_require_precontext=True,
        event_v4_precontext_fallback="shifted_event_window",
        event_v4_bins_per_polarity=args.bins_per_polarity,
        event_v4_storage_dtype=args.storage_dtype,
        materialize_splits=(("train",) if args.train_only else ("train", "validation")),
        include_rgb=False,
        include_masks=False,
        workers=args.workers,
        preprocessing_device="cpu",
        compression="none",
    )
    result = materialize_garlttc_lhr_cache(
        eap_root=args.eap_root.resolve(),
        garlttc_root=args.garlttc_root.resolve(),
        split_path=args.split.resolve(),
        output_dir=args.output_dir.resolve(),
        config=config,
        max_samples_per_split=maximum,
        resume=args.resume,
        exact_sample_tokens=exact_tokens,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "profile": args.profile,
                "config": asdict(config),
                "manifest": (args.output_dir.resolve() / "manifest.json").as_posix(),
                "split_counts": result.get("split_counts"),
                "precontext_motion_valid_fraction": result.get(
                    "precontext_motion_valid_fraction"
                ),
                "event_v4_precontext_valid_fraction": result.get(
                    "event_v4_precontext_valid_fraction"
                ),
                "event_v4_precontext_source_counts": result.get(
                    "event_v4_precontext_source_counts"
                ),
                "object_lhr_extension": result.get("object_lhr_extension"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
