#!/usr/bin/env python
"""Build the three-step common-coordinate Object Event TTC v4 cache."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner  # noqa: E402
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, verify  # noqa: E402
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
    parser.add_argument("--compression", choices=("none", "gzip"), default="none")
    parser.add_argument("--compression-level", type=int, choices=range(1, 10), default=1)
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
    parser.add_argument("--other-reserved-bytes", type=int)
    args = parser.parse_args()

    maximum = args.max_samples_per_split
    exact_tokens = None
    preparation_source = None
    train_index_paths = None
    expected_source_rows = 88_744
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
        labels_source = partitions["outputs"]["train_labels"]
        expected_labels = Path(preparation["source_partitions"]["path"]).resolve().parent
        if Path(labels_source["path"]).resolve() != expected_labels / "train_labels.parquet":
            raise ValueError("Capacity labels escaped the signed TRAIN partition")
        train_index_paths = (metadata, verify(labels_source))
        import pyarrow.parquet as pq

        expected_source_rows = pq.read_metadata(metadata).num_rows
        if pq.read_metadata(train_index_paths[1]).num_rows != expected_source_rows:
            raise ValueError("Capacity TRAIN metadata and label counts differ")
        split = json.loads(args.split.read_text(encoding="utf-8"))
        if set(split.get("assignments", {}).get("train", [])) != sequences:
            raise ValueError("Capacity cache TRAIN split differs from the registered cohort")
        exact_tokens = tuple(_signed(preparation["cohort"])["ordered_tokens"])
        if (
            args.other_reserved_bytes is None
            or args.other_reserved_bytes < 0
            or args.roi_size != 128
            or args.margin_fraction != 0.25
            or args.bins_per_polarity != 5
            or args.storage_dtype != "float32"
            or args.compression != "gzip"
            or args.compression_level != 1
            or args.workers != 1
            or not 1 <= args.shard_size <= 256
        ):
            raise ValueError(
                "Capacity reservation requires historical FP32 geometry, gzip level 1, "
                "one CPU worker, shards <=256, and explicit nonnegative other-reserved-bytes"
            )
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
        compression=args.compression,
        compression_level=args.compression_level,
        expected_train_rows=expected_source_rows,
    )

    def build(resource_check: Callable[[int], None] | None = None) -> dict[str, Any]:
        return materialize_garlttc_lhr_cache(
            eap_root=args.eap_root.resolve(),
            garlttc_root=args.garlttc_root.resolve(),
            split_path=args.split.resolve(),
            output_dir=args.output_dir.resolve(),
            config=config,
            max_samples_per_split=maximum,
            resume=args.resume,
            exact_sample_tokens=exact_tokens,
            resource_check=resource_check,
            train_index_paths=train_index_paths,
        )

    if exact_tokens is None:
        result = build()
    else:
        output = args.output_dir.resolve()
        allocated = sum(p.stat().st_size for p in (output / "train").glob("shard-*.pt.gz"))
        cap = 40_000_000_000
        total_budget = 45_000_000_000
        if allocated > cap:
            raise RuntimeError("RESOURCE_BLOCKED: existing compressed shards exceed 40 GB cap")
        other = int(args.other_reserved_bytes)
        ownership = output.parent / f"{output.name}_ownership"
        with ResourceOwner(
            ownership,
            remaining_bytes=total_budget - allocated + other,
            planned_ram_bytes=4 * 1024**3,
        ) as owner:
            atomic_json(
                ownership / f"RESOURCE_ADMISSION_{time.time_ns()}.json",
                {
                    "compressed_shards_cap_bytes": cap,
                    "total_incremental_budget_bytes": total_budget,
                    "already_allocated_shard_bytes": allocated,
                    "other_reserved_bytes": other,
                    "planned_ram_bytes": 4 * 1024**3,
                    "minimum_disk_free_bytes": 40_000_000_000,
                    "preparation": preparation_source,
                },
            )

            def check(published: int) -> None:
                if published > cap:
                    raise RuntimeError("RESOURCE_BLOCKED: compressed cache exceeds 40 GB cap")
                owner.estimated_remaining_bytes = total_budget - max(allocated, published) + other
                owner.check()

            result = build(check)
    print(
        json.dumps(
            {
                "status": "completed",
                "profile": args.profile,
                "config": asdict(config),
                "manifest": (args.output_dir.resolve() / "manifest.json").as_posix(),
                "split_counts": result.get("split_counts"),
                "precontext_motion_valid_fraction": result.get("precontext_motion_valid_fraction"),
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
