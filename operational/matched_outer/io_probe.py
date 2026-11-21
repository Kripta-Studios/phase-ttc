"""Measure one matched outer-fold shard from source media and from temporal cache."""

from __future__ import annotations

import argparse
import gzip
import json
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import psutil

from .contracts import load_frozen_protocol

if TYPE_CHECKING:
    from e_jepa_ttc.data.garlttc_lhr_cache import GarlTTCLHRCacheConfig

T = TypeVar("T")


def call_row_materializer(
    materializer: Callable[..., T], row: Mapping[str, object], **kwargs: object
) -> T:
    """Call the current cache API with an explicit disabled mask reader."""
    return materializer(row, mask_reader=None, **kwargs)


def materialize_records_current(
    rows: Sequence[dict[str, object]],
    *,
    eap_root: Path,
    config: GarlTTCLHRCacheConfig,
    first_track_lookup: Mapping[tuple[object, object], object],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sequential current-signature materialization for the bounded CPU probe."""
    from e_jepa_ttc.data.eap import EAPEventReader
    from e_jepa_ttc.data.garlttc_lhr_cache import (
        CalibrationResolver,
        _materialize_row,
        _RGBTarReader,
    )

    event_readers: dict[Path, EAPEventReader] = {}
    calibration = CalibrationResolver(config.calibration_mode, eap_root=eap_root)
    rgb_reader = _RGBTarReader(eap_root) if config.include_rgb else None
    records: list[dict[str, Any]] = []
    errors: Counter[str] = Counter()
    for row in rows:
        fallback = row["timestamp_us"]
        first_timestamp = first_track_lookup.get((row["sequence_id"], row["track_id"]), fallback)
        try:
            sample = call_row_materializer(
                _materialize_row,
                row,
                eap_root=eap_root,
                config=config,
                event_readers=event_readers,
                rgb_reader=rgb_reader,
                calibration=calibration,
                first_track_timestamp_us=int(first_timestamp),
            )
        except Exception as error:
            errors[f"{type(error).__name__}:{str(error)[:120]}"] += 1
        else:
            records.append(sample)
    return records, dict(sorted(errors.items()))


def measure_operation(
    operation: Callable[[], T], *, bytes_path: Path | None = None
) -> tuple[T, dict[str, Any]]:
    """Measure wall time and process peak RSS while one bounded operation runs."""
    process = psutil.Process()
    before = process.memory_info().rss
    peak = [before]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.005):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        result = operation()
    finally:
        wall = time.perf_counter() - started
        stop.set()
        thread.join()
        peak[0] = max(peak[0], process.memory_info().rss)
    return result, {
        "wall_seconds": wall,
        "rss_before_bytes": before,
        "peak_rss_bytes": peak[0],
        "peak_rss_delta_bytes": max(peak[0] - before, 0),
        "bytes": bytes_path.stat().st_size if bytes_path is not None and bytes_path.exists() else 0,
        "peak_vram_bytes": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--garlttc-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--rows-per-role", type=int, default=8)
    args = parser.parse_args()
    if args.rows_per_role <= 0:
        parser.error("--rows-per-role must be positive")
    protocol = load_frozen_protocol(args.protocol, args.freeze_receipt)
    design = protocol["outer_design"]
    outer = set(map(str, design[f"fold_{args.outer_fold}"]))
    all_sequences = {
        str(sequence) for fold in range(int(design["folds"])) for sequence in design[f"fold_{fold}"]
    }
    training = all_sequences - outer

    from e_jepa_ttc.data.garlttc_eap import load_garlttc_train_index
    from e_jepa_ttc.data.garlttc_lhr_cache import GarlTTCLHRCacheConfig, _atomic_torch_save

    index = load_garlttc_train_index(args.garlttc_root, sorted(all_sequences))
    rows = index.merged.sort_values(
        ["sequence_id", "timestamp_us", "track_id", "sample_token"], kind="mergesort"
    )
    selected = []
    for role, sequences in (("train", training), ("outer", outer)):
        part = rows[rows["sequence_id"].astype(str).isin(sequences)].head(args.rows_per_role)
        for record in part.to_dict(orient="records"):
            record["probe_role"] = role
            selected.append(record)
    first_track = (
        index.merged.groupby(["sequence_id", "track_id"], dropna=False)["timestamp_us"]
        .min()
        .to_dict()
    )
    config = GarlTTCLHRCacheConfig(
        store_full_frame_events=False,
        store_garl_event_roi=True,
        store_jepa_event_roi=False,
        store_event_v4_common_roi=False,
        include_rgb=True,
        include_masks=False,
        workers=1,
        compression="gzip",
        compression_level=1,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = args.output_dir / "outer0_media_probe.pt.gz"

    def materialize_and_write() -> list[dict[str, Any]]:
        records, errors = materialize_records_current(
            selected,
            eap_root=args.eap_root,
            config=config,
            first_track_lookup=first_track,
        )
        if errors or len(records) != len(selected):
            raise RuntimeError(
                f"media probe discarded rows: requested={len(selected)} "
                f"materialized={len(records)} errors={errors}"
            )
        _atomic_torch_save(records, shard_path, compression="gzip", compression_level=1)
        return records

    records, media = measure_operation(materialize_and_write, bytes_path=shard_path)
    record_count = len(records)
    del records

    def load_cache() -> int:
        import torch

        with gzip.open(shard_path, "rb") as stream:
            loaded = torch.load(stream, map_location="cpu", weights_only=False)
        return len(loaded)

    loaded_count, cache = measure_operation(load_cache, bytes_path=shard_path)
    if loaded_count != record_count:
        raise RuntimeError("cache read changed the materialized row count")
    media["rows"] = record_count
    media["rows_per_second"] = record_count / media["wall_seconds"]
    media["bytes_written"] = media.pop("bytes")
    cache["rows"] = loaded_count
    cache["rows_per_second"] = loaded_count / cache["wall_seconds"]
    cache["bytes_read"] = cache.pop("bytes")
    full_rows = int(len(index.merged))
    result = {
        "artifact_type": "phase_ttc_matched_outer_io_probe_v1",
        "status": "completed_bounded_probe",
        "protocol_sha256": protocol["protocol_sha256"],
        "outer_fold": args.outer_fold,
        "rows_per_role": args.rows_per_role,
        "roles": {"train": args.rows_per_role, "outer": args.rows_per_role},
        "cpu_only": True,
        "media_to_cache": media,
        "cache_read": cache,
        "projection": {
            "source_rows": full_rows,
            "full_cache_bytes_linear": int(media["bytes_written"] / record_count * full_rows),
            "full_media_build_seconds_linear": media["wall_seconds"] / record_count * full_rows,
            "full_cache_read_seconds_linear": cache["wall_seconds"] / record_count * full_rows,
            "caveat": "linear projection from one bounded 16-row shard; not training time",
        },
        "resume": {
            "shard_path": shard_path.as_posix(),
            "shard_bytes": shard_path.stat().st_size,
            "reusable_after_hash": True,
        },
    }
    (args.output_dir / "IO_PROBE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["call_row_materializer", "materialize_records_current", "measure_operation"]
