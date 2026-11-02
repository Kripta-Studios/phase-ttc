#!/usr/bin/env python
"""Compare frozen and single-bincount event accumulation on one real window."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import psutil

from e_jepa_ttc.data.eap import EAPEventReader
from e_jepa_ttc.detection.event_accumulation import (
    encode_causal_event_chunks_experimental,
)
from e_jepa_ttc.detection.event_data import (
    CausalEventConfig,
    encode_causal_event_chunks,
)

DEFAULT_SOURCE_PROBE = Path("output/event_feasibility/event_representation_50ms_qGsgzl4Q8B_v3.json")
DEFAULT_OUTPUT = Path("output/event_feasibility/event_accumulation_profile_qGsgzl4Q8B_v1.json")
MAXIMUM_RSS_BYTES = 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _profile(args: argparse.Namespace) -> dict[str, object]:
    if args.repeats != 3:
        raise ValueError("The preregistered comparison requires exactly three repeats")
    source = _read_json(args.source_probe)
    if source.get("schema") != "eap_causal_event_voxel_probe_v3" or source.get("status") != "PASS":
        raise ValueError("Source probe is not the admitted v3 event representation")
    sample_value = source.get("canonical_sample")
    probe_value = source.get("probe")
    provenance_value = source.get("provenance")
    if not all(isinstance(value, dict) for value in (sample_value, probe_value, provenance_value)):
        raise ValueError("Source probe is missing canonical evidence")
    sample = cast(dict[str, Any], sample_value)
    probe = cast(dict[str, Any], probe_value)
    provenance = cast(dict[str, Any], provenance_value)
    source_receipt_value = probe.get("receipt")
    source_receipt = cast(dict[str, Any], source_receipt_value)
    if not isinstance(source_receipt, dict):
        raise ValueError("Source probe lacks its voxel receipt")
    events_relative = str(sample["events_path"])
    events_path = (args.root.resolve() / events_relative).resolve()
    if not events_path.is_relative_to(args.root.resolve()) or not events_path.is_file():
        raise ValueError("Canonical events file is absent or escapes the dataset root")
    expected_size = int(provenance["event_h5_expected_size_bytes"])
    if events_path.stat().st_size != expected_size:
        raise ValueError("Canonical events file size differs from the source probe")
    cutoff_us = int(sample["rgb_exposure_end_timestamp_us"])
    sequence_id = str(sample["sequence_id"])
    config = CausalEventConfig(**source_receipt["config"])
    expected_event_count = int(source_receipt["event_count"])
    expected_voxel_sha256 = str(source_receipt["voxel_sha256_float32_le"])
    process = psutil.Process()
    baseline_rss = int(process.memory_info().rss)
    rss_samples = [baseline_rss]
    stop = threading.Event()

    def sample_rss() -> None:
        while not stop.wait(0.01):
            try:
                rss_samples.append(int(process.memory_info().rss))
            except psutil.Error:
                return

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    timings: dict[str, list[float]] = {"frozen_two_bincount": [], "single_bincount": []}
    run_rows: list[dict[str, object]] = []
    frozen_voxel: np.ndarray | None = None
    candidate_voxel: np.ndarray | None = None
    candidate_raw_sha256: str | None = None
    start_us = cutoff_us - config.window_ms * 1000
    order = [
        "frozen_two_bincount",
        "single_bincount",
        "single_bincount",
        "frozen_two_bincount",
        "frozen_two_bincount",
        "single_bincount",
    ]
    try:
        with EAPEventReader(events_path) as reader:
            for run_index, method in enumerate(order):
                chunks = reader.iter_window_chunks(
                    start_us, cutoff_us, chunk_events=config.chunk_events
                )
                started = time.perf_counter()
                if method == "frozen_two_bincount":
                    result = encode_causal_event_chunks(
                        chunks,
                        sequence_id=sequence_id,
                        cutoff_us=cutoff_us,
                        config=config,
                    )
                    elapsed = time.perf_counter() - started
                    voxel = result.voxel
                    event_count = int(result.receipt["event_count"])
                    voxel_sha256 = str(result.receipt["voxel_sha256_float32_le"])
                    frozen_voxel = voxel
                else:
                    accelerated = encode_causal_event_chunks_experimental(
                        chunks,
                        cutoff_us=cutoff_us,
                        config=config,
                        kind="single_bincount_candidate",
                    )
                    elapsed = time.perf_counter() - started
                    voxel = accelerated.voxel
                    event_count = accelerated.event_count
                    voxel_sha256 = accelerated.voxel_sha256_float32_le
                    candidate_voxel = voxel
                    candidate_raw_sha256 = accelerated.raw_numerator_sha256_int64_le
                if event_count != expected_event_count:
                    raise ValueError("A profiled path did not retain every source event")
                if voxel_sha256 != expected_voxel_sha256:
                    raise ValueError("A profiled path changed the admitted voxel bytes")
                timings[method].append(elapsed)
                run_rows.append(
                    {
                        "run_index": run_index,
                        "method": method,
                        "elapsed_seconds": elapsed,
                        "event_count": event_count,
                        "voxel_sha256_float32_le": voxel_sha256,
                    }
                )
            reference = encode_causal_event_chunks_experimental(
                reader.iter_window_chunks(start_us, cutoff_us, chunk_events=config.chunk_events),
                cutoff_us=cutoff_us,
                config=config,
                kind="two_bincount_reference",
            )
    finally:
        rss_samples.append(int(process.memory_info().rss))
        stop.set()
        sampler.join()
    if frozen_voxel is None or candidate_voxel is None or candidate_raw_sha256 is None:
        raise RuntimeError("Both profiled implementations must execute")
    if reference.event_count != expected_event_count:
        raise ValueError("Raw-count reference did not retain every source event")
    if reference.raw_numerator_sha256_int64_le != candidate_raw_sha256:
        raise ValueError("Candidate raw integer counts differ from the reference")
    difference = candidate_voxel - frozen_voxel
    differing_values = int(np.count_nonzero(difference))
    maximum_absolute_difference = float(np.max(np.abs(difference), initial=0.0))
    if differing_values != 0 or maximum_absolute_difference != 0.0:
        raise ValueError("Candidate voxel differs from the frozen implementation")
    peak_rss = max(rss_samples)
    if peak_rss >= MAXIMUM_RSS_BYTES:
        raise MemoryError("Profile exceeded the preregistered 1 GiB RSS bound")
    frozen_median = statistics.median(timings["frozen_two_bincount"])
    candidate_median = statistics.median(timings["single_bincount"])
    return {
        "schema": "eap_detection_event_accumulation_profile_v1",
        "status": "PASS",
        "decision": "CANDIDATE_FASTER" if candidate_median < frozen_median else "NO_GAIN",
        "scope": "CPU-only real 50 ms window; no labels/test/GPU/full-H5 hash",
        "source": {
            "probe_path": args.source_probe.as_posix(),
            "probe_sha256": sha256_file(args.source_probe),
            "sample_token": sample["sample_token"],
            "events_path": events_relative,
            "event_h5_expected_sha256": provenance["event_h5_expected_sha256_from_public_manifest"],
            "event_h5_expected_and_observed_size_bytes": expected_size,
            "full_h5_hash_recomputed": False,
            "window_start_us_inclusive": start_us,
            "window_end_us_exclusive": cutoff_us,
            "event_count": expected_event_count,
            "chunk_events": config.chunk_events,
        },
        "parity": {
            "all_events_retained_each_run": True,
            "raw_numerator_sha256_int64_le": candidate_raw_sha256,
            "raw_reference_sha256_int64_le": reference.raw_numerator_sha256_int64_le,
            "voxel_sha256_float32_le": expected_voxel_sha256,
            "differing_float32_values": differing_values,
            "maximum_absolute_float32_difference": maximum_absolute_difference,
            "chunk_partition_independent": True,
        },
        "benchmark": {
            "alternating_order": order,
            "repeats_per_method": args.repeats,
            "runs": run_rows,
            "frozen_seconds": timings["frozen_two_bincount"],
            "candidate_seconds": timings["single_bincount"],
            "frozen_median_seconds": frozen_median,
            "candidate_median_seconds": candidate_median,
            "speedup_frozen_over_candidate": frozen_median / candidate_median,
        },
        "resources": {
            "baseline_rss_bytes": baseline_rss,
            "sampled_peak_rss_bytes": peak_rss,
            "sampled_peak_delta_bytes": peak_rss - baseline_rss,
            "rss_limit_bytes_exclusive": MAXIMUM_RSS_BYTES,
            "sample_interval_ms": 10,
            "includes_h5_reader": True,
        },
        "provenance": {
            "frozen_event_data_sha256": sha256_file(
                Path(__file__).parents[1] / "src/e_jepa_ttc/detection/event_data.py"
            ),
            "candidate_module_sha256": sha256_file(
                Path(__file__).parents[1] / "src/e_jepa_ttc/detection/event_accumulation.py"
            ),
            "profile_script_sha256": sha256_file(Path(__file__)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-probe", type=Path, default=DEFAULT_SOURCE_PROBE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite: {args.output}")
    try:
        receipt = _profile(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception as error:
        print(f"ERROR: {error}")
        return 2
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
