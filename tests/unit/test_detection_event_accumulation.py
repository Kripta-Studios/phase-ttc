from __future__ import annotations

import numpy as np
import pytest

from e_jepa_ttc.detection.event_accumulation import (
    FLOAT64_EXACT_INTEGER_MAX,
    accumulate_linear_chunk,
    encode_causal_event_chunks_experimental,
    validate_exact_accumulation_bounds,
)
from e_jepa_ttc.detection.event_data import (
    CausalEventConfig,
    encode_causal_event_chunks,
)


def _chunk(start_us: int) -> dict[str, np.ndarray]:
    return {
        "x": np.asarray([0, 1, 1278, 1279, 640, 320], dtype=np.int32),
        "y": np.asarray([0, 1, 718, 719, 360, 180], dtype=np.int32),
        "t": np.asarray(
            [
                start_us,
                start_us + 1,
                start_us + 12_500,
                start_us + 25_000,
                start_us + 49_998,
                start_us + 49_999,
            ],
            dtype=np.int64,
        ),
        "p": np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8),
    }


def _split(chunk: dict[str, np.ndarray], stops: tuple[int, ...]) -> list[dict[str, np.ndarray]]:
    boundaries = (0, *stops, len(chunk["x"]))
    return [
        {key: value[left:right] for key, value in chunk.items()}
        for left, right in zip(boundaries, boundaries[1:], strict=False)
    ]


def test_single_bincount_matches_frozen_output_and_reference_raw_counts() -> None:
    cutoff = 1_000_000
    start = cutoff - 50_000
    config = CausalEventConfig(width=8, height=6, normalize=True)
    chunk = _chunk(start)
    frozen = encode_causal_event_chunks(
        [chunk], sequence_id="sequence", cutoff_us=cutoff, config=config
    )
    reference = encode_causal_event_chunks_experimental(
        [chunk], cutoff_us=cutoff, config=config, kind="two_bincount_reference"
    )
    candidate = encode_causal_event_chunks_experimental(
        [chunk], cutoff_us=cutoff, config=config, kind="single_bincount_candidate"
    )
    assert np.array_equal(candidate.raw_numerator, reference.raw_numerator)
    assert candidate.raw_numerator_sha256_int64_le == reference.raw_numerator_sha256_int64_le
    assert np.array_equal(candidate.voxel, reference.voxel)
    assert np.array_equal(candidate.voxel, frozen.voxel)
    assert candidate.voxel_sha256_float32_le == frozen.receipt["voxel_sha256_float32_le"]
    assert candidate.event_count == 6


def test_candidate_is_exactly_chunk_partition_independent() -> None:
    cutoff = 1_000_000
    chunk = _chunk(cutoff - 50_000)
    config = CausalEventConfig(width=8, height=6, normalize=False)
    whole = encode_causal_event_chunks_experimental([chunk], cutoff_us=cutoff, config=config)
    partitioned = encode_causal_event_chunks_experimental(
        _split(chunk, (1, 3, 5)), cutoff_us=cutoff, config=config
    )
    assert np.array_equal(whole.raw_numerator, partitioned.raw_numerator)
    assert np.array_equal(whole.voxel, partitioned.voxel)
    assert whole.raw_numerator_sha256_int64_le == partitioned.raw_numerator_sha256_int64_le


def test_float64_and_int64_overflow_bounds_fail_closed() -> None:
    maximum_events = FLOAT64_EXACT_INTEGER_MAX // 50_000
    validate_exact_accumulation_bounds(event_count=maximum_events, window_us=50_000)
    with pytest.raises(OverflowError, match="float64"):
        validate_exact_accumulation_bounds(event_count=maximum_events + 1, window_us=50_000)
    accumulator = np.asarray([np.iinfo(np.int64).max], dtype=np.int64)
    with pytest.raises(OverflowError, match="Cross-chunk"):
        accumulate_linear_chunk(
            accumulator,
            lower_flat=np.asarray([0], dtype=np.int64),
            upper_flat=np.asarray([0], dtype=np.int64),
            upper_remainder=np.asarray([0], dtype=np.int64),
            window_us=1,
            kind="single_bincount_candidate",
        )


def test_accumulator_rejects_invalid_indices_and_remainders() -> None:
    accumulator = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="exceeds"):
        accumulate_linear_chunk(
            accumulator,
            lower_flat=np.asarray([2]),
            upper_flat=np.asarray([0]),
            upper_remainder=np.asarray([0]),
            window_us=10,
            kind="single_bincount_candidate",
        )
    with pytest.raises(ValueError, match="remainder"):
        accumulate_linear_chunk(
            accumulator,
            lower_flat=np.asarray([0]),
            upper_flat=np.asarray([1]),
            upper_remainder=np.asarray([10]),
            window_us=10,
            kind="single_bincount_candidate",
        )
