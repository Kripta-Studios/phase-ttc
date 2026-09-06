"""Physical raw-count and TRAIN-normalization integration invariants."""

from dataclasses import fields

import numpy as np
import pytest

from e_jepa_ttc.data.event_context128 import (
    COUNT_SHAPE,
    ArchitectureInput,
    RateMoments,
    RateNormalization,
    Window128,
    checked_count_cast,
    four_contexts,
    log_rates,
    rasterize128,
)


def test_eight_integer_bins_preserve_both_polarities_and_calibration():
    window = Window128(100, 116, (5.0, 0.0, 21.0, 16.0), (17, 15))
    timestamps = np.arange(100, 116, dtype=np.int64)
    counts, diagnostic = rasterize128(
        np.zeros(16),
        np.zeros(16),
        timestamps,
        np.arange(16) % 2,
        window,
    )
    assert counts.shape == (8, 2, 128, 128)
    np.testing.assert_array_equal(counts[:, :, 0, 0], np.ones((8, 2)))
    assert counts.sum() == diagnostic["roi_events"] == 16
    assert window.support().sum() == 120 * 128


def test_large_absolute_clock_is_not_cast_before_binning():
    shift = 2**58
    x = np.array([0, 3, 8, 15])
    t = np.array([100, 105, 110, 115], dtype=np.int64)
    first = Window128(100, 116, (5.0, 0.0, 21.0, 16.0), (17, 15))
    second = Window128(100 + shift, 116 + shift, first.roi, first.sensor_size)
    a, _ = rasterize128(x, x, t, x % 2, first)
    b, _ = rasterize128(x, x, t + shift, x % 2, second)
    np.testing.assert_array_equal(a, b)


def test_sensor_and_roi_exclusions_are_counted_separately():
    window = Window128(0, 80, (5.0, 0.0, 21.0, 16.0), (17, 15))
    counts, diagnostic = rasterize128(
        np.array([-1, 0, 15, 16, 17]),
        np.array([0, 0, 14, 0, 0]),
        np.arange(5),
        np.ones(5),
        window,
    )
    assert counts.sum() == 2
    assert diagnostic == {"raw_events": 5, "outside_sensor": 2, "outside_roi": 1, "roi_events": 2}


def test_empty_activity_retained_and_window_end_excluded():
    window = Window128(0, 80, (0.0, 0.0, 128.0, 128.0), (128, 128), x_offset=0)
    empty = np.array([], dtype=np.int64)
    counts, _ = rasterize128(empty, empty, empty, empty, window)
    assert counts.sum() == 0
    with pytest.raises(ValueError, match="half-open"):
        rasterize128(np.array([1]), np.array([1]), np.array([80]), np.array([1]), window)
    with pytest.raises(ValueError, match="integer"):
        rasterize128(np.array([1]), np.array([1]), np.array([2.5]), np.array([1]), window)


def test_overflow_requires_whole_version_upgrade():
    values = np.array([65535, 65536], dtype=np.uint32)
    with pytest.raises(OverflowError, match="whole cache version"):
        checked_count_cast(values)
    np.testing.assert_array_equal(checked_count_cast(values, "uint32"), values)


def test_four_contexts_are_bin_major_then_negative_positive():
    counts = np.zeros(COUNT_SHAPE, dtype=np.uint16)
    for window in range(2):
        for bin_id in range(8):
            for p in range(2):
                counts[window, bin_id, p] = window * 16 + bin_id * 2 + p
    normalization = RateNormalization((0.0, 0.0), (1.0, 1.0), "f" * 64, 1)
    support = np.ones((128, 128), dtype=bool)
    inputs = four_contexts(
        counts,
        np.ones((2, 8)),
        support,
        np.array([-350000, -250000, -150000, -50000]),
        normalization,
    )
    np.testing.assert_allclose(
        inputs.contexts[:, :, 0, 0].numpy(), np.log1p(np.arange(32).reshape(4, 8)), rtol=1e-7
    )
    assert {field.name for field in fields(ArchitectureInput)} == {
        "contexts",
        "times",
        "sensor_support",
    }


def test_streaming_moments_match_direct_population_moments_excluding_padding():
    generator = np.random.default_rng(7)
    support = np.zeros((128, 128), dtype=bool)
    support[10:20, 30:40] = True
    moments = RateMoments()
    rows = []
    for _ in range(3):
        counts = generator.integers(0, 30, size=COUNT_SHAPE, dtype=np.uint16)
        transformed = log_rates(counts, np.ones((2, 8)) * 0.0125)
        rows.append(transformed)
        moments.add(transformed, support)
    normalization = moments.finish("a" * 64)
    for p in range(2):
        expected = np.concatenate([row[:, :, p, support].reshape(-1) for row in rows]).astype(
            np.float64
        )
        assert normalization.mean[p] == pytest.approx(expected.mean(), abs=1e-12)
        assert normalization.std[p] == pytest.approx(expected.std(), abs=1e-12)
    assert normalization.valid_values_per_polarity == 3 * 16 * 100
