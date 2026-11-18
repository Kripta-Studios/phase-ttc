"""Content-scale estimator: exact area binning, NCC, known-zoom recovery and causality."""

from __future__ import annotations

import math

import numpy as np
import pytest

from operational.saturation_scale import content_scale as cs


def _events(points: np.ndarray) -> np.ndarray:
    inside = (
        (points[:, 0] >= 0)
        & (points[:, 0] < cs.SENSOR_WIDTH)
        & (points[:, 1] >= 0)
        & (points[:, 1] < cs.SENSOR_HEIGHT)
    )
    q = points[inside]
    return cs.integral_image(q[:, 0].astype(int), q[:, 1].astype(int))


def _texture(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pieces = []
    for _ in range(250):
        start = rng.uniform([-1, -1], [1, 1]) * (300, 200)
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        t = rng.uniform(0, rng.uniform(10, 60), 120)
        pieces.append(start + t[:, None] * direction)
    return np.concatenate(pieces)


def test_integral_image_counts_and_rejects_outside_events() -> None:
    x = np.asarray([0, 1, 1, 1279, 1280, -1, 5])
    y = np.asarray([0, 0, 0, 719, 3, 2, 720])
    integral = cs.integral_image(x, y)
    assert integral.shape == (cs.SENSOR_HEIGHT + 1, cs.SENSOR_WIDTH + 1)
    assert integral[-1, -1] == 4
    assert integral[1, 2] == 3  # row 0, cols 0..1


def test_area_binning_conserves_mass_and_splits_pixels_uniformly() -> None:
    integral = cs.integral_image(np.asarray([10] * 4), np.asarray([20] * 4))
    whole = cs.bin_regions(integral, np.asarray([[10.5, 20.5]]), np.asarray([[0.5, 0.5]]), 2)
    assert whole.sum() == pytest.approx(4.0)
    assert np.allclose(whole, 1.0)  # four quarter-pixels of one pixel with 4 events
    region = cs.bin_regions(integral, np.asarray([[12.0, 22.0]]), np.asarray([[5.0, 5.0]]), 4)
    assert region.sum() == pytest.approx(4.0)


def test_prepare_is_gain_covariant() -> None:
    rng = np.random.default_rng(1)
    mass = rng.poisson(3.0, (2, 16, 16)).astype(float)
    assert np.allclose(cs.prepare(4.0 * mass), 2.0 * cs.prepare(mass))


def test_ncc_finds_an_embedded_template() -> None:
    rng = np.random.default_rng(2)
    image = rng.normal(size=(cs.GRID, cs.GRID))
    template = cs.make_template(image[5 : 5 + cs.TEMPLATE, 9 : 9 + cs.TEMPLATE], 1e4)
    scores = cs.ncc_maps(image[None], template)[0]
    assert np.unravel_index(np.argmax(scores), scores.shape) == (5, 9)
    assert scores.max() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("k_true", [0.6, -0.4])
def test_query_content_recovers_a_known_expansion_rate(k_true: float) -> None:
    texture = _texture()
    center = np.asarray([640.0, 360.0])
    current = _events(center + texture)
    previous = {
        lag: _events(center + texture / (1.0 + k_true * lag * 1e-6) + (4.0, -2.0))
        for lag in cs.LAGS_US
    }
    region = cs.box_region((390, 160, 889, 559))
    out = cs.query_content(current, previous, region, {lag: None for lag in cs.LAGS_US})
    assert out["valid"] == 1.0 and out["n_valid"] == len(cs.LAGS_US)
    assert out["k_content"] == pytest.approx(k_true, abs=0.06)
    for lag in cs.LAGS_US:
        assert out[f"rho_{lag // 1000}"] == pytest.approx(math.log1p(k_true * lag * 1e-6), abs=0.02)


def test_query_content_without_events_or_region_is_empty() -> None:
    empty = cs.integral_image(np.zeros(0, int), np.zeros(0, int))
    region = cs.box_region((100, 100, 300, 300))
    out = cs.query_content(empty, {lag: empty for lag in cs.LAGS_US}, region, {})
    assert out["valid"] == 0.0 and out["n_valid"] == 0.0
    assert cs.query_content(None, {}, region, {}) == cs.empty_raw()
    assert cs.query_content(empty, {}, cs.Region(5, 5, 5, 9), {}) == cs.empty_raw()


def test_regions_clip_to_sensor_and_virtual_crop() -> None:
    full = cs.box_region((-5, 0, 1400, 719))
    assert (full.x0, full.y0, full.x1, full.y1) == (0.0, 0.0, 1280.0, 720.0)
    crop = cs.crop_region((100.0, 50.0, 4.0))
    assert crop.x1 == pytest.approx(420.0) and crop.y1 == pytest.approx(230.0)
    assert full.intersect(crop) == crop
    assert cs.Region(0, 0, 10, 10).intersect(cs.Region(20, 20, 30, 30)).empty


def test_fit_rate_is_exact_under_constant_velocity() -> None:
    k = 0.8
    estimates = {
        lag: cs.LagEstimate(math.log1p(k * lag * 1e-6), 0.5 + lag * 1e-7, 0.1, False, True)
        for lag in cs.LAGS_US
    }
    slope, rms = cs.fit_rate(estimates)
    assert slope == pytest.approx(k) and rms == pytest.approx(0.0, abs=1e-12)


def test_search_range_and_causal_window_key() -> None:
    lo, hi = cs.rho_bounds(0.6)
    assert lo == pytest.approx(math.log(0.4)) and hi == pytest.approx(math.log(2.8))
    assert cs.window_key_ms(1_045_202_680) * 1000 <= 1_045_202_680
    assert max(cs.LAGS_US) + cs.TAU_US <= 650_000  # declared causal budget


def test_learned_inputs_shape_and_invalid_rows_are_zero() -> None:
    raw = np.zeros((3, len(cs.RAW_C_FIELDS)))
    raw[1, cs.RAW_C_FIELDS.index("phi_content")] = 2.0
    raw[1, cs.RAW_C_FIELDS.index("valid")] = 1.0
    c = cs.learned_c(raw)
    assert c.shape == (3, len(cs.C_FIELDS)) and c.dtype == np.float32
    assert np.all(c[0] == 0) and c[1, 0] == pytest.approx(cs.PHASE_CLIP / cs.PHASE_SCALE)
