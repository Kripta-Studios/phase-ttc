"""Fast synthetic tests for the literature-baseline conversions, aggregation and subset."""

from __future__ import annotations

import math

import numpy as np
import pytest

from operational.literature_baselines.common import (
    CROP_SIZE,
    DT,
    affine_rho,
    aggregate,
    allocate,
    box_in_crop,
    box_ratio_rho,
    clean_rho,
    common_square,
    cross_fitted_scale,
    fit_scale,
    phase_to_ttc,
    region_mask,
    rho_from_depth_ratio,
    rho_to_phase_cv,
    rho_to_phase_exp,
    select_subset,
    ttc_to_rho_cv,
    zoom_square,
)
from operational.phase_geometry.metrics import row_mid, ttc_to_phase


def _expanding_flow(size: int, centre: tuple[float, float], ratio: float) -> np.ndarray:
    """Flow image1 -> image2 of a pattern whose image-2 scale is ``ratio`` times image 1."""
    centres = np.arange(size) + 0.5
    xx, yy = np.meshgrid(centres, centres)
    return np.stack([(ratio - 1.0) * (xx - centre[0]), (ratio - 1.0) * (yy - centre[1])])


@pytest.mark.parametrize("ttc", [0.4, 1.0, 2.5, 5.0, 8.0, -1.5, -6.0])
def test_constant_velocity_conversion_recovers_phase(ttc: float) -> None:
    """A pattern approaching at constant velocity with known TTC gives zero MiD."""
    # Depths Z(t) = T*V and Z(t - 0.1) = (T + 0.1)*V: scale s ~ 1/Z.
    rho = math.log((ttc + DT) / ttc)
    assert rho == pytest.approx(ttc_to_rho_cv(ttc, DT))
    phase = float(rho_to_phase_cv(rho, DT))
    assert phase == pytest.approx(float(ttc_to_phase(np.asarray([ttc]))[0]), abs=1e-12)
    decoded = phase_to_ttc(np.asarray([phase]))
    assert row_mid(np.asarray([ttc]), decoded)[0] == pytest.approx(0.0, abs=1e-6)


def test_constant_velocity_conversion_rescales_interval() -> None:
    """A 50 ms pair of the same motion yields the same 0.1 s phase."""
    ttc = 2.0
    rho_half = math.log((ttc + 0.05) / ttc)
    assert float(rho_to_phase_cv(rho_half, 0.05)) == pytest.approx(-math.log1p(-DT / ttc))


def test_exponential_conversion_and_domain_clip() -> None:
    """EXP is linear in rho; CV saturates at MAX_PHASE when the pair implies T <= 0.1 s."""
    assert float(rho_to_phase_exp(0.02, 0.1)) == pytest.approx(0.02)
    assert float(rho_to_phase_exp(0.02, 0.05)) == pytest.approx(0.04)
    assert float(rho_to_phase_cv(math.log(2.5), 0.1)) == 8.0
    assert phase_to_ttc(np.asarray([8.0]))[0] > DT


def test_depth_ratio_and_box_ratio_sign_conventions() -> None:
    """Approaching objects give rho > 0 for every native convention."""
    # Image 2 is the earlier frame: Z2/Z1 = Z(t-d)/Z(t) > 1 when approaching.
    assert float(rho_from_depth_ratio(np.asarray(1.05))) == pytest.approx(math.log(1.05))
    assert box_ratio_rho([0, 0, 99, 49], [0, 0, 109, 54]) == pytest.approx(
        0.5 * math.log(110 * 55 / (100 * 50))
    )
    cleaned, failed = clean_rho(np.asarray([0.1, np.nan, np.inf]))
    assert cleaned.tolist() == [0.1, 0.0, 0.0]
    assert failed.tolist() == [False, True, True]


@pytest.mark.parametrize("ratio", [0.9, 0.97, 1.0, 1.04, 1.12])
def test_affine_fit_recovers_scale_and_trims_outliers(ratio: float) -> None:
    """``-0.5 log det`` of the fitted affine equals ``log(s(t)/s(t-d))`` despite outliers."""
    size = 64
    flow = _expanding_flow(size, (30.0, 34.0), ratio) + np.asarray([1.5, -2.0])[:, None, None]
    rng = np.random.default_rng(0)
    corrupt = rng.random((size, size)) < 0.1
    flow[:, corrupt] += 25.0
    mask = region_mask([8, 8, 56, 56], size)
    assert affine_rho(flow, mask) == pytest.approx(-math.log(ratio), abs=5e-3)
    assert math.isnan(affine_rho(flow, np.zeros((size, size), dtype=bool)))


def test_dense_median_aggregation_inside_box() -> None:
    """The median ignores the background and non-finite pixels."""
    size = 32
    values = np.full((size, size), 0.5)
    values[8:24, 8:24] = 0.02
    values[10, 10] = np.nan
    values[11, 11] = 9.0
    mask = region_mask([8, 8, 24, 24], size)
    assert mask.sum() == 256
    assert aggregate(values, mask, "median") == pytest.approx(0.02)
    inner = region_mask([8, 8, 24, 24], size, inner=True)
    assert inner.sum() == 64
    assert aggregate(values, inner, "median") == pytest.approx(0.02)
    assert math.isnan(aggregate(values, np.zeros_like(mask), "median"))


def test_crop_geometry_preserves_scale_ratio() -> None:
    """Both endpoint boxes are mapped with the same square, so their size ratio is kept."""
    previous, current = [100, 200, 199, 299], [95, 195, 204, 304]
    square = common_square([previous, current])
    assert square[2] - square[0] == pytest.approx(110 * 1.5)
    a = box_in_crop(previous, square, CROP_SIZE)
    b = box_in_crop(current, square, CROP_SIZE)
    assert (b[2] - b[0]) / (a[2] - a[0]) == pytest.approx(110 / 100)
    zoomed = zoom_square(square, (150.0, 250.0), 1.05)
    assert (square[2] - square[0]) / (zoomed[2] - zoomed[0]) == pytest.approx(1.05)


def test_allocation_is_proportional_and_exact() -> None:
    """Largest remainder sums exactly and breaks ties by band order."""
    assert allocate([10, 30, 20, 40], 10) == [1, 3, 2, 4]
    assert allocate([1, 1, 1, 1], 2) == [1, 1, 0, 0]
    assert sum(allocate([7, 13, 0, 21], 17)) == 17
    with pytest.raises(ValueError):
        allocate([1, 1], 3)


def _population(seed: int = 3) -> tuple[list[str], list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    tokens, sequences, bands = [], [], []
    for s in range(5):
        for i in range(int(rng.integers(60, 90))):
            tokens.append(f"seq{s}_{i:04d}")
            sequences.append(f"seq{s}")
            bands.append(int(rng.integers(0, 4)))
    return tokens, sequences, np.asarray(bands)


def test_subset_is_deterministic_stratified_and_prefix_stable() -> None:
    """Same inputs give the same rows in any input order; smaller sizes are prefixes."""
    tokens, sequences, bands = _population()
    first = select_subset(tokens, sequences, bands, per_sequence=20, seed=11)
    again = select_subset(tokens, sequences, bands, per_sequence=20, seed=11)
    assert np.array_equal(first, again)
    order = np.random.default_rng(5).permutation(len(tokens))
    shuffled = select_subset(
        [tokens[i] for i in order],
        [sequences[i] for i in order],
        bands[order],
        per_sequence=20,
        seed=11,
    )
    assert [tokens[i] for i in first] == [tokens[order[i]] for i in shuffled]
    chosen_sequences = np.asarray(sequences)[first]
    assert all((chosen_sequences == f"seq{s}").sum() == 20 for s in range(5))
    assert len(set(first.tolist())) == len(first)
    smaller = set(select_subset(tokens, sequences, bands, per_sequence=10, seed=11).tolist())
    assert smaller <= set(first.tolist())
    other = select_subset(tokens, sequences, bands, per_sequence=20, seed=12)
    assert not np.array_equal(first, other)


def test_one_parameter_calibration_recovers_scale_and_cross_fits() -> None:
    """A uniformly shrunk estimator is recovered by the weighted-median scale."""
    rng = np.random.default_rng(1)
    truth = rng.normal(0.0, 0.03, 400)
    bands = rng.integers(0, 4, 400)
    estimate = 0.6 * truth
    assert fit_scale(truth, estimate, bands) == pytest.approx(1 / 0.6)
    folds = np.repeat(np.arange(4), 100)
    calibrated, scales = cross_fitted_scale(truth, estimate, bands, folds)
    assert set(scales) == {0, 1, 2, 3}
    assert np.allclose(calibrated, truth)
