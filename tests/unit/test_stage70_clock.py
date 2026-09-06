"""Clock failures that would silently change PPS's physical hypothesis."""

import numpy as np
import pytest

from e_jepa_ttc.data.stage70_clock import context_clock


def test_integer_subtraction_precedes_float_cast() -> None:
    base = context_clock(((0, 100_001), (100_001, 200_003)), (900_001, 1_000_003), 1_000_003)
    shift = 2**58
    shifted = context_clock(
        ((shift, shift + 100_001), (shift + 100_001, shift + 200_003)),
        (shift + 900_001, shift + 1_000_003),
        shift + 1_000_003,
    )
    np.testing.assert_array_equal(base.midpoint_seconds(), shifted.midpoint_seconds())
    assert base.midpoint_twice_relative_us == (-350006, -250005, -150003, -50001)
    assert base.gap_us == 0


def test_gap_is_retained_not_filled() -> None:
    clock = context_clock(((0, 80), (100, 200)), (1080, 1200), 1200)
    assert clock.gap_us == 20
    np.testing.assert_allclose(clock.midpoint_seconds(), [-0.00018, -0.00014, -0.000075, -0.000025])


def test_overlap_is_retained_with_ordered_midpoints() -> None:
    clock = context_clock(((0, 100), (80, 180)), (1000, 1080), 1080)
    assert clock.gap_us == -20


@pytest.mark.parametrize(
    ("windows", "frames", "anchor", "reason"),
    [
        (((0, 100), (100, 200)), (1000, 1100), 1000, "later selected frame"),
        (((0, 100), (100, 200)), (1000, 1106), 1106, "offset drift"),
        (((0, 7), (7, 14)), (1007, 1014), 1014, "positive-duration"),
        (((0, 100), (0, 110)), (1000, 1010), 1010, "strictly ordered"),
        (((0.0, 100), (100, 200)), (1000, 1100), 1100, "integer microseconds"),
    ],
)
def test_invalid_clock_contracts_fail(windows, frames, anchor, reason) -> None:
    with pytest.raises(ValueError, match=reason):
        context_clock(windows, frames, anchor)
