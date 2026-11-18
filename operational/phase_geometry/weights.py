"""Per-row training mass for the released group-balanced and the band-weighted objectives."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .metrics import BAND_WEIGHTS, BANDS, band_index


def group_mass(groups: NDArray[Any]) -> NDArray[np.float32]:
    """Released mass ``1/(G * n_group)``: every acquisition sequence weighs the same."""
    names = np.asarray(groups).astype(str)
    _, inverse, counts = np.unique(names, return_inverse=True, return_counts=True)
    return (1.0 / (len(counts) * counts[inverse])).astype(np.float32)


def band_mass(truth_ttc: NDArray[Any]) -> NDArray[np.float32]:
    """Metric-matched mass ``w_band / n_band`` (Overall weights 0.5/0.3/0.1/0.1).

    Rows outside the four official bands get zero mass.  Bands absent from the training
    rows are dropped and the remaining weights renormalized so the mass sums to one.
    """
    bands = band_index(truth_ttc)
    mass = np.zeros(bands.shape, dtype=np.float64)
    present = {i: int((bands == i).sum()) for i in range(len(BANDS))}
    total_weight = sum(BAND_WEIGHTS[BANDS[i]] for i, n in present.items() if n > 0)
    if total_weight <= 0:
        raise ValueError("No training row falls in an official band")
    for index, count in present.items():
        if count:
            mass[bands == index] = BAND_WEIGHTS[BANDS[index]] / total_weight / count
    return mass.astype(np.float32)


def weighted_l1(
    prediction_phase: NDArray[Any], target_phase: NDArray[Any], mass: NDArray[Any]
) -> float:
    """Population-scaled weighted L1 point term in released loss units (|dphi|/0.03)."""
    p = np.asarray(prediction_phase, dtype=np.float64)
    t = np.asarray(target_phase, dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64)
    return float((len(m) * m * np.abs(p - t) / 0.03).mean())


__all__ = ["band_mass", "group_mass", "weighted_l1"]
