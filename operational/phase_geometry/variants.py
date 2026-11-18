"""Pre-registered variant definitions: inputs, anchor, width base and training mass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from . import geometry
from .weights import band_mass, group_mass

ELIGIBLE = ("REL_BAND", "REL_GEO", "REL_GEO_BAND", "RGB_GEO_BAND", "GEO_LEARNED")
MLP_VARIANTS = ("REL", *ELIGIBLE)


@dataclass(frozen=True)
class Variant:
    """One learned head configuration from PROTOCOL.json."""

    name: str
    inputs: str
    loss: str
    anchor: str


VARIANTS = {
    "REL": Variant("REL", "phase20", "GROUP", "rgb"),
    "REL_BAND": Variant("REL_BAND", "phase20", "BAND", "rgb"),
    "REL_GEO": Variant("REL_GEO", "phase20+g", "GROUP", "rgb"),
    "REL_GEO_BAND": Variant("REL_GEO_BAND", "phase20+g", "BAND", "rgb"),
    "RGB_GEO_BAND": Variant("RGB_GEO_BAND", "rgb10+g", "BAND", "rgb"),
    "GEO_LEARNED": Variant("GEO_LEARNED", "g", "BAND", "geometry"),
    "GBM_REF": Variant("GBM_REF", "phase20+g", "BAND", "none"),
}


def scaled_phase20(features: NDArray[Any]) -> NDArray[np.float32]:
    """Released ``PhaseOnlyFinal.inputs`` scaling (fields 0:5 and 10:15 divided by 0.03, FP32)."""
    selected = torch.as_tensor(np.asarray(features, dtype=np.float32)).clone()
    selected[:, :5] /= 0.03
    selected[:, 10:15] /= 0.03
    return selected.numpy()


def design(
    variant: Variant, arrays: dict[str, NDArray[Any]], rows: NDArray[Any] | slice
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Return (inputs, anchor phase, width-logit base) for the selected rows."""
    features = np.asarray(arrays["features"][rows], dtype=np.float32)
    phase20 = scaled_phase20(features)
    g = np.asarray(arrays["geometry_g"][rows], dtype=np.float32)
    if variant.inputs == "phase20":
        x = phase20
    elif variant.inputs == "phase20+g":
        x = np.concatenate((phase20, g), axis=1)
    elif variant.inputs == "rgb10+g":
        x = np.concatenate((phase20[:, :10], g), axis=1)
    elif variant.inputs == "g":
        x = g
    else:
        raise ValueError(f"Unknown input set {variant.inputs}")
    if variant.anchor == "geometry":
        phi = geometry.raw_column(arrays["geometry_raw"][rows], "phi_area")
        anchor = np.clip(phi, -geometry.LEARNED_PHASE_CLIP, geometry.LEARNED_PHASE_CLIP)
        width = np.zeros((len(x), 2), dtype=np.float32)
    else:
        anchor = features[:, 0]
        width = np.asarray(arrays["rgb_width_logits"][rows], dtype=np.float32)
    return x.astype(np.float32), np.asarray(anchor, dtype=np.float32), width


def training_mass(
    variant: Variant, groups: NDArray[Any], truth_ttc: NDArray[Any]
) -> NDArray[np.float32]:
    """GROUP (released) or BAND (metric-matched) mass over the training rows only."""
    if variant.loss == "GROUP":
        return group_mass(groups)
    if variant.loss == "BAND":
        return band_mass(truth_ttc)
    raise ValueError(f"Unknown loss {variant.loss}")


def input_count(variant: Variant) -> int:
    """Number of head inputs."""
    return {"phase20": 20, "phase20+g": 30, "rgb10+g": 20, "g": 10}[variant.inputs]


__all__ = [
    "ELIGIBLE",
    "MLP_VARIANTS",
    "VARIANTS",
    "Variant",
    "design",
    "input_count",
    "training_mass",
]
