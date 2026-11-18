"""Pre-registered EVT_* head variants and one input assembly shared by eAP and FCWD.

The same :func:`design` is used for the TRAIN40 out-of-fold rows and for FCWD queries, so
the feature assembly cannot drift between training and deployment.  Event fields are the ten
fusion coordinates ``block[160:170]`` of the sealed E_CTX frozen state (PHASE20 fields 10-19);
the first five are divided by 0.03 exactly as ``phase_geometry.variants.scaled_phase20``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from operational.phase_geometry import geometry
from operational.phase_geometry.weights import band_mass

PHASE_SCALE = 0.03
EVENT_FIELDS = 10
E5 = (0, 1, 2, 3, 4)  # E_CTX raw location, point phase, A5/C2F/PAIR expert phases
FCWD_ELIGIBLE = ("EVT_GEO_BAND", "EVT_GEO_BAND_GA")
NEW_VARIANTS = ("EVT_GEO_BAND", "EVT_GEO_BAND_GA", "EVT10_GEO_BAND", "EVT_BAND", "REL_E5_GEO_BAND")
REFERENCE_VARIANTS = ("REL", "REL_GEO_BAND", "RGB_GEO_BAND", "GEO_LEARNED")


@dataclass(frozen=True)
class Variant:
    """One head: which blocks it reads, its anchor and whether it uses RGB width logits."""

    name: str
    rgb: bool
    event_fields: tuple[int, ...]
    geometry: bool
    anchor: str  # "event" | "geometry" | "rgb"

    @property
    def inputs(self) -> int:
        return (10 if self.rgb else 0) + len(self.event_fields) + (10 if self.geometry else 0)

    @property
    def uses_rgb_width(self) -> bool:
        return self.anchor == "rgb"


VARIANTS: dict[str, Variant] = {
    "EVT_GEO_BAND": Variant("EVT_GEO_BAND", False, E5, True, "event"),
    "EVT_GEO_BAND_GA": Variant("EVT_GEO_BAND_GA", False, E5, True, "geometry"),
    "EVT10_GEO_BAND": Variant("EVT10_GEO_BAND", False, tuple(range(10)), True, "event"),
    "EVT_BAND": Variant("EVT_BAND", False, E5, False, "event"),
    "REL_E5_GEO_BAND": Variant("REL_E5_GEO_BAND", True, E5, True, "rgb"),
}


def scale_event10(event10: NDArray[Any]) -> NDArray[np.float32]:
    """Scale ``[N, 10]`` event fields like ``scaled_phase20`` does for columns 10:20."""
    values = np.array(event10, dtype=np.float32, copy=True)
    if values.ndim != 2 or values.shape[1] != EVENT_FIELDS:
        raise ValueError("event fields must be [N, 10]")
    values[:, :5] /= np.float32(PHASE_SCALE)
    return values


def scale_rgb10(rgb10: NDArray[Any]) -> NDArray[np.float32]:
    """Scale ``[N, 10]`` RGB fields like ``scaled_phase20`` does for columns 0:10."""
    values = np.array(rgb10, dtype=np.float32, copy=True)
    if values.ndim != 2 or values.shape[1] != 10:
        raise ValueError("RGB fields must be [N, 10]")
    values[:, :5] /= np.float32(PHASE_SCALE)
    return values


def geometry_anchor(geometry_raw: NDArray[Any]) -> NDArray[np.float32]:
    """GEO_LEARNED anchor: phi_area clipped to the learned-input range."""
    phi = geometry.raw_column(geometry_raw, "phi_area")
    clip = geometry.LEARNED_PHASE_CLIP
    return np.clip(phi, -clip, clip).astype(np.float32)


def design(
    variant: Variant,
    event10: NDArray[Any],
    geometry_raw: NDArray[Any],
    rgb10: NDArray[Any] | None = None,
    rgb_width_logits: NDArray[Any] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Return (inputs, anchor phase, width-logit base) for aligned rows.

    ``event10`` are unscaled PHASE20 fields 10-19, ``geometry_raw`` the raw table of
    ``geometry.RAW_FIELDS``; ``rgb10``/``rgb_width_logits`` are required only for RGB variants.
    """
    events = scale_event10(event10)
    raw = np.asarray(geometry_raw, dtype=np.float64)
    if len(raw) != len(events):
        raise ValueError("event and geometry rows are misaligned")
    parts: list[NDArray[np.float32]] = []
    if variant.rgb:
        if rgb10 is None:
            raise ValueError(f"{variant.name} needs RGB fields")
        parts.append(scale_rgb10(rgb10))
    parts.append(events[:, list(variant.event_fields)])
    if variant.geometry:
        parts.append(geometry.learned_inputs(raw))
    x = np.concatenate(parts, axis=1).astype(np.float32)
    if variant.anchor == "event":
        anchor = np.asarray(event10, dtype=np.float32)[:, 0]
    elif variant.anchor == "geometry":
        anchor = geometry_anchor(raw)
    elif variant.anchor == "rgb":
        if rgb10 is None:
            raise ValueError(f"{variant.name} needs RGB fields")
        anchor = np.asarray(rgb10, dtype=np.float32)[:, 0]
    else:
        raise ValueError(f"unknown anchor {variant.anchor}")
    if variant.uses_rgb_width:
        if rgb_width_logits is None:
            raise ValueError(f"{variant.name} needs RGB width logits")
        width = np.asarray(rgb_width_logits, dtype=np.float32)
    else:
        width = np.zeros((len(x), 2), dtype=np.float32)
    if x.shape[1] != variant.inputs or not np.isfinite(x).all() or not np.isfinite(anchor).all():
        raise ValueError(f"{variant.name}: design is not finite or has the wrong width")
    return x, np.ascontiguousarray(anchor, dtype=np.float32), width


def stack_columns(variant: Variant) -> list[int]:
    """Columns of the stacked matrix ``[scaled PHASE20 (20) | G (10)]`` used by ``variant``."""
    columns: list[int] = list(range(10)) if variant.rgb else []
    columns += [10 + i for i in variant.event_fields]
    if variant.geometry:
        columns += list(range(20, 30))
    return columns


ANCHOR_COLUMN = {"rgb": 0, "geometry": 1, "event": 2}


def stacked_anchors(features: NDArray[Any], geometry_raw: NDArray[Any]) -> NDArray[np.float32]:
    """``[N, 3]`` anchor matrix (RGB field 0, clipped phi_area, event field 10)."""
    f = np.asarray(features, dtype=np.float32)
    return np.stack((f[:, 0], geometry_anchor(geometry_raw), f[:, 10]), axis=1)


def training_mass(truth_ttc: NDArray[Any]) -> NDArray[np.float32]:
    """Every EVT_* variant uses the BAND (metric-matched) mass."""
    return band_mass(truth_ttc)


__all__ = [
    "ANCHOR_COLUMN",
    "E5",
    "FCWD_ELIGIBLE",
    "NEW_VARIANTS",
    "REFERENCE_VARIANTS",
    "VARIANTS",
    "Variant",
    "design",
    "geometry_anchor",
    "scale_event10",
    "scale_rgb10",
    "stack_columns",
    "stacked_anchors",
    "training_mass",
]
