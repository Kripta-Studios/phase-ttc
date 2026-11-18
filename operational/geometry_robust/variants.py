"""Pre-registered robust-geometry variants (PROTOCOL.md §4) and their design matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from operational.phase_geometry import geometry
from operational.phase_geometry.variants import scaled_phase20
from operational.phase_geometry.weights import band_mass

from . import robust

AUG_MASS_FRACTION = 0.25
PHASE20 = list(range(20))
RGB10 = list(range(10))
G_COLS = list(range(20, 30))
R_SHORT_COLS = list(range(30, 43))
R_LONG_COLS = list(range(43, 56))
ANCHOR_RGB, ANCHOR_AREA, ANCHOR_ROBUST, ANCHOR_ROBUST_LONG = 0, 1, 2, 3


@dataclass(frozen=True)
class Variant:
    """Columns of the full design, anchor column, width use, augmentation and comparator."""

    name: str
    columns: tuple[int, ...]
    anchor: int
    use_width: bool
    augmented: bool
    comparator: str
    role: str


VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in (
        Variant(
            "RGB_GEO_BAND", (*RGB10, *G_COLS), ANCHOR_RGB, True, False, "", "comparator (refit)"
        ),
        Variant("GEO_LEARNED", tuple(G_COLS), ANCHOR_AREA, False, False, "", "comparator (refit)"),
        Variant(
            "RGB_ROBUST",
            (*RGB10, *G_COLS, *R_SHORT_COLS),
            ANCHOR_RGB,
            True,
            False,
            "RGB_GEO_BAND",
            "Test12 eligible",
        ),
        Variant(
            "RGB_ROBUST_AUG",
            (*RGB10, *G_COLS, *R_SHORT_COLS),
            ANCHOR_RGB,
            True,
            True,
            "RGB_GEO_BAND",
            "Test12 eligible",
        ),
        Variant(
            "GEO_ROBUST",
            (*G_COLS, *R_SHORT_COLS),
            ANCHOR_ROBUST,
            False,
            False,
            "GEO_LEARNED",
            "ablation",
        ),
        Variant(
            "GEO_ROBUST_AUG",
            (*G_COLS, *R_SHORT_COLS),
            ANCHOR_ROBUST,
            False,
            True,
            "GEO_LEARNED",
            "FCWD #1",
        ),
        Variant(
            "GEO_ROBUST_AUG_MEM",
            (*G_COLS, *R_LONG_COLS),
            ANCHOR_ROBUST_LONG,
            False,
            True,
            "GEO_LEARNED",
            "FCWD #2",
        ),
    )
}
LEARNED = tuple(VARIANTS)
TEST12_ELIGIBLE = ("RGB_ROBUST", "RGB_ROBUST_AUG")
FCWD_VARIANTS = ("GEO_ROBUST_AUG", "GEO_ROBUST_AUG_MEM")


def full_design(
    features: NDArray[Any],
    width_logits: NDArray[Any],
    g_raw: NDArray[Any],
    r_short: NDArray[Any],
    r_long: NDArray[Any] | None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Full input matrix ``[N, 56]``, anchor matrix ``[N, 4]`` and RGB width logits ``[N, 2]``.

    Without ``r_long`` (Test12) its 13 columns and anchor are zero; no Test12 variant reads them.
    """
    f = np.asarray(features, dtype=np.float32)
    long = np.zeros_like(r_short) if r_long is None else r_long
    x = np.concatenate(
        (
            scaled_phase20(f),
            geometry.learned_inputs(g_raw),
            robust.learned_r(r_short),
            robust.learned_r(long),
        ),
        axis=1,
    )
    clip = geometry.LEARNED_PHASE_CLIP
    phi = np.clip(geometry.raw_column(g_raw, "phi_area"), -clip, clip)
    anchors = np.stack(
        (
            f[:, 0],
            phi.astype(np.float32),
            robust.robust_anchor(r_short),
            robust.robust_anchor(long) if r_long is not None else np.zeros(len(f), np.float32),
        ),
        axis=1,
    )
    return x.astype(np.float32), anchors.astype(np.float32), np.asarray(width_logits, np.float32)


def select(
    variant: Variant, x_full: NDArray[Any], anchors: NDArray[Any], widths: NDArray[Any]
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Inputs, anchor and width base of one variant."""
    x = np.asarray(x_full, dtype=np.float32)[:, list(variant.columns)]
    anchor = np.asarray(anchors, dtype=np.float32)[:, variant.anchor]
    width = (
        np.asarray(widths, dtype=np.float32)
        if variant.use_width
        else np.zeros((len(x), 2), dtype=np.float32)
    )
    return x, anchor, width


def training_mass(truth_ttc: NDArray[Any], augmented: bool) -> NDArray[np.float32]:
    """BAND mass; with augmentation ``0.75*BAND(originals) ++ 0.25*BAND(copies)`` (copies share
    the original targets, so the copy rows follow the original rows)."""
    base = band_mass(truth_ttc).astype(np.float64)
    if not augmented:
        return base.astype(np.float32)
    return np.concatenate(((1.0 - AUG_MASS_FRACTION) * base, AUG_MASS_FRACTION * base)).astype(
        np.float32
    )


__all__ = [
    "FCWD_VARIANTS",
    "LEARNED",
    "TEST12_ELIGIBLE",
    "VARIANTS",
    "Variant",
    "full_design",
    "select",
    "training_mass",
]
