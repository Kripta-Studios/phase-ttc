"""Pre-registered arms (PROTOCOL.md §3) on top of the ``geometry_robust`` design matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from operational.geometry_robust import robust
from operational.geometry_robust import variants as gr

from . import content_scale as cs

C_COLS = list(range(56, 56 + len(cs.C_FIELDS)))
ANCHOR_RGB, ANCHOR_AREA, ANCHOR_ROBUST = gr.ANCHOR_RGB, gr.ANCHOR_AREA, gr.ANCHOR_ROBUST


@dataclass(frozen=True)
class Arm:
    """Columns of the full design, anchor column, width use, augmentation and role."""

    name: str
    columns: tuple[int, ...]
    anchor: int
    use_width: bool
    augmented: bool
    role: str


ARMS: dict[str, Arm] = {
    a.name: a
    for a in (
        Arm(
            "GEO_ROBUST_AUG",
            (*gr.G_COLS, *gr.R_SHORT_COLS),
            ANCHOR_ROBUST,
            False,
            True,
            "baseline to beat (refit)",
        ),
        Arm(
            "GEO_ROBUST_AUG_CS",
            (*gr.G_COLS, *gr.R_SHORT_COLS, *C_COLS),
            ANCHOR_ROBUST,
            False,
            True,
            "primary (eAP and FCWD)",
        ),
        Arm("GEO_LEARNED", tuple(gr.G_COLS), ANCHOR_AREA, False, False, "context (refit)"),
        Arm(
            "RGB_GEO_BAND",
            (*gr.RGB10, *gr.G_COLS),
            ANCHOR_RGB,
            True,
            False,
            "Test12 gate comparator (refit)",
        ),
        Arm(
            "RGB_ROBUST",
            (*gr.RGB10, *gr.G_COLS, *gr.R_SHORT_COLS),
            ANCHOR_RGB,
            True,
            False,
            "context (refit)",
        ),
        Arm(
            "RGB_ROBUST_CS",
            (*gr.RGB10, *gr.G_COLS, *gr.R_SHORT_COLS, *C_COLS),
            ANCHOR_RGB,
            True,
            False,
            "Test12 gate only",
        ),
    )
}
LEARNED = tuple(ARMS)
PRIMARY = "GEO_ROBUST_AUG_CS"
BASELINE = "GEO_ROBUST_AUG"
GATE_ARMS = ("GEO_ROBUST_AUG_CS", "RGB_ROBUST_CS")
GATE_COMPARATOR = "RGB_GEO_BAND"
SENSITIVITY = "CONTENT_FIXED"


def full_design(
    features: NDArray[Any],
    width_logits: NDArray[Any],
    g_raw: NDArray[Any],
    r_short: NDArray[Any],
    c_raw: NDArray[Any],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """``geometry_robust`` design ``[N, 56]`` (no long-memory block) plus C ``[N, 9]``."""
    x, anchors, widths = gr.full_design(features, width_logits, g_raw, r_short, None)
    c = cs.learned_c(c_raw)
    if not np.isfinite(c).all():
        raise ValueError("content inputs must be finite")
    return np.concatenate((x, c), axis=1).astype(np.float32), anchors, widths


def select(
    arm: Arm, x_full: NDArray[Any], anchors: NDArray[Any], widths: NDArray[Any]
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Inputs, anchor and width base of one arm."""
    x = np.asarray(x_full, dtype=np.float32)[:, list(arm.columns)]
    anchor = np.asarray(anchors, dtype=np.float32)[:, arm.anchor]
    width = (
        np.asarray(widths, dtype=np.float32)
        if arm.use_width
        else np.zeros((len(x), 2), dtype=np.float32)
    )
    return x, anchor, width


def content_fixed_phase(r_short: NDArray[Any], c_raw: NDArray[Any]) -> NDArray[np.float64]:
    """CONTENT_FIXED: ``phi_content`` where the content estimate is valid, else ``phi_robust``."""
    valid = cs.raw_c_column(c_raw, "valid") > 0
    return np.where(
        valid, cs.raw_c_column(c_raw, "phi_content"), robust.raw_r_column(r_short, "phi_robust")
    )


training_mass = gr.training_mass

__all__ = [
    "ARMS",
    "BASELINE",
    "C_COLS",
    "GATE_ARMS",
    "GATE_COMPARATOR",
    "LEARNED",
    "PRIMARY",
    "SENSITIVITY",
    "Arm",
    "content_fixed_phase",
    "full_design",
    "select",
    "training_mass",
]
