"""Pinhole cuboid-edge inversion for diagnostics, not an admitted detector change."""

from __future__ import annotations

import numpy as np


def depth_from_cuboid_edges(
    camera_offsets: np.ndarray,
    intrinsic: np.ndarray,
    center_uv: np.ndarray,
    distances_lrtb: np.ndarray,
) -> np.ndarray:
    """Recover four optical depths from complete projected cuboid edge distances.

    Inputs contain rotated offsets from the cuboid center, not its translation.
    For pixel-axis offset numerator a_i and positive edge distance t,
    containment requires z >= a_i / t - delta_z_i for every corner. The first
    touching corner gives max_i(a_i/t-delta_z_i). Clipped edges are NOT equalities
    and must be excluded by the caller. Predicted geometry need not be accurate.
    """
    offsets = np.asarray(camera_offsets, dtype=np.float64)
    k = np.asarray(intrinsic, dtype=np.float64)
    uv = np.asarray(center_uv, dtype=np.float64)
    edges = np.asarray(distances_lrtb, dtype=np.float64)
    if offsets.shape != (8, 3) or k.shape != (3, 3) or uv.shape != (2,) or edges.shape != (4,):
        raise ValueError("Expected eight offsets, 3x3 intrinsic, UV and four edge distances")
    if not all(np.isfinite(v).all() for v in (offsets, k, uv, edges)) or (edges <= 0).any():
        raise ValueError("Finite geometry and positive edge distances required")
    if not np.allclose(k[2], [0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError("Pinhole homogeneous row required")
    numerators = offsets @ k[:2].T - offsets[:, 2:3] * uv[None]
    signed = np.stack(
        (-numerators[:, 0], numerators[:, 0], -numerators[:, 1], numerators[:, 1]), axis=1
    )
    return (signed / edges[None] - offsets[:, 2:3]).max(axis=0)
