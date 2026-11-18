"""Causal truncation of PHASE17 slot histories to the K most recent observations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

GAP_FIELD = 2  # clock_features index: anchor minus the previous kept anchor (0 for the first)
HISTORY_KEYS = ("features", "timing", "valid")


def truncate_history(
    features: np.ndarray, timing: np.ndarray, valid: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only the ``k`` most recent valid slots of each row.

    Slots are ordered oldest -> newest with the current observation last, and valid
    slots form a contiguous suffix (the released contract). Dropped slots become
    invalid with exactly-zero features and timing. On the oldest kept slot the
    inter-anchor gap is reset to 0, as ``clock_features`` does for the first
    observation of a history. The output for a row depends only on that row's own
    past slots, so the operator is causal. ``k >= length`` is the identity.
    """
    if k < 1:
        raise ValueError("K must be at least one slot")
    features = np.asarray(features)
    timing = np.asarray(timing)
    valid = np.asarray(valid, dtype=np.bool_)
    if (
        features.ndim != 3
        or valid.shape != features.shape[:2]
        or timing.shape != (*features.shape[:2], 4)
    ):
        raise ValueError("Expected features [N,L,F], timing [N,L,4], valid [N,L]")
    if len(valid) and (not valid[:, -1].all() or np.any(valid[:, :-1] & ~valid[:, 1:])):
        raise ValueError("Histories must have a valid current slot and contiguous valid suffix")
    length = valid.shape[1]
    if k >= length:
        return features.copy(), timing.copy(), valid.copy()
    keep = valid & (np.arange(length)[None, :] >= length - k)
    out_features = np.where(keep[..., None], features, 0).astype(features.dtype, copy=False)
    out_timing = np.where(keep[..., None], timing, 0).astype(timing.dtype, copy=False)
    first = keep.argmax(axis=1)
    out_timing[np.arange(len(keep)), first, GAP_FIELD] = 0
    return out_features, out_timing, keep


def truncate_mapping(values: Mapping[str, np.ndarray], k: int) -> dict[str, np.ndarray]:
    """Apply :func:`truncate_history` to a cache/fragment mapping; copy other fields."""
    features, timing, valid = truncate_history(
        values["features"], values["timing"], values["valid"], k
    )
    result = {key: value for key, value in values.items() if key not in HISTORY_KEYS}
    result.update(features=features, timing=timing, valid=valid)
    return result


__all__ = ["GAP_FIELD", "truncate_history", "truncate_mapping"]
