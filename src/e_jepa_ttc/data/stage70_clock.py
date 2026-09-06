"""Integer-clock contracts for the four E128-T8 physical contexts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContextClock:
    """Physical intervals and midpoint offsets from the documented label anchor."""

    bin_edges_us: tuple[tuple[int, ...], ...]
    midpoint_twice_relative_us: tuple[int, ...]
    label_anchor_event_us: int
    frame_to_event_offsets_us: tuple[int, int]
    gap_us: int

    def midpoint_seconds(self) -> np.ndarray:
        """Cast only after integer subtraction, preserving large-clock precision."""
        return np.asarray(self.midpoint_twice_relative_us, dtype=np.float64) / 2_000_000


def context_clock(
    windows: Sequence[Sequence[int]],
    frames: Sequence[int],
    annotation_timestamp_us: int,
) -> ContextClock:
    """Validate front-synchronized windows and construct ordered physical contexts.

    Source semantics must separately establish that the TTC label belongs to the
    later selected frame. This function checks numeric consistency, not provenance.
    It never interprets a sample-token suffix as an event timestamp.
    """
    quantities = [v for pair in windows for v in pair] + list(frames) + [annotation_timestamp_us]
    if any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in quantities
    ):
        raise ValueError("clock values must be integer microseconds")
    windows = tuple(tuple(int(v) for v in pair) for pair in windows)
    frames = tuple(int(v) for v in frames)
    if len(windows) != 2 or any(len(pair) != 2 for pair in windows) or len(frames) != 2:
        raise ValueError("exactly two windows and two frame timestamps required")
    if frames[1] <= frames[0] or int(annotation_timestamp_us) != frames[1]:
        raise ValueError("label timestamp must equal the later selected frame")
    if any(end - start < 8 for start, end in windows):
        raise ValueError("eight positive-duration bins required per window")
    offsets = (frames[0] - windows[0][1], frames[1] - windows[1][1])
    if abs(offsets[1] - offsets[0]) > 5:
        raise ValueError("frame/event offset drift exceeds the pinned five-microsecond bound")
    anchor = windows[1][1]
    if any(end > anchor for _, end in windows):
        raise ValueError("observation extends past the label anchor")
    edges = tuple(tuple(start + (end - start) * i // 8 for i in range(9)) for start, end in windows)
    midpoints = tuple(
        (edge[left] - anchor) + (edge[right] - anchor)
        for edge in edges
        for left, right in ((0, 4), (4, 8))
    )
    if any(b <= a for a, b in zip(midpoints, midpoints[1:], strict=False)):
        raise ValueError("four context midpoints must be strictly ordered without sorting")
    return ContextClock(edges, midpoints, anchor, offsets, windows[1][0] - windows[0][1])
