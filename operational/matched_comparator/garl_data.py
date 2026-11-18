"""Garl-TTC visible-height pairs aligned to the study rows (TRAIN40 OOF order, Test12 tokens).

``EVENT`` comes from the frozen predictions of the released event-only checkpoint (TRAIN40
fragments with heights, bound to the pinned final NPZ; Test12 CSV with heights).  ``FULL`` comes
from ``garl_runner`` fragments written by this study.  Missing or rejected queries get NaN
heights (they become invalid Garl features and official failures for the Garl-alone arms).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .sources import ENV_V12_ARTIFACTS, location_root, verified

EVENT_FRAGMENTS = "train40_system_20261005/garl_train_predictions"
SOURCES = ("EVENT", "FULL")


def _align(tokens: NDArray[Any], known: NDArray[Any], heights: NDArray[Any]) -> NDArray[np.float64]:
    position = {str(t): i for i, t in enumerate(known)}
    out = np.full((len(tokens), 2), np.nan)
    for row, token in enumerate(np.asarray(tokens).astype(str)):
        index = position.get(token)
        if index is not None:
            out[row] = heights[index]
    return out


def event_train40(tokens: NDArray[Any]) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Event-only heights for TRAIN40 rows, verified against the pinned TTC vector."""
    stored = np.load(verified("garl_event_train40_predictions"), allow_pickle=False)
    ttc, ordinals = stored["ttc"], stored["ordinals"]
    heights = np.full((len(ttc), 2), np.nan, dtype=np.float32)
    directory = location_root(ENV_V12_ARTIFACTS) / EVENT_FRAGMENTS
    count = 0
    for path in sorted(directory.glob("batch_*.npz")):
        with np.load(path, allow_pickle=False) as part:
            heights[part["ordinals"]] = part["heights"]
            count += 1
    if not np.array_equal(ordinals, np.arange(len(ttc))) or np.isnan(heights).any():
        raise ValueError("Event-only TRAIN40 fragments do not cover every ordinal")
    with np.errstate(divide="ignore", invalid="ignore"):
        rebuilt = 0.1 / (1.0 - heights[:, 0] / heights[:, 1])  # float32, as the frozen run
    if not np.array_equal(rebuilt, ttc, equal_nan=True):
        raise ValueError("Event-only fragments disagree with the pinned TTC vector")
    rows = pd.read_parquet(verified("train40_rows"), columns=["sample_token"])
    known = rows["sample_token"].astype(str).to_numpy()
    info = {"fragments": count, "rows": len(ttc), "ttc_bit_identical_to_pinned": True}
    return _align(tokens, known, heights.astype(np.float64)), info


def event_test12(tokens: NDArray[Any]) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Event-only heights on Test12 tokens from the frozen label-free CSV."""
    frame = pd.read_csv(verified("garl_event_test12_predictions"))
    heights = np.asarray(
        [json.loads(h) if isinstance(h, str) else [np.nan, np.nan] for h in frame["heights"]],
        dtype=np.float64,
    )
    info = {"rows": len(frame), "status_counts": frame["status"].value_counts().to_dict()}
    return _align(tokens, frame["sample_token"].astype(str).to_numpy(), heights), info


def runner_heights(
    directory: Path, population: str, tokens: NDArray[Any], model: str
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Heights written by ``garl_runner`` (``model`` = ``full`` or ``event``)."""
    from .garl_runner import load_fragments

    loaded = load_fragments(directory, population)
    info = {
        "rows": len(loaded["sample_token"]),
        "failed": len(loaded["failed_token"]),
        "manifest": (directory / f"{population}_MANIFEST.json").exists(),
    }
    return _align(tokens, loaded["sample_token"], loaded[f"{model}_heights"]), info


def heights_for(
    source: str, population: str, tokens: NDArray[Any], runner_dir: Path
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Dispatch: ``source`` in SOURCES, ``population`` in {train40, test12}."""
    if source == "EVENT":
        return event_train40(tokens) if population == "train40" else event_test12(tokens)
    if source == "FULL":
        # Amendment 1: on TRAIN40 the FULL model was run only on the S8 subsample.
        prefix = "train40s8" if population == "train40" else population
        return runner_heights(runner_dir, prefix, tokens, "full")
    raise ValueError(f"Unknown Garl source {source}")


__all__ = ["SOURCES", "event_test12", "event_train40", "heights_for", "runner_heights"]
