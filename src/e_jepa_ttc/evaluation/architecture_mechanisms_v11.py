"""Prospective input interventions; no optimizer, target transformation or candidate selection."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from e_jepa_ttc.data.event_context128 import ArchitectureInput, RateNormalization
from e_jepa_ttc.training.stage72_augmentation import time_derangement

MECHANISM_RECIPE = {
    "clock_half": "multiply observed relative clock times by0.5; context tensors unchanged",
    "clock_double": "multiply observed relative clock times by2; context tensors unchanged",
    "same_context": "repeat last observed context in all four chronological slots",
    "zero_events": "zero raw counts, then apply frozen TRAIN normalization and sensor support",
    "zero_geometry": "zero all evidence and validity channels; preserve arm gaps and direct logits",
    "pair_time_permutation": (
        "seed-salted token-only time derangement for geometry and gaps; direct content unchanged"
    ),
    "translation_only": (
        "last observed texture translated horizontally by[-3,-1,1,3]ROI pixels; "
        "no scale change, normalized zero-event padding"
    ),
    "synthetic_target_note": (
        "translation/same-context/zero-event inputs have no invented TTC truth"
    ),
    "clock_target_note": (
        "clock perturbations are sensitivity diagnostics, never supervised augmentations"
    ),
    "selection": "nonselectable evaluation only",
}
MECHANISM_NAMES = (
    "clock_half",
    "clock_double",
    "same_context",
    "zero_events",
    "zero_geometry",
    "pair_time_permutation",
    "translation_only",
)


def zero_event_contexts(sample: ArchitectureInput, normalization: RateNormalization) -> Tensor:
    """Raw zero counts become normalized negative means inside sensor support, not tensor zero."""
    mean = torch.tensor(normalization.mean, dtype=torch.float32, device=sample.contexts.device)
    std = torch.tensor(normalization.std, dtype=torch.float32, device=sample.contexts.device)
    channels = (-mean / std).repeat(4)
    return channels[None, :, None, None].expand_as(sample.contexts).clone() * sample.sensor_support


def mechanism_input(
    sample: ArchitectureInput,
    normalization: RateNormalization,
    kind: str,
    *,
    token: str,
    seed: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Create a deterministic intervention without changing its source tensor or training RNG."""
    contexts, times = sample.contexts.clone(), sample.times.clone()
    kwargs: dict[str, Any] = {}
    if kind in ("clock_half", "clock_double"):
        times *= 0.5 if kind == "clock_half" else 2.0
    elif kind == "same_context":
        contexts = contexts[-1:].expand_as(contexts).clone()
    elif kind == "zero_events":
        contexts = zero_event_contexts(sample, normalization)
    elif kind == "zero_geometry":
        kwargs["diagnostic_zero_geometry"] = True
    elif kind == "pair_time_permutation":
        kwargs["diagnostic_time_permutation"] = torch.tensor(
            [time_derangement(token, seed)],
            device=contexts.device,
        )
    elif kind == "translation_only":
        contexts = zero_event_contexts(sample, normalization)
        texture = sample.contexts[-1]
        for index, shift in enumerate((-3, -1, 1, 3)):
            if shift < 0:
                contexts[index, :, :, :shift] = texture[:, :, -shift:]
            else:
                contexts[index, :, :, shift:] = texture[:, :, :-shift]
        contexts *= sample.sensor_support
    else:
        raise ValueError("unknown prospectively fixed mechanism intervention")
    return contexts, times, kwargs
