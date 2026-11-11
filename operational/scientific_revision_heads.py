"""Equal-capacity final-head controls; frozen upstream features and unchanged loss."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional

from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, emitted_phase, phase_to_ttc
from operational.final_refit.training import PhaseOnlyFinal


class MatchedHead(PhaseOnlyFinal):
    """Change only event access or the residual coordinate of the original head."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in {"phase_true", "phase_zero_event", "inverse_residual"}:
            raise ValueError("Unknown matched head mode")
        self.mode = mode

    def inputs(self, features: Tensor) -> Tensor:
        """Keep all timing/support fields; ZERO removes the complete event descriptor."""
        selected = super().inputs(features)
        if self.mode == "phase_zero_event":
            selected[:, 10:] = 0
        elif self.mode == "inverse_residual":
            for start in (0, 10):
                phase = (selected[:, start : start + 5] * 0.03).clamp(MIN_PHASE, MAX_PHASE)
                selected[:, start : start + 5] = -torch.expm1(-phase) / 0.03
        return selected

    def forward(self, features: Tensor, rgb_width_logits: Tensor) -> dict[str, Tensor]:
        """Return the common phase-domain objective fields for every arm."""
        if self.mode != "inverse_residual":
            return super().forward(features, rgb_width_logits)
        correction = self.output(self.mlp(self.inputs(features)))
        anchor = -torch.expm1(-features[:, 160].clamp(MIN_PHASE, MAX_PHASE)) / 0.1
        rate = anchor + 0.3 * correction[:, 0]
        lower = -float(np.expm1(-MIN_PHASE)) / 0.1
        upper = -float(np.expm1(-MAX_PHASE)) / 0.1
        location = -torch.log1p(-0.1 * rate.clamp(lower, upper))
        widths = 0.03 * functional.softplus(rgb_width_logits + correction[:, 1:])
        return {
            "raw_location": location,
            "point_phase": emitted_phase(location),
            "ttc": phase_to_ttc(location),
            "q10": (location - widths[:, 0]).clamp(MIN_PHASE, MAX_PHASE),
            "q90": (location + widths[:, 1]).clamp(MIN_PHASE, MAX_PHASE),
            "correction": correction,
        }


def source_training_mask(values: dict[str, Any]) -> np.ndarray:
    """Admit only the original source-domain OOF population, never transfer labels."""
    selected = np.asarray(values["domain"]) == "eap"
    if int(selected.sum()) != 88744:
        raise ValueError("Expected 88744 original source-domain OOF rows")
    if len(set(values["group_id"][selected])) != 40:
        raise ValueError("Expected 40 original source recordings")
    if not all(str(t).startswith("eap:") for t in values["sample_token"][selected]):
        raise ValueError("Source feature identities differ")
    return selected
