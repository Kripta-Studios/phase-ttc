"""Detached float64 reporting of the separate registered supervised objective terms."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from e_jepa_ttc.models.phase_distribution_v11 import crps, ttc_to_phase


@torch.no_grad()
def supervised_loss_diagnostics(
    logits: Tensor,
    emitted_ttc: Tensor,
    target_phase: Tensor,
    edges: Tensor,
    global_mass: Tensor,
    universe_rows: int,
    objective: Tensor,
) -> dict[str, object]:
    """Observe the same emitted TTC and global masses without modifying the backward objective."""
    error = (ttc_to_phase(emitted_ttc.detach()) - target_phase.detach()).abs()
    distribution_loss = crps(logits.detach(), target_phase.detach(), edges)
    weight = universe_rows * global_mass.detach().double()
    record: dict[str, object] = {
        "emitted_phase_l1_weighted": float((weight * error.double()).mean()),
        "crps_weighted": float((weight * distribution_loss.double()).mean()),
        "registered_objective": float(objective.detach()),
        "global_mass": global_mass.detach().double().cpu().tolist(),
        "universe_rows": universe_rows,
        "microbatch_rows": len(logits),
        "component_reporting_precision": "float64 aggregation of registered float32 loss terms",
        "mass_renormalized_within_batch": False,
    }
    for key in ("emitted_phase_l1_weighted", "crps_weighted", "registered_objective"):
        value = record[key]
        if isinstance(value, float) and not math.isfinite(value):
            record[key] = None
    return record
