"""Detached diagnostic terms preserve the actual objective, gradients and population weights."""

import pytest
import torch

from e_jepa_ttc.models.phase_distribution_v11 import (
    phase_edges,
    phase_to_finite_ttc,
    quantile,
    weighted_training_loss,
)
from e_jepa_ttc.training.architecture_loss_diagnostics_v11 import supervised_loss_diagnostics


def test_reporting_does_not_modify_gradient_rng_or_renormalize_weights():
    logits = torch.linspace(-1, 1, 216).reshape(2, 108).clone().requires_grad_()
    targets = torch.tensor([-0.02, 0.04])
    mass = torch.tensor([0.003, 0.007], dtype=torch.float64)
    edges = phase_edges()
    objective = weighted_training_loss(logits, targets, edges, mass, 100)
    expected = torch.autograd.grad(objective, logits, retain_graph=True)[0]
    rng = torch.get_rng_state()
    record = supervised_loss_diagnostics(
        logits, phase_to_finite_ttc(quantile(logits, edges)), targets, edges, mass, 100, objective
    )
    observed = torch.autograd.grad(objective, logits)[0]
    assert torch.equal(expected, observed) and torch.equal(rng, torch.get_rng_state())
    assert record["global_mass"] == [0.003, 0.007]
    assert record["registered_objective"] == float(objective.detach())
    assert (
        record["emitted_phase_l1_weighted"] + 0.2 * record["crps_weighted"]
    ) / 0.03 == pytest.approx(record["registered_objective"], rel=1e-6)
    assert not record["mass_renormalized_within_batch"]
