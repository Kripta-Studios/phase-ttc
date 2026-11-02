from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from e_jepa_ttc.detection.fixed_state_trial_executor import (
    capture_state,
    exact_training_operation,
    restore_state,
    verify_frozen_operation_source,
)

ROOT = Path(__file__).parents[2]
FROZEN_PRODUCER = (
    ROOT / "tests/fixtures/detection_candidate_sources/r0_train_detection_campaign.py.txt"
)


class Stateful:
    def __init__(self) -> None:
        self.value = 3

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.value = state["value"]


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, images, calibrations, targets, sizes):
        del calibrations, targets, sizes
        return {"prediction": self.linear(images)}


class Criterion(torch.nn.Module):
    weight_dict = {"a": 2.0, "b": 3.0}

    def forward(self, outputs, targets):
        target = targets[0]
        return {
            "a": (outputs["prediction"] - target).square().mean(),
            "ignored": outputs["prediction"].sum() * 0,
            "b": outputs["prediction"].abs().mean(),
        }


def test_reset_reproduces_exact_real_cpu_update() -> None:
    torch.manual_seed(7)
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler, sampler = Stateful(), Stateful()
    snapshot = capture_state(model, optimizer, scheduler, sampler)
    batch = (torch.tensor([[1.0, 2.0]]), None, [torch.tensor([[0.5]])], None)
    first = exact_training_operation(
        model=model,
        criterion=Criterion(),
        optimizer=optimizer,
        batch=batch,
        clip=1.0,
        finite_validator=lambda *args: None,
    )
    scheduler.value = sampler.value = 99
    random.random()
    np.random.rand()
    torch.rand(1)
    restore_state(snapshot, model, optimizer, scheduler, sampler)
    second = exact_training_operation(
        model=model,
        criterion=Criterion(),
        optimizer=optimizer,
        batch=batch,
        clip=1.0,
        finite_validator=lambda *args: None,
    )
    assert scheduler.value == sampler.value == 3
    assert torch.equal(first["loss"], second["loss"])
    assert torch.equal(first["gradient_norm_before_clip"], second["gradient_norm_before_clip"])
    for name in first["gradients_before_clip"]:
        assert torch.equal(
            first["gradients_before_clip"][name], second["gradients_before_clip"][name]
        )
        assert torch.equal(
            first["parameter_step_deltas"][name], second["parameter_step_deltas"][name]
        )
        assert torch.equal(
            first["gradients_after_clip"][name], second["gradients_after_clip"][name]
        )


def test_frozen_inline_training_ast_matches_executor_core() -> None:
    proof = verify_frozen_operation_source(str(FROZEN_PRODUCER))
    assert proof["producer_sha256"] == (
        "659dfe453aaf574f67b7fb037a40bf56d2e77783e86da8b20a69dc8e07b326fc"
    )
    assert len(proof["inline_span_sha256"]) == 64


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("loss = loss + term", "loss = loss - term"),
        ("value * criterion.weight_dict[key]", "value * 1.0"),
        ("optimizer.zero_grad(set_to_none=True)", "optimizer.zero_grad(set_to_none=False)"),
        ("error_if_nonfinite=True", "error_if_nonfinite=False"),
        ("config.optimization.gradient_clip_norm", "1.0"),
    ],
)
def test_ast_proof_rejects_mathematical_mutations(tmp_path: Path, old: str, new: str) -> None:
    source = FROZEN_PRODUCER.read_text(encoding="utf-8")
    assert old in source
    changed = tmp_path / "mutated.py"
    changed.write_text(source.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_frozen_operation_source(str(changed))
