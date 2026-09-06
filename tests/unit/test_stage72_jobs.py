"""Full-phase reservations and immutable publication; no scientific optimizer steps."""

import json
from dataclasses import replace

import pytest
import torch

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.training.stage72_jobs import (
    FitJob,
    _publish_weights,
    require_scientific_lock,
    reserve_composite_job,
)
from e_jepa_ttc.training.stage73_factorial import PlannedFit


def job(name="S73-PPS-A"):
    stage = "S72" if name.startswith("S72") else "S73"
    return FitJob(
        PlannedFit(stage, name, 0, 7),
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
        ("train",),
        ("outer",),
        "f" * 64,
    )


class ReservationLedger:
    def __init__(self, budget):
        self.budget = budget
        self.jobs = {}

    def reserve(self, spec):
        if spec.job_id in self.jobs:
            assert self.jobs[spec.job_id] == spec
            return
        if sum(s.endpoint for s in self.jobs.values()) + spec.endpoint > self.budget:
            raise RuntimeError("entire remaining job exceeds reserved budget")
        self.jobs[spec.job_id] = spec


def test_pretext_plus_full_supervision_are_reserved_before_fit_and_resume_idempotently():
    ledger = ReservationLedger(8000)
    with pytest.raises(RuntimeError, match="entire remaining"):
        reserve_composite_job(ledger, job(), "1" * 64)
    assert sum(spec.endpoint for spec in ledger.jobs.values()) == 8000
    ledger.budget = 11000
    encoder, task = reserve_composite_job(ledger, job(), "1" * 64)
    assert encoder.endpoint == 3000 and task.endpoint == 8000
    assert not encoder.technical and not task.technical
    assert encoder.recipe_sha256 == task.recipe_sha256 == job().identity
    assert encoder.training_view_sha256 != task.training_view_sha256
    assert reserve_composite_job(ledger, job(), "1" * 64) == (encoder, task)
    assert sum(spec.endpoint for spec in ledger.jobs.values()) == 11000


def test_scratch_and_compute_control_cannot_inherit_pretext():
    ledger = ReservationLedger(20000)
    with pytest.raises(ValueError, match="inherit"):
        reserve_composite_job(ledger, job("S72-DIRECT"), "1" * 64)
    assert not ledger.jobs
    pretext, task = reserve_composite_job(ledger, job("S73-PPS-EXTRA"), None)
    assert pretext is None and task.endpoint == 11000


def test_conditional_fit_requires_a_decision_and_disjoint_groups():
    with pytest.raises(ValueError, match="decision hash"):
        replace(job(), plan=PlannedFit("S74", "S73-PPS-A", 0, 13))
    with pytest.raises(ValueError, match="overlap"):
        replace(job(), excluded_groups=("train",))
    with pytest.raises(ValueError, match="conditioned"):
        replace(job(), stage_authorization_sha256="1" * 64)


def test_incomplete_lock_cannot_enable_scientific_execution(tmp_path):
    path = tmp_path / "TRAINING_LOCK.json"
    value = dict(
        kind="STAGE70_76_TRAINING_LOCK",
        tracked_worktree_clean_at_freeze=True,
        implementation_complete_before_first_fit=True,
        scientific_updates_at_freeze=0,
        code_commit="a" * 40,
        preflight_receipts={},
    )
    signature = object_digest(value)
    path.write_text(json.dumps({**value, "artifact_sha256": signature}))
    with pytest.raises(ValueError, match="mandatory"):
        require_scientific_lock(path, signature)
    with pytest.raises(ValueError, match="hash changed"):
        require_scientific_lock(path, "b" * 64)


def test_orphan_inference_publication_is_verified_not_retrained_or_overwritten(tmp_path):
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "weights.pt"
    first = _publish_weights(model, path)
    assert _publish_weights(model, path) == first
    with torch.no_grad():
        model.weight.add_(1)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="orphan"):
        _publish_weights(model, path)
    assert path.read_bytes() == before
