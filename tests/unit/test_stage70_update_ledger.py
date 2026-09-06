"""Synthetic accounting events exercise resource limits and interrupted publication."""

import json
import os

import psutil
import pytest

from e_jepa_ttc.training.stage70_update_ledger import FitSpec, UpdateLedger


@pytest.fixture
def ledger(tmp_path):
    (tmp_path / "ACTIVE_OWNER.json").write_text(
        json.dumps({"pid": os.getpid(), "created": psutil.Process().create_time()})
    )
    return UpdateLedger(tmp_path / "ledger.json", tmp_path)


def spec(job="fixture", endpoint=10, technical=False):
    return FitSpec(job, "SYNTHETIC_TEST", "DIRECT", 0, 7, endpoint, "a" * 64, "b" * 64, technical)


def step(ledger):
    ledger.begin_step("fixture")
    ledger.confirm_step("fixture")


def test_entire_remaining_budget_reserved_and_failed_reservation_rolls_back(ledger):
    ledger.reserve(spec(endpoint=899999))
    ledger.reserve(spec("last", endpoint=1))
    with pytest.raises(RuntimeError, match="900000"):
        ledger.reserve(spec("excess", endpoint=1))
    assert ledger.totals()["reserved_remaining_updates"] == 900000
    reloaded = UpdateLedger(ledger.path, ledger.owner_root)
    assert reloaded.totals() == ledger.totals()
    with pytest.raises(ValueError, match="changed"):
        ledger.reserve(spec(endpoint=8000))


def test_technical_limit_independent_of_scientific_allowance(ledger):
    with pytest.raises(RuntimeError, match="3000"):
        ledger.reserve(spec(endpoint=3001, technical=True))
    assert ledger.totals()["reserved_remaining_updates"] == 0


def test_recovery_charges_known_replay_and_retains_uncertain_attempt(ledger, tmp_path):
    ledger.reserve(spec())
    step(ledger)
    checkpoint = tmp_path / "checkpoint_1.fixture"
    checkpoint.write_bytes(b"synthetic accounting fixture, not a trained endpoint")
    ledger.checkpoint("fixture", checkpoint, 1)
    step(ledger)
    ledger.begin_step("fixture")  # Simulated process death with an in-flight update.
    reloaded = UpdateLedger(ledger.path, ledger.owner_root)
    restored = reloaded.recover("fixture")
    assert restored is not None and restored["path"] == str(checkpoint.resolve())
    assert reloaded.totals()["confirmed_optimizer_updates"] == 2
    assert reloaded.totals()["uncertain_optimizer_attempts"] == 1
    assert reloaded.totals()["charged_optimizer_attempts"] == 3
    assert reloaded.totals()["reserved_remaining_updates"] == 9
    assert reloaded.begin_step("fixture") == 2


def test_endpoint_requires_latest_complete_prescribed_checkpoint(ledger, tmp_path):
    ledger.reserve(spec(endpoint=2))
    step(ledger)
    partial = tmp_path / "partial.fixture"
    partial.write_bytes(b"partial")
    ledger.checkpoint("fixture", partial, 1)
    with pytest.raises(ValueError, match="partial"):
        ledger.complete("fixture")
    step(ledger)
    with pytest.raises(ValueError, match="partial"):
        ledger.complete("fixture")
    final = tmp_path / "final.fixture"
    final.write_bytes(b"final accounting fixture")
    ledger.checkpoint("fixture", final, 2)
    ledger.complete("fixture")
    assert ledger.totals()["optimizer_endpoints_complete"] == 1
    with pytest.raises(ValueError, match="endpoint"):
        ledger.begin_step("fixture")
    with pytest.raises(ValueError, match="cannot restart"):
        ledger.recover("fixture")


def test_checkpoint_substitution_and_foreign_writer_are_rejected(ledger, tmp_path):
    ledger.reserve(spec())
    step(ledger)
    checkpoint = tmp_path / "state.fixture"
    checkpoint.write_bytes(b"state")
    ledger.checkpoint("fixture", checkpoint, 1)
    checkpoint.write_bytes(b"other")
    with pytest.raises(ValueError, match="identity changed"):
        ledger.recover("fixture")
    (tmp_path / "ACTIVE_OWNER.json").write_text(json.dumps({"pid": -1, "created": 0}))
    with pytest.raises(RuntimeError, match="exclusive campaign owner"):
        ledger.begin_step("fixture")
