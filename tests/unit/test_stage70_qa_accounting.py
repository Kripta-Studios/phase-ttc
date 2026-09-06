"""Accounting fixtures only: these tests execute no optimizer or model fit."""

import importlib.util
import json
import os
from pathlib import Path

import psutil
import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.training.stage70_update_ledger import UpdateLedger

PLUGIN = Path(__file__).resolve().parents[2] / "scripts/qa_support/stage70_optimizer_audit.py"
SPEC = importlib.util.spec_from_file_location("qa_audit_fixture", PLUGIN)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AccountingFixture:
    """Weak-referenceable stand-in; never calls an optimizer."""


def test_shared_quota_confirmation_uncertainty_and_no_overrun(tmp_path):
    path = tmp_path / "audit.sqlite"
    AUDIT.initialize(path, 2, "a" * 64)
    first, second = AUDIT.Recorder(path, "a" * 64), AUDIT.Recorder(path, "a" * 64)
    one, two = AccountingFixture(), AccountingFixture()
    first.before(one)
    first.after(one)
    second.before(two)  # Unconfirmed accounting intent simulates interrupted work.
    with pytest.raises(RuntimeError, match="exhausted"):
        first.before(one)
    record = AUDIT.snapshot(path)
    assert (record["charged"], record["confirmed"], record["uncertain"]) == (2, 1, 1)
    assert record["optimizer_instances_started"] == 2
    with pytest.raises(ValueError, match="identity"):
        AUDIT.Recorder(path, "b" * 64)
    with pytest.raises(ValueError, match="fresh"):
        AUDIT.initialize(path, 2, "a" * 64)


def test_bounded_qa_is_reserved_and_not_a_fake_model_endpoint(tmp_path):
    (tmp_path / "ACTIVE_OWNER.json").write_text(
        json.dumps({"pid": os.getpid(), "created": psutil.Process().create_time()})
    )
    ledger = UpdateLedger(tmp_path / "ledger.json", tmp_path)
    ledger.reserve_technical_audit("fixture", 3000, "a" * 64)
    with pytest.raises(RuntimeError, match="3000"):
        ledger.reserve_technical_audit("overflow", 1, "b" * 64)
    assert ledger.totals()["reserved_remaining_updates"] == 3000
    assert ledger.totals()["optimizer_endpoints_complete"] == 0
    receipt = {
        "identity": "a" * 64,
        "limit": 3000,
        "charged": 3,
        "confirmed": 2,
        "uncertain": 1,
        "optimizer_instances_started": 2,
        "all_processes_closed": True,
    }
    path = tmp_path / "closed.json"
    path.write_text(json.dumps({**receipt, "sha256": object_digest(receipt)}))
    ledger.settle_technical_audit("fixture", path)
    totals = ledger.totals()
    assert totals["charged_optimizer_attempts"] == 3
    assert totals["confirmed_optimizer_updates"] == 2
    assert totals["uncertain_optimizer_attempts"] == 1
    assert totals["optimizer_fits_started"] == 2
    assert totals["optimizer_endpoints_complete"] == 0
    assert totals["reserved_remaining_updates"] == 0
    with pytest.raises(ValueError, match="settled"):
        ledger.settle_technical_audit("fixture", path)


def test_unclosed_or_overquota_receipt_cannot_release_reservation(tmp_path):
    (tmp_path / "ACTIVE_OWNER.json").write_text(
        json.dumps({"pid": os.getpid(), "created": psutil.Process().create_time()})
    )
    ledger = UpdateLedger(tmp_path / "ledger.json", tmp_path)
    ledger.reserve_technical_audit("fixture", 2, "a" * 64)
    receipt = {
        "identity": "a" * 64,
        "limit": 2,
        "charged": 3,
        "confirmed": 3,
        "uncertain": 0,
        "optimizer_instances_started": 1,
        "all_processes_closed": False,
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({**receipt, "sha256": object_digest(receipt)}))
    with pytest.raises(ValueError, match="closed-process"):
        ledger.settle_technical_audit("fixture", path)
    receipt["all_processes_closed"] = True
    path.write_text(json.dumps({**receipt, "sha256": object_digest(receipt)}))
    with pytest.raises(ValueError, match="accounting"):
        ledger.settle_technical_audit("fixture", path)
    assert ledger.totals()["reserved_remaining_updates"] == 2
    assert ledger.totals()["charged_optimizer_attempts"] == 0
