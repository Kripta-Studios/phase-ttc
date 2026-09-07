"""Cross-campaign accounting without executing an optimizer or touching real ledgers."""

import json

import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.training.stage70_update_ledger import FitSpec, UpdateLedger


def predecessor(tmp_path, *, charged=600000, technical=2000, remaining=0, terminal=True):
    parent = {
        "version": 1,
        "pending": None,
        "ridge_fits": [],
        "jobs": {
            "old": {
                "spec": {"technical": False},
                "charged": charged,
                "confirmed": charged,
                "uncertain": 0,
                "remaining": remaining,
                "update": charged,
                "status": "COMPLETE",
            },
            "qa": {
                "spec": {"technical": True},
                "charged": technical,
                "confirmed": technical,
                "uncertain": 0,
                "remaining": 0,
                "update": technical,
                "status": "COMPLETE",
            },
        },
    }
    ledger = tmp_path / "parent.json"
    atomic_json(ledger, {**parent, "sha256": object_digest(parent)})
    execution = tmp_path / "terminal.json"
    atomic_json(
        execution,
        {
            "status": "AUTHORIZED_GRAPH_TERMINAL" if terminal else "RUNNING",
            "stages": {s: {} for s in ("S72", "S73", "S74", "S75", "S76")},
            "update_ledger": binding(ledger),
            "optimizer_totals": {"charged_optimizer_attempts": charged + technical},
        },
    )
    return binding(execution)


def local_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(UpdateLedger, "_assert_owner", lambda self: None)
    return UpdateLedger(tmp_path / "local.json", tmp_path / "isolated_owner")


def envelope(tmp_path):
    record = {
        "kind": "PROSPECTIVE_USER_AUTHORIZED_SHARED_BUDGET_ENVELOPE",
        "maximum_new_optimizer_updates": 1_900_000,
        "original_campaign_ceiling_reserved": 900_000,
        "expanded_technical_and_recovery_updates_reserved": 10_000,
        "predecessor_terminal_required": True,
        "all_job_remaining_updates_must_be_reserved_before_start": True,
        "original_frozen_recipe_or_lock_modified": False,
    }
    record["artifact_sha256"] = object_digest(record)
    path = tmp_path / "envelope.json"
    atomic_json(path, record)
    return binding(path)


def test_expanded_whole_fit_reservation_includes_original_charges(tmp_path, monkeypatch):
    ledger = local_ledger(tmp_path, monkeypatch)
    ledger.bind_capacity_envelope(envelope(tmp_path))
    ledger.inherit_closed_campaign(predecessor(tmp_path))
    ledger.reserve(FitSpec("first", "S72", "DIRECT", 0, 7, 800000, "a" * 64, "b" * 64))
    with pytest.raises(RuntimeError, match="1900000"):
        ledger.reserve(FitSpec("second", "S72", "DIRECT", 1, 7, 500000, "a" * 64, "b" * 64))
    assert set(ledger.state["jobs"]) == {"first"}
    assert local_ledger(tmp_path, monkeypatch).state == ledger.state


def test_expanded_qa_allowance_is_additional_but_still_bounded(tmp_path, monkeypatch):
    ledger = local_ledger(tmp_path, monkeypatch)
    ledger.bind_capacity_envelope(envelope(tmp_path))
    ledger.inherit_closed_campaign(predecessor(tmp_path, technical=2900))
    ledger.reserve_technical_audit("expanded", 10000, "a" * 64)
    with pytest.raises(RuntimeError, match="10000"):
        ledger.reserve_technical_audit("extra", 1, "b" * 64)


def test_expanded_envelope_cannot_be_added_after_reservations_or_changed(tmp_path, monkeypatch):
    ledger = local_ledger(tmp_path, monkeypatch)
    source = envelope(tmp_path)
    ledger.reserve_technical_audit("prior", 1, "a" * 64)
    with pytest.raises(ValueError, match="before any local"):
        ledger.bind_capacity_envelope(source)
    other = tmp_path / "other"
    other.mkdir()
    ledger2 = local_ledger(other, monkeypatch)
    ledger2.bind_capacity_envelope(source)
    record = json.loads((tmp_path / "envelope.json").read_text())
    record["maximum_new_optimizer_updates"] += 1
    atomic_json(tmp_path / "envelope.json", record)
    with pytest.raises(ValueError):
        local_ledger(other, monkeypatch)


def test_shared_ceiling_rejects_whole_reservation_without_faking_old_local_fits(
    tmp_path, monkeypatch
):
    ledger = local_ledger(tmp_path, monkeypatch)
    ledger.inherit_closed_campaign(predecessor(tmp_path))
    assert ledger.totals()["charged_optimizer_attempts"] == 0
    assert ledger.totals()["optimizer_fits_started"] == 0
    assert ledger.state["prior_campaign"]["charged"] == 602000
    spec = FitSpec("future", "S72", "DIRECT", 0, 7, 300000, "a" * 64, "b" * 64)
    with pytest.raises(RuntimeError, match="900000"):
        ledger.reserve(spec)
    assert not ledger.state["jobs"]


@pytest.mark.parametrize("terminal,remaining", [(False, 0), (True, 1)])
def test_active_or_reserved_predecessor_is_rejected(tmp_path, monkeypatch, terminal, remaining):
    ledger = local_ledger(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        ledger.inherit_closed_campaign(
            predecessor(tmp_path, terminal=terminal, remaining=remaining)
        )


def test_inherited_technical_updates_consume_shared_qa_ceiling(tmp_path, monkeypatch):
    ledger = local_ledger(tmp_path, monkeypatch)
    ledger.inherit_closed_campaign(predecessor(tmp_path, technical=2900))
    with pytest.raises(RuntimeError, match="3000"):
        ledger.reserve_technical_audit("new_qa", 101, "a" * 64)


def test_changed_predecessor_is_rejected_when_local_ledger_reopens(tmp_path, monkeypatch):
    ledger = local_ledger(tmp_path, monkeypatch)
    inherited = predecessor(tmp_path)
    ledger.inherit_closed_campaign(inherited)
    record = json.loads((tmp_path / "terminal.json").read_text())
    record["optimizer_totals"]["charged_optimizer_attempts"] -= 1
    atomic_json(tmp_path / "terminal.json", record)
    with pytest.raises(ValueError):
        local_ledger(tmp_path, monkeypatch)


def preparation_receipt(tmp_path, *, duplicate=False, wrong_total=False):
    hook = dict(
        identity="c" * 64,
        limit=8,
        charged=8,
        confirmed=8,
        uncertain=0,
        optimizer_instances_started=2,
        all_processes_closed=True,
    )
    hook["sha256"] = object_digest(hook)
    atomic_json(tmp_path / "hook.json", hook)
    atomic_json(tmp_path / "qa.json", {"fixture": "closed synthetic QA"})
    item = dict(receipt=binding(tmp_path / "hook.json"), charged=8, confirmed=8, uncertain=0)
    record = dict(
        kind="COMPARATOR_CPU_INTEGRATION_QA_AND_ACCOUNTING",
        scientific_candidates=0,
        audited_optimizer_receipts=[item, item] if duplicate else [item],
        unhooked_initial_suite=dict(
            charged_updates=4,
            source=binding(tmp_path / "qa.json"),
            incident="fixture absent hook",
            accounting_method="reconstructed, not a hook receipt",
        ),
        total_new_technical_optimizer_updates=11 if wrong_total else 12,
        qa_sources=[binding(tmp_path / "qa.json")],
    )
    record["artifact_sha256"] = object_digest(record)
    atomic_json(tmp_path / "preparation.json", record)
    return binding(tmp_path / "preparation.json")


def test_preparation_debit_is_counted_once_and_keeps_unhooked_uncertainty(tmp_path, monkeypatch):
    ledger = local_ledger(tmp_path, monkeypatch)
    ledger.bind_capacity_envelope(envelope(tmp_path))
    ledger.inherit_closed_campaign(predecessor(tmp_path))
    source = preparation_receipt(tmp_path)
    ledger.carry_preparation_accounting(source)
    ledger.carry_preparation_accounting(source)
    restored = local_ledger(tmp_path, monkeypatch)
    assert restored.state == ledger.state
    assert ledger.totals()["charged_optimizer_attempts"] == 12
    assert ledger.totals()["confirmed_optimizer_updates"] == 8
    assert ledger.totals()["uncertain_optimizer_attempts"] == 4
    assert ledger.totals()["optimizer_endpoints_complete"] == 0
    with pytest.raises(RuntimeError, match="10000"):
        ledger.reserve_technical_audit("too_much", 9989, "d" * 64)
    (tmp_path / "hook.json").write_text("{}")
    with pytest.raises(ValueError):
        local_ledger(tmp_path, monkeypatch)


@pytest.mark.parametrize("defect", ["duplicate", "wrong_total", "late"])
def test_preparation_debit_rejects_duplicates_mismatch_and_late_import(
    tmp_path, monkeypatch, defect
):
    ledger = local_ledger(tmp_path, monkeypatch)
    ledger.bind_capacity_envelope(envelope(tmp_path))
    ledger.inherit_closed_campaign(predecessor(tmp_path))
    source = preparation_receipt(
        tmp_path, duplicate=defect == "duplicate", wrong_total=defect == "wrong_total"
    )
    if defect == "late":
        ledger.reserve(FitSpec("local", "S72", "DIRECT", 0, 7, 8000, "a" * 64, "b" * 64))
    before = json.loads(json.dumps(ledger.state))
    with pytest.raises(ValueError):
        ledger.carry_preparation_accounting(source)
    assert ledger.state == before
