"""Three tiny synthetic ridge solves exercise immutable endpoints and publication recovery."""

import numpy as np
import pytest

from e_jepa_ttc.training import nested_full_cost_fit_v11 as module


def inputs():
    rng = np.random.default_rng(7)
    return rng.normal(size=(64, 45)), abs(rng.normal(size=(64, 5))), np.full(64, 1 / 64)


def fit(tmp_path):
    features, costs, mass = inputs()
    return module.fit_full_cost_endpoint(
        features,
        costs,
        mass,
        experts=("A5", "C2F", "PAIR", "DIRECT", "PPS"),
        identity={"technical_synthetic_fixture": True},
        output=tmp_path / "fit",
    )


def test_completed_ridge_resume_never_solves_again(tmp_path, monkeypatch):
    first = fit(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("completed endpoint attempted another ridge fit")

    monkeypatch.setattr(module.FullCostRouterV11, "fit", forbidden)
    assert fit(tmp_path) == first
    _, model = module.load_full_cost_endpoint(tmp_path / "fit/RIDGE_ENDPOINT.json")
    assert model.predict_regret(inputs()[0]).shape == (64, 5)
    assert first["completed_ridge_solves"] == 1 and first["optimizer_updates"] == 0


def test_closed_completed_attempt_recovers_after_index_publication_failure(tmp_path, monkeypatch):
    publish = module.atomic_json

    def fail_index(path, record):
        if path.name == "RIDGE_ENDPOINT.json":
            raise OSError("injected Windows publication interruption")
        publish(path, record)

    monkeypatch.setattr(module, "atomic_json", fail_index)
    with pytest.raises(OSError, match="injected"):
        fit(tmp_path)
    monkeypatch.setattr(module, "atomic_json", publish)

    def forbidden(*args, **kwargs):
        pytest.fail("closed complete solve was unnecessarily recomputed")

    monkeypatch.setattr(module.FullCostRouterV11, "fit", forbidden)
    record = fit(tmp_path)
    assert record["earlier_incomplete_attempts"] == 0
    assert len(list((tmp_path / "fit").glob("attempt_*"))) == 1


def test_changed_coefficients_block_completed_evidence_without_refitting(tmp_path, monkeypatch):
    first = fit(tmp_path)
    from pathlib import Path

    Path(first["weights"]["path"]).write_bytes(b"damaged fixture")

    def forbidden(*args, **kwargs):
        pytest.fail("corrupt completed ridge was silently refitted")

    monkeypatch.setattr(module.FullCostRouterV11, "fit", forbidden)
    with pytest.raises(ValueError, match="physical identity changed"):
        fit(tmp_path)


def test_real_solve_is_counted_once_without_charging_optimizer_updates(tmp_path, monkeypatch):
    import json
    import os

    import psutil

    from e_jepa_ttc.training.stage70_update_ledger import UpdateLedger

    (tmp_path / "ACTIVE_OWNER.json").write_text(
        json.dumps({"pid": os.getpid(), "created": psutil.Process().create_time()})
    )
    ledger = UpdateLedger(tmp_path / "LEDGER.json", tmp_path)
    features, costs, mass = inputs()
    kwargs = {
        "experts": ("A5", "C2F", "PAIR", "DIRECT", "PPS"),
        "identity": {"technical_synthetic_fixture": True},
        "output": tmp_path / "fit",
        "on_attempt": ledger.record_ridge_attempt,
    }
    module.fit_full_cost_endpoint(features, costs, mass, **kwargs)
    assert ledger.totals()["ridge_fits"] == 1
    assert ledger.totals()["confirmed_optimizer_updates"] == 0
    assert ledger.state["ridge_fits"][0]["completed"] is not None

    def forbidden(*args, **kwargs):
        pytest.fail("ledger recovery repeated an already completed solve")

    monkeypatch.setattr(module.FullCostRouterV11, "fit", forbidden)
    module.fit_full_cost_endpoint(features, costs, mass, **kwargs)
    assert ledger.totals()["ridge_fits"] == 1
