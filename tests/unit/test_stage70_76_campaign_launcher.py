"""Finite graph orchestration and honest interruption without scientific fits."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, object_digest


@pytest.fixture
def launcher():
    path = Path(__file__).resolve().parents[2] / "scripts/run_scientific_recovery_v10_stage70_76.py"
    spec = importlib.util.spec_from_file_location("stage70_campaign_launcher_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("interrupt", [False, True])
def test_launcher_executes_all_independent_and_conditional_backends(
    launcher, monkeypatch, tmp_path, interrupt
):
    calls = []
    lock = {
        "campaign_root": str(tmp_path),
        "startup_reservation": {"disk_bytes": 1, "ram_bytes": 1},
    }
    monkeypatch.setattr(launcher, "require_scientific_lock", lambda *args: lock)

    class Owner:
        def __init__(self, root, **kwargs):
            self.root = root

        def __enter__(self):
            calls.append("owner_enter")
            return self

        def __exit__(self, *args):
            calls.append("owner_closed")

    def ledger(path, root):
        path.write_text("synthetic ledger")
        return SimpleNamespace(path=path, totals=lambda: {"confirmed_optimizer_updates": 0})

    class Runtime:
        def __init__(self, *args):
            pass

        def fit_initial_stages(self):
            calls.extend(["S72", "S73"])
            return {"S72": {}, "S73": {}}

        def infer_and_export(self, barriers):
            assert set(barriers) == {"S72", "S73"}
            calls.append("outer_after_both_stages")
            return {"S72-negative": {}, "S73-independent": {}}

        def analyze_initial_stages(self, exports, barriers):
            calls.append("analysis")
            return {"S74": {}, "S75": {}}

    def replication(*args):
        calls.append("S74")
        if interrupt:
            raise RuntimeError("injected resource interruption")
        return {"status": "NOT_TRIGGERED"}

    monkeypatch.setattr(launcher, "ResourceOwner", Owner)
    monkeypatch.setattr(launcher, "UpdateLedger", ledger)
    monkeypatch.setattr(launcher, "InitialArchitectureRuntime", Runtime)
    monkeypatch.setattr(launcher, "run_replication", replication)
    monkeypatch.setattr(
        launcher, "run_new_nested_producers", lambda *args: calls.append("S75_inner") or {}
    )
    monkeypatch.setattr(
        launcher, "run_nested_routers", lambda *args: calls.append("S75_ridge") or {}
    )
    monkeypatch.setattr(
        launcher,
        "publish_blocked_confirmation",
        lambda *args: calls.append("S76") or {"status": "PREREQUISITE_BLOCKED"},
    )
    if interrupt:
        with pytest.raises(RuntimeError, match="injected"):
            launcher.execute(tmp_path, tmp_path / "lock", "a" * 64)
    else:
        launcher.execute(tmp_path, tmp_path / "lock", "a" * 64)
    result = json.loads((tmp_path / "CAMPAIGN_EXECUTION.json").read_text())
    assert result["optimizer_totals"]["confirmed_optimizer_updates"] == 0
    assert result["status"] == ("INTERRUPTED" if interrupt else "AUTHORIZED_GRAPH_TERMINAL")
    assert calls[:6] == ["owner_enter", "S72", "S73", "outer_after_both_stages", "analysis", "S74"]
    assert calls[-1] == "owner_closed"
    if not interrupt:
        assert calls[-4:] == ["S75_inner", "S75_ridge", "S76", "owner_closed"]


def test_freeze_rejects_incomplete_closure_before_git_or_training(launcher, tmp_path):
    closure = {"status": "INCOMPLETE", "closed_blockers": ["F1"]}
    closure["artifact_sha256"] = object_digest(closure)
    atomic_json(tmp_path / "preflight/FREEZE_CLOSURE.json", closure)
    with pytest.raises(ValueError, match="finite mandatory"):
        launcher.freeze(tmp_path, tmp_path / "protocol", tmp_path / "paths")


def test_cli_modes_are_mutually_exclusive(launcher):
    with pytest.raises(SystemExit):
        launcher.parser().parse_args(
            [
                "--campaign",
                "x",
                "--local-paths",
                "y",
                "--output-root",
                "z",
                "--execute-authorized",
                "--package-only",
            ]
        )


def test_interruption_bundle_is_verified_without_claiming_completed_science(tmp_path, monkeypatch):
    import zipfile

    from e_jepa_ttc.artifacts.risk_geometry_v10 import binding

    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/package_scientific_recovery_v10_stage70_76.py"
    )
    spec = importlib.util.spec_from_file_location("stage70_packager_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = tmp_path / "project"
    root = project / "artifacts/stage70_76_architecture"
    root.mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", project)
    monkeypatch.setattr(module, "require_scientific_lock", lambda *args: {})
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda args, **kwargs: "c" * 40 if "rev-parse" in args else b"",
    )
    cache = root / "synthetic_cache_manifest.json"
    atomic_json(cache, {"synthetic": True, "raw_arrays": "not packaged"})
    lock = {
        "code_commit": "b" * 40,
        "artifact_sha256": "a" * 64,
        "canonical_candidates": {"C_DIRECT": "S73-DIRECT-D", "C_PPS": "S73-PPS-AD"},
        "count_caches": {"original": binding(cache)},
        "preflight_receipts": {"handoff_integrity": binding(cache)},
    }
    for field in (
        "endpoint_root",
        "outer_inference_root",
        "inner_inference_root",
        "treatment_export_root",
        "replication_export_root",
        "initial_evidence_root",
        "replication_evidence_root",
        "stage_barrier_root",
        "stage_decision_root",
        "nested_source_root",
        "nested_export_root",
        "router_root",
    ):
        lock[field] = str(root / field)
    atomic_json(root / "TRAINING_LOCK.json", lock)
    ledger = {"jobs": {}, "pending": None}
    ledger["sha256"] = object_digest(ledger)
    atomic_json(root / "UPDATE_LEDGER.json", ledger)
    execution = {
        "status": "INTERRUPTED",
        "training_lock_sha256": "a" * 64,
        "optimizer_totals": {"confirmed_optimizer_updates": 0},
        "stages": {},
        "error": {"message": "synthetic resource interruption"},
    }
    atomic_json(root / "CAMPAIGN_EXECUTION.json", execution)
    result = module.package(root)
    assert result["crc_verified"] and result["extracted_hashes_verified"] > 0
    decision = json.loads((root / "NEXT_DECISION_V4.json").read_text())
    assert decision["scientific_optimizer_endpoints_complete"] == 0
    assert decision["next_action"] == "RESUME_EXACT_PENDING_GRAPH"
    with zipfile.ZipFile(result["zip"]["path"]) as archive:
        assert "BUNDLE_VERIFICATION.json" in archive.namelist()
        assert "CODEX_STAGE70_76_FINAL_REPORT.md" in archive.namelist()
        assert not any(name.endswith(".zip") for name in archive.namelist())
    execution["status"] = "AUTHORIZED_GRAPH_TERMINAL"
    atomic_json(root / "CAMPAIGN_EXECUTION.json", execution)
    with pytest.raises(ValueError, match="omitted"):
        module.package(root)
