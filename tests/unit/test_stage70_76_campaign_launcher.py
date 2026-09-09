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


@pytest.mark.parametrize("corruption", [None, "missing", "cross_fold_parent"])
def test_capacity_comparator_graph_freezes_every_head_before_export(
    launcher, tmp_path, monkeypatch, corruption
):
    from dataclasses import asdict

    from e_jepa_ttc.artifacts.risk_geometry_v10 import binding
    from e_jepa_ttc.training.stage70_update_ledger import FitSpec

    jobs = {}
    for fold in range(3):
        for inner in (None, 0, 1, 2):
            protocol = tmp_path / f"protocol_{fold}_{inner}.json"
            atomic_json(protocol, {"inner_fold": inner})
            for arm in ("A5", "C2F", "PAIR"):
                name = f"{arm}_{fold}_{inner}"
                spec = FitSpec(
                    name,
                    "COMPARATOR",
                    arm,
                    fold,
                    7,
                    6840 if arm == "PAIR" else 18,
                    "a" * 64,
                    "b" * 64,
                )
                jobs[name] = {"spec": asdict(spec), "protocol": binding(protocol)}
                if arm == "PAIR":
                    jobs[name]["a5_job_id"] = f"A5_{fold}_{inner}"
    if corruption == "missing":
        jobs.pop("PAIR_2_2")
    elif corruption:
        jobs["PAIR_2_2"]["a5_job_id"] = "A5_1_2"
    lock = {
        "comparator_jobs": jobs,
        "campaign_root": str(tmp_path),
        "capacity_comparison_output": str(tmp_path / "comparison"),
    }
    monkeypatch.setattr(launcher, "require_scientific_lock", lambda *args: lock)
    calls = []
    state = {"jobs": {}}

    def reserve(spec):
        state["jobs"].setdefault(spec.job_id, {"status": "RESERVED"})

    ledger = SimpleNamespace(state=state, reserve=reserve, totals=lambda: {})

    def runner(job_id, *, fit_only, **kwargs):
        if fit_only:
            state["jobs"][job_id]["status"] = "COMPLETE"
        else:
            assert len(state["jobs"]) == 36
            assert all(j["status"] == "COMPLETE" for j in state["jobs"].values())
        calls.append((fit_only, job_id))

    def publish(*args):
        assert len(calls) == 72
        atomic_json(tmp_path / "comparison/COMPARISON_COMPLETE.json", {"fixture": True})
        return {"fixture": True}

    monkeypatch.setattr(
        "scripts.train_scientific_recovery_v8_router_expert.execute_capacity_comparator", runner
    )
    monkeypatch.setattr(
        "scripts.build_scientific_recovery_v9_stage61_reference.execute_capacity_pair", runner
    )
    monkeypatch.setattr(
        "scripts.run_scientific_recovery_v9_stage65.publish_capacity_risk17", publish
    )
    owner = SimpleNamespace()
    if corruption:
        with pytest.raises(ValueError):
            launcher.execute_capacity_comparator_graph(
                tmp_path / "lock", "a" * 64, owner, ledger, lambda _: None
            )
        assert not calls and not state["jobs"]
    else:
        launcher.execute_capacity_comparator_graph(
            tmp_path / "lock", "a" * 64, owner, ledger, lambda _: None
        )
        assert all(fit for fit, _ in calls[:36])
        assert all(not fit for fit, _ in calls[36:])


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


@pytest.mark.parametrize("capacity", [False, True])
def test_interruption_bundle_is_verified_without_claiming_completed_science(
    tmp_path, monkeypatch, capacity
):
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
    root = project / (
        "artifacts/stage70_76_capacity" if capacity else "artifacts/stage70_76_architecture"
    )
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
    if capacity:
        lock.update(
            capacity_cohort=binding(cache),
            capacity_comparison_output=str(root / "comparison"),
            comparator_jobs={"PAIR": {"output_root": str(root / "comparators/PAIR")}},
        )
        atomic_json(root / "comparators/PAIR/prediction/PAIR_EXPORT.json", {"synthetic": True})
        atomic_json(
            root / "comparators/PAIR/training_inputs/PRIVATE_LOCAL_CACHE.json", {"synthetic": True}
        )
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
        if capacity:
            assert "run/comparators/PAIR/prediction/PAIR_EXPORT.json" in archive.namelist()
            assert not any("PRIVATE_LOCAL_CACHE" in name for name in archive.namelist())
    execution["status"] = "AUTHORIZED_GRAPH_TERMINAL"
    atomic_json(root / "CAMPAIGN_EXECUTION.json", execution)
    with pytest.raises(ValueError, match="omitted"):
        module.package(root)


def test_capacity_launcher_cannot_write_original_root(launcher, monkeypatch, tmp_path):
    original = tmp_path / "artifacts/stage70_76_architecture"
    original.mkdir(parents=True)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(
        launcher.sys,
        "argv",
        [
            "runner",
            "--capacity",
            "--campaign",
            "unused",
            "--local-paths",
            "unused",
            "--output-root",
            str(original),
            "--execute-authorized",
        ],
    )
    with pytest.raises(ValueError, match="another campaign"):
        launcher.main()


def test_powershell_logs_and_resumes_existing_lock_without_hiding_exit_code(tmp_path):
    import shutil
    import subprocess
    import sys

    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell unavailable on this host")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(
        Path(__file__).resolve().parents[2] / "scripts/run_stage70_76.ps1",
        scripts / "run_stage70_76.ps1",
    )
    (scripts / "run_scientific_recovery_v10_stage70_76.py").write_text(
        "import sys,json\nfrom pathlib import Path\n"
        "a=sys.argv\nroot=Path(a[a.index('--output-root')+1])\nroot.mkdir(parents=True,exist_ok=True)\n"
        "(root/'TRAINING_LOCK.json').write_text('{}')\n"
        "print('resume='+str('--resume' in a),flush=True)\n"
        "print('cwd_valid='+str(Path.cwd()==Path(__file__).resolve().parents[1]),flush=True)\n"
        "print('synthetic warning',file=sys.stderr,flush=True)\n"
        "sys.exit(json.loads(Path(a[a.index('--campaign')+1]).read_text())['exit'])\n",
        encoding="utf-8",
    )
    campaign = tmp_path / "campaign.json"
    local = tmp_path / "local.json"
    local.write_text("{}")
    output = tmp_path / "artifacts/stage70_76_capacity"
    command = [
        shell,
        "-NoProfile",
        "-File",
        str(scripts / "run_stage70_76.ps1"),
        "-Python",
        sys.executable,
        "-Campaign",
        str(campaign),
        "-LocalPaths",
        str(local),
        "-OutputRoot",
        str(output),
        "-Capacity",
    ]
    for code, resume in ((0, False), (0, True), (17, True)):
        campaign.write_text(json.dumps({"exit": code}))
        run = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert run.returncode == code, run.stdout + run.stderr
        log_root = output.with_name(output.name + "_launcher_logs")
        state = json.loads(sorted(log_root.glob("*.json"))[-1].read_text(encoding="utf-8-sig"))
        assert state["resume"] is resume and state["exit_code"] == code
        assert state["status"] == ("COMMAND_FINISHED" if code == 0 else "INTERRUPTED")
        text = Path(state["log"]).read_text(encoding="utf-8-sig")
        assert "synthetic warning" in text and f"resume={resume}" in text
        assert "cwd_valid=True" in text


@pytest.mark.parametrize("preparation_exit,bad_hash", [(0, False), (17, False), (0, True)])
def test_powershell_count_preparation_precedes_training_and_failure_stops_chain(
    tmp_path, preparation_exit, bad_hash
):
    import hashlib
    import shutil
    import subprocess
    import sys

    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell unavailable on this host")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(
        Path(__file__).resolve().parents[2] / "scripts/run_stage70_76.ps1",
        scripts / "run_stage70_76.ps1",
    )
    (scripts / "prepare_stage70_capacity.py").write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "root=Path(__file__).resolve().parents[1]\n"
        "assert Path.cwd()==root\n"
        "a=sys.argv\n"
        "assert Path(a[a.index('--output-root')+1])=="
        "root/'artifacts/stage70_76_capacity/representation_30000'\n"
        "(root/'PREPARED.json').write_text(json.dumps(a))\n"
        "print('synthetic preparation',flush=True)\n"
        f"sys.exit({preparation_exit})\n",
        encoding="utf-8",
    )
    (scripts / "run_scientific_recovery_v10_stage70_76.py").write_text(
        "from pathlib import Path\n"
        "root=Path(__file__).resolve().parents[1]\n"
        "assert (root/'PREPARED.json').is_file()\n"
        "(root/'TRAINING_CALLED').write_text('synthetic only')\n"
        "print('synthetic training',flush=True)\n",
        encoding="utf-8",
    )
    campaign = tmp_path / "campaign.json"
    local = tmp_path / "local.json"
    usability = tmp_path / "usability.json"
    for path in (campaign, local, usability):
        path.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(usability.read_bytes()).hexdigest()
    output = tmp_path / "artifacts/stage70_76_capacity"
    command = [
        shell,
        "-NoProfile",
        "-File",
        str(scripts / "run_stage70_76.ps1"),
        "-Python",
        sys.executable,
        "-Campaign",
        str(campaign),
        "-LocalPaths",
        str(local),
        "-OutputRoot",
        str(output),
        "-Capacity",
        "-CountUsability",
        str(usability),
        "-CountUsabilitySha256",
        "0" * 64 if bad_hash else digest,
    ]
    result = subprocess.run(command, capture_output=True, encoding="utf-8", timeout=60)
    if bad_hash:
        assert result.returncode != 0
        assert "Count usability physical SHA256 changed" in result.stderr
        assert not (tmp_path / "PREPARED.json").exists()
        assert not (tmp_path / "TRAINING_CALLED").exists()
        assert not output.with_name(output.name + "_launcher_logs").exists()
        return
    assert result.returncode == preparation_exit, result.stdout + result.stderr
    assert (tmp_path / "TRAINING_CALLED").exists() == (preparation_exit == 0)
    prepared = json.loads((tmp_path / "PREPARED.json").read_text())
    assert prepared[prepared.index("--count-usability") + 1] == str(usability)
    assert prepared[prepared.index("--count-usability-sha256") + 1] == digest
    log_root = output.with_name(output.name + "_launcher_logs")
    state = json.loads(next(log_root.glob("*.json")).read_text(encoding="utf-8-sig"))
    assert state["exit_code"] == preparation_exit
    step = state["preparation_steps"][0]
    assert step["exit_code"] == preparation_exit
    assert step["status"] == ("COMMAND_FINISHED" if preparation_exit == 0 else "INTERRUPTED")
    text = Path(state["log"]).read_text(encoding="utf-8-sig")
    assert "synthetic preparation" in text
    if preparation_exit == 0:
        assert text.index("synthetic preparation") < text.index("synthetic training")
