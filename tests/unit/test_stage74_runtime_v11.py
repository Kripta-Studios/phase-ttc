"""Replicate all canonical seeds before their first outer observation, without optimizer work."""

from pathlib import Path
from types import SimpleNamespace

from e_jepa_ttc.training import stage74_runtime_v11 as module
from e_jepa_ttc.training.stage72_jobs import FitJob
from e_jepa_ttc.training.stage73_factorial import PlannedFit, treatment


def test_every_new_encoder_head_and_pretext_finishes_before_any_replication_outer_read(
    tmp_path, monkeypatch
):
    plans = tuple(
        PlannedFit("S74", name, fold, seed)
        for name in ("S73-DIRECT-D", "S73-PPS-AD")
        for seed in (13, 23)
        for fold in (0, 1, 2)
    )
    events = []
    runtime = SimpleNamespace(
        root=tmp_path,
        lock_sha256="a" * 64,
        lock_path=tmp_path / "lock.json",
        lock={
            "replication_export_root": str(tmp_path / "exports"),
            "outer_tokens_by_fold": {str(i): [f"outer{i}"] for i in range(3)},
        },
        paths={
            "endpoint_root": tmp_path / "endpoints",
            "stage_barrier_root": tmp_path / "barriers",
            "outer_inference_root": tmp_path / "inference",
        },
        owner=SimpleNamespace(check=lambda: None),
        ledger=SimpleNamespace(totals=lambda: {"scientific": 0}, path=tmp_path / "ledger.json"),
        original=object(),
        _reserve_resources=lambda key: events.append(("reserve", key)),
    )
    decision = {"artifact_sha256": "f" * 64}
    monkeypatch.setattr(module, "verify_stage_decision", lambda *args: (decision, plans))

    def resolve(plan):
        assert plan.seed == 7 and plan.stage == "S73"
        return FitJob(
            plan, "a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64, ("train",), ("outer",), "9" * 64
        ), SimpleNamespace(view=None, normalization=None, supervision=None)

    runtime.resolve = resolve

    def fit(job, **kwargs):
        assert events[-1] == ("reserve", job.plan.job_id)
        assert job.plan.seed in (13, 23) and job.stage_authorization_sha256 == "f" * 64
        events.append(("fit", job.plan.job_id))
        if treatment(job.plan.treatment_name).pretext_updates:
            root = runtime.paths["endpoint_root"] / job.plan.job_id / "pretext"
            root.mkdir(parents=True)
            for update in (0, 3000):
                (root / f"checkpoint_{update:06d}.pt").write_bytes(
                    b"synthetic orchestration fixture"
                )

    monkeypatch.setattr(module, "execute_architecture_job", fit)
    monkeypatch.setattr(module, "diagnose_supervised_series", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "PretextTextures", lambda *args: None)
    monkeypatch.setattr(
        module,
        "diagnose_pretext_checkpoint",
        lambda *args, **kwargs: events.append(("diagnostic", kwargs["seed"])),
    )

    def freeze(jobs, **kwargs):
        assert len(jobs) == 12
        assert sum(event[0] == "fit" for event in events) == 12
        path = kwargs["output"]
        path.parent.mkdir(parents=True)
        path.write_text("synthetic barrier fixture")
        events.append(("barrier", None))

    monkeypatch.setattr(module, "freeze_replication_stage", freeze)

    def infer(job, **kwargs):
        assert ("barrier", None) in events
        assert sum(event[0] == "fit" for event in events) == 12
        events.append(("infer", job.plan.job_id))

    monkeypatch.setattr(module, "infer_initial_endpoint", infer)

    def export(name, **kwargs):
        assert sum(event[0] == "infer" for event in events) == 12
        root = kwargs["output"]
        root.mkdir(parents=True)
        (root / "TREATMENT_EXPORT.json").write_text("synthetic export fixture")
        events.append(("export", (name, kwargs["seed"])))

    monkeypatch.setattr(module, "export_initial_treatment", export)

    def analyze(exports, *args, **kwargs):
        assert sum(event[0] == "export" for event in events) == 4
        assert len(exports) == 4
        events.append(("analysis", None))
        return {"synthetic_analysis": True}

    monkeypatch.setattr(module, "build_replication_evidence", analyze)
    result = module.run_replication(runtime, {"path": str(tmp_path / "decision.json")})
    assert len(result["exports"]) == 4
    assert result["analysis"] == {"synthetic_analysis": True}
    assert sum(event[0] == "diagnostic" for event in events) == 12
    assert [event[1] for event in events if event[0] == "fit"] == [plan.job_id for plan in plans]


def test_not_triggered_replication_does_not_reserve_or_resolve_any_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "verify_stage_decision", lambda *args: ({}, ()))
    runtime = SimpleNamespace(
        root=tmp_path,
        lock={},
        lock_sha256="a" * 64,
        ledger=SimpleNamespace(totals=lambda: {}, path=Path("ledger")),
    )
    result = module.run_replication(runtime, {"path": "synthetic"})
    assert result["status"] == "NOT_TRIGGERED" and result["exports"] == {}
