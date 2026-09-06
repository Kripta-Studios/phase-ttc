"""The S75 executor creates all new inner producers before any routing prediction read."""

from types import SimpleNamespace

from e_jepa_ttc.data.event_context128 import RateNormalization
from e_jepa_ttc.training import stage75_runtime_v11 as module
from e_jepa_ttc.training.stage73_factorial import PlannedFit, treatment


def test_all_prescribed_inner_producers_finish_before_prediction(tmp_path, monkeypatch):
    plans = tuple(
        PlannedFit("S75", name, outer, 7, inner)
        for name in ("S73-DIRECT-D", "S73-PPS-AD")
        for outer in range(3)
        for inner in range(3)
    )
    events = []
    runtime = SimpleNamespace(
        root=tmp_path,
        lock_sha256="a" * 64,
        lock_path=tmp_path / "lock.json",
        lock={
            "nested_source_root": str(tmp_path / "sources"),
            "inner_inference_root": str(tmp_path / "inference"),
            "nested_export_root": str(tmp_path / "exports"),
            "source_role_sha256": "e" * 64,
        },
        paths={
            "endpoint_root": tmp_path / "endpoints",
            "stage_barrier_root": tmp_path / "barriers",
        },
        owner=SimpleNamespace(check=lambda: None),
        ledger=SimpleNamespace(totals=lambda: {}, path=tmp_path / "ledger.json"),
        original=object(),
        expanded=object(),
        roles=object(),
        _reserve_resources=lambda key: events.append(("reserve", key)),
    )
    monkeypatch.setattr(
        module, "verify_stage_decision", lambda *args: ({"artifact_sha256": "f" * 64}, plans)
    )

    def materialize(plan, **kwargs):
        events.append(("source", plan.job_id))
        return SimpleNamespace(
            view=SimpleNamespace(identity="b" * 64),
            normalization=RateNormalization((0.0, 0.0), (1.0, 1.0), "b" * 64, 100),
            supervision=SimpleNamespace(identity_sha256="d" * 64),
            source={"allowed_train_groups": ["train"], "excluded_groups": ["inner", "outer"]},
            source_sha256="9" * 64,
        )

    monkeypatch.setattr(module, "materialize_nested_sources", materialize)

    def fit(job, **kwargs):
        assert events[-2:] == [("reserve", job.plan.job_id), ("source", job.plan.job_id)]
        assert job.plan.stage == "S75" and job.plan.seed == 7 and job.plan.inner_fold in range(3)
        events.append(("fit", job.plan.job_id))
        if treatment(job.plan.treatment_name).pretext_updates:
            root = runtime.paths["endpoint_root"] / job.plan.job_id / "pretext"
            root.mkdir(parents=True)
            for update in (0, 3000):
                (root / f"checkpoint_{update:06d}.pt").write_bytes(
                    b"synthetic orchestration fixture"
                )

    monkeypatch.setattr(module, "execute_architecture_job", fit)
    monkeypatch.setattr(module, "PretextTextures", lambda *args: None)
    monkeypatch.setattr(
        module,
        "diagnose_pretext_checkpoint",
        lambda *args, **kwargs: events.append(("diagnostic", None)),
    )

    def freeze(jobs, **kwargs):
        assert tuple(job.plan for job in jobs) == plans
        assert sum(event[0] == "fit" for event in events) == 18
        path = kwargs["output"]
        path.parent.mkdir(parents=True)
        path.write_text("synthetic barrier fixture")
        events.append(("barrier", None))

    monkeypatch.setattr(module, "freeze_nested_stage", freeze)
    monkeypatch.setattr(
        module,
        "prospective_nested_recipe",
        lambda plan, lock: ({}, {"inner_prediction_tokens": [f"inner{plan.inner_fold}"]}),
    )

    def infer(job, **kwargs):
        assert ("barrier", None) in events
        assert kwargs["tokens"] == (f"inner{job.plan.inner_fold}",)
        events.append(("infer", job.plan.job_id))
        root = kwargs["output"]
        root.mkdir(parents=True)
        (root / "INFERENCE_COMPLETE.json").write_text("synthetic inference fixture")

    monkeypatch.setattr(module, "infer_initial_endpoint", infer)

    def export(name, outer, **kwargs):
        assert sum(event[0] == "infer" for event in events) == 18
        root = tmp_path / "exports" / name / f"outer{outer}"
        root.mkdir(parents=True)
        (root / "NESTED_EXPORT.json").write_text("synthetic export fixture")

    monkeypatch.setattr(module, "export_nested_treatment", export)
    result = module.run_new_nested_producers(runtime, {"path": "synthetic decision"})
    assert len(result["inference"]) == 18
    assert len(result["exports"]) == 6
    assert not result["router_fit_complete"]
    assert sum(event[0] == "diagnostic" for event in events) == 18
    assert [event[1] for event in events if event[0] == "fit"] == [p.job_id for p in plans]


def test_gate_not_triggered_cannot_materialize_or_train(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "verify_stage_decision", lambda *args: ({}, ()))
    runtime = SimpleNamespace(
        root=tmp_path,
        lock={},
        lock_sha256="a" * 64,
        ledger=SimpleNamespace(totals=lambda: {}, path=tmp_path / "ledger.json"),
    )
    result = module.run_new_nested_producers(runtime, {"path": "synthetic decision"})
    assert result["status"] == "NOT_TRIGGERED"
