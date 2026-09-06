"""Finite execution order tests isolate orchestration; no optimizer or model is constructed."""

from pathlib import Path
from types import SimpleNamespace

from e_jepa_ttc.training import stage72_73_runtime_v11 as runtime
from e_jepa_ttc.training.stage72_jobs import FitJob
from e_jepa_ttc.training.stage73_factorial import Availability, initial_jobs, treatment


def test_expanded_exports_keep_every_capacity_separate_after_all_inferences(tmp_path, monkeypatch):
    engine = runtime.InitialArchitectureRuntime.__new__(runtime.InitialArchitectureRuntime)
    engine.plans = initial_jobs(
        Availability(True, False, "a" * 64, "b" * 64, ("2M", "4M", "8M"), "4M")
    )
    engine.lock = {"outer_tokens_by_fold": {str(fold): [f"t{fold}"] for fold in range(3)}}
    engine.lock_path = tmp_path / "lock.json"
    engine.lock_sha256 = "a" * 64
    engine.original = object()
    engine.owner = SimpleNamespace(check=lambda: None)
    engine.paths = {
        "outer_inference_root": tmp_path / "inference",
        "treatment_export_root": tmp_path / "exports",
    }
    monkeypatch.setattr(engine, "_reserve_resources", lambda *args: None)
    monkeypatch.setattr(engine, "_progress", lambda *args: None)
    monkeypatch.setattr(engine, "resolve", lambda plan: (plan, SimpleNamespace(normalization=None)))
    monkeypatch.setattr(runtime, "verify", lambda source: Path(source))
    inferred, exported = [], []
    monkeypatch.setattr(
        runtime, "infer_initial_endpoint", lambda plan, **kwargs: inferred.append(plan.job_id)
    )

    def export(name, **kwargs):
        assert len(inferred) == 42
        exported.append((name, kwargs["capacity"]))
        path = kwargs["output"] / "TREATMENT_EXPORT.json"
        path.parent.mkdir(parents=True)
        path.write_text("synthetic export dispatch fixture")

    monkeypatch.setattr(runtime, "export_initial_treatment", export)
    result = engine.infer_and_export({"S72": "s72", "S73": "s73"})
    assert len(result) == len(set(exported)) == 14
    assert {size for name, size in exported if name == "S72-PPS-TRAJ"} == {"2M", "4M", "8M"}


def test_s73_all_available_jobs_run_without_reading_or_gating_on_s72_scores(tmp_path, monkeypatch):
    availability = Availability(True, True, "a" * 64, "b" * 64)
    engine = runtime.InitialArchitectureRuntime.__new__(runtime.InitialArchitectureRuntime)
    engine.availability = availability
    engine.plans = initial_jobs(availability)
    engine.lock_path = tmp_path / "SYNTHETIC_LOCK.json"
    engine.lock = {}
    engine.lock_sha256 = "a" * 64
    engine.paths = {
        "endpoint_root": tmp_path / "endpoints",
        "stage_barrier_root": tmp_path / "barriers",
    }
    engine.owner = SimpleNamespace(check=lambda: None)
    engine.ledger = object()
    events = []
    monkeypatch.setattr(engine, "_reserve_resources", lambda key: events.append(("reserve", key)))
    monkeypatch.setattr(engine, "_progress", lambda *args: None)

    def resolve(plan):
        job = FitJob(
            plan, "a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64, ("train",), ("outer",), "f" * 64
        )
        return job, SimpleNamespace(view=None, normalization=None, supervision=None)

    monkeypatch.setattr(engine, "resolve", resolve)

    def fit(job, **kwargs):
        assert events[-1] == ("reserve", job.plan.job_id)
        events.append(("fit", job.plan.job_id))
        if treatment(job.plan.treatment_name).pretext_updates:
            directory = engine.paths["endpoint_root"] / job.plan.job_id / "pretext"
            directory.mkdir(parents=True)
            for update in (0, 3000):
                (directory / f"checkpoint_{update:06d}.pt").write_bytes(
                    b"synthetic orchestration fixture"
                )

    monkeypatch.setattr(runtime, "execute_architecture_job", fit)
    monkeypatch.setattr(runtime, "observe_job_deployment", lambda *args: {})
    monkeypatch.setattr(
        runtime,
        "diagnose_supervised_series",
        lambda *args, **kwargs: events.append(("supervised_diagnostic", kwargs["endpoint"])),
    )
    monkeypatch.setattr(runtime, "PretextTextures", lambda *args: None)
    monkeypatch.setattr(
        runtime,
        "diagnose_pretext_checkpoint",
        lambda *args, **kwargs: events.append(("diagnostic", None)),
    )

    def freeze(stage, jobs, availability, **kwargs):
        count = sum(event[0] == "fit" for event in events)
        assert count == (12 if stage == "S72" else 27)
        assert len(jobs) == (12 if stage == "S72" else 15)
        events.append(("freeze", stage))
        path = Path(kwargs["output"])
        path.parent.mkdir(exist_ok=True)
        path.write_text("synthetic stage receipt")

    monkeypatch.setattr(runtime, "freeze_initial_stage", freeze)

    def forbidden(*args, **kwargs):
        raise AssertionError("initial fitting attempted to inspect an outer score")

    monkeypatch.setattr(runtime, "infer_initial_endpoint", forbidden)
    monkeypatch.setattr(runtime, "export_initial_treatment", forbidden)
    result = engine.fit_initial_stages()
    assert set(result) == {"S72", "S73"}
    assert [event[1] for event in events if event[0] == "fit"] == [p.job_id for p in engine.plans]
    assert sum(event[0] == "diagnostic" for event in events) == 12
