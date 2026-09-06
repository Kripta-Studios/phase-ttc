"""Finite execution order tests isolate orchestration; no optimizer or model is constructed."""

from pathlib import Path
from types import SimpleNamespace

from e_jepa_ttc.training import stage72_73_runtime_v11 as runtime
from e_jepa_ttc.training.stage72_jobs import FitJob
from e_jepa_ttc.training.stage73_factorial import Availability, initial_jobs, treatment


def test_s73_all_available_jobs_run_without_reading_or_gating_on_s72_scores(tmp_path, monkeypatch):
    availability = Availability(True, True, "a" * 64, "b" * 64)
    engine = runtime.InitialArchitectureRuntime.__new__(runtime.InitialArchitectureRuntime)
    engine.availability = availability
    engine.plans = initial_jobs(availability)
    engine.lock_path = tmp_path / "SYNTHETIC_LOCK.json"
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
