"""A completed timing attempt is adopted, never rerun to select a faster measurement."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.data.event_context128 import RateNormalization
from e_jepa_ttc.evaluation import job_deployment_cost_v11 as module
from e_jepa_ttc.training.stage72_jobs import FitJob
from e_jepa_ttc.training.stage73_factorial import PlannedFit


@pytest.mark.parametrize("capacity", [False, True])
def test_completed_timing_survives_index_publication_failure_without_rebenchmark(
    tmp_path, monkeypatch, capacity
):
    norm = RateNormalization((0.0, 0.0), (1.0, 1.0), "a" * 64, 10)
    job = FitJob(
        PlannedFit("S72", "S72-DIRECT", 0, 7, capacity="4M" if capacity else "2M"),
        "b" * 64,
        "a" * 64,
        object_digest(asdict(norm)),
        "c" * 64,
        "d" * 64,
        ("train",),
        ("outer",),
        "e" * 64,
    )
    loaded = SimpleNamespace(
        view=SimpleNamespace(identity="a" * 64),
        normalization=norm,
        source_sha256="e" * 64,
        source={"train_sequences": ["extra_train"]},
    )
    recipe = tmp_path / "recipe.json"
    atomic_json(
        recipe,
        {
            "rows": 16,
            "warmup_rows": 4,
            "complete_measured_passes": 2,
            "batch": 1,
            "roi_pixels": [128, 128],
            "optimizer_updates": 0,
            "labels_used": False,
        },
    )
    weights = tmp_path / "weights.pt"
    torch.save(torch.nn.Linear(2, 2).state_dict(), weights)
    lock = {
        "endpoint_root": str(tmp_path / "endpoints"),
        "code_commit": "f" * 40,
        "deployment_cost_recipe": binding(recipe),
        "raw_deployment": {
            "raw_train_root": "synthetic",
            "train_metadata": "synthetic",
            "bindings": {},
            "clock_charter": {},
        },
    }
    if capacity:
        lock["capacity_cohort"] = {"synthetic": True}
    monkeypatch.setattr(
        module, "verify_fit_endpoint", lambda *args: {"inference_weights": binding(weights)}
    )

    class Model(torch.nn.Linear):
        def __init__(self, **kwargs):
            super().__init__(2, 2)

        def cuda(self):
            return self

    monkeypatch.setattr(module, "ProjectivePhaseNetwork", Model)

    def access(*args, **kwargs):
        assert kwargs.get("allowed_sequences") == (frozenset({"extra_train"}) if capacity else None)
        kwargs["journal_path"].write_text("synthetic physical read ledger")

    monkeypatch.setattr(module, "ReadOnlyTrainAccess", access)
    monkeypatch.setattr(module, "ExpansionTrainAccess", access)
    monkeypatch.setattr(
        module, "RawDeploymentInputs", lambda *args: SimpleNamespace(close=lambda: None)
    )
    calls = []

    def benchmark(*args, **kwargs):
        calls.append(1)
        return {"synthetic_timing_fixture": True}

    monkeypatch.setattr(module, "benchmark_raw_deployment", benchmark)
    final = tmp_path / "endpoints" / job.plan.job_id / "deployment_cost/DEPLOYMENT_COST.json"
    publish = module.atomic_json

    def interrupt(path, record):
        if path == final:
            raise RuntimeError("injected final publication interruption")
        publish(path, record)

    monkeypatch.setattr(module, "atomic_json", interrupt)
    rng = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="publication interruption"):
        module.observe_job_deployment(job, loaded, lock, SimpleNamespace(check=lambda: None))
    assert torch.equal(rng, torch.get_rng_state())
    monkeypatch.setattr(module, "atomic_json", publish)
    result = module.observe_job_deployment(job, loaded, lock, SimpleNamespace(check=lambda: None))
    assert calls == [1] and final.exists()
    assert result["measurement"] == {"synthetic_timing_fixture": True}
    assert result == module.observe_job_deployment(
        job, loaded, lock, SimpleNamespace(check=lambda: None)
    )
    assert calls == [1]
