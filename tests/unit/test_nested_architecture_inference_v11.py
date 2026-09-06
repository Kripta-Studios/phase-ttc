"""New inner inference cannot consume outer rows or an exposed acquisition alias."""

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from e_jepa_ttc.evaluation import standalone_outer_v11 as module
from e_jepa_ttc.training.stage72_jobs import FitJob
from e_jepa_ttc.training.stage73_factorial import PlannedFit


@pytest.fixture
def fixture(monkeypatch):
    job = FitJob(
        PlannedFit("S75", "S73-PPS-AD", 0, 7, 0),
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
        ("train_group",),
        ("inner_group", "outer_group"),
        "f" * 64,
        "9" * 64,
    )
    reader = SimpleNamespace(
        identity="cache",
        metadata=pd.DataFrame(
            {
                "sample_token": ["inner0", "inner1", "outer0", "train0"],
                "sequence_id": ["inner", "inner", "outer", "train"],
                "outer_fold": [1, 1, 0, 2],
            }
        ),
    )
    recipe = {
        "inner_prediction_tokens": ["inner0", "inner1"],
        "inner_prediction_sequences": ["inner"],
        "ordered_tokens": ["train0"],
        "outer_excluded_sequences": ["outer"],
    }
    monkeypatch.setattr(module, "prospective_nested_recipe", lambda *args: ({}, recipe))
    lock = {
        "outer_count_cache_sha256": "cache",
        "sequence_to_group": {
            "inner": "inner_group",
            "outer": "outer_group",
            "train": "train_group",
        },
    }
    return job, reader, lock, recipe


def test_inner_prediction_uses_exact_prescribed_membership(fixture):
    job, reader, lock, _ = fixture
    module.validate_inner_view(job, reader, ("inner0", "inner1"), lock)
    for tokens in (("outer0",), ("inner0",), ("inner1", "inner0"), ("train0",)):
        with pytest.raises(ValueError):
            module.validate_inner_view(job, reader, tokens, lock)


@pytest.mark.parametrize("damage", ["alias", "missing_exclusion", "outer", "fit_token", "cache"])
def test_inner_prediction_rejects_transitive_exposure(fixture, damage):
    job, reader, lock, recipe = fixture
    if damage == "alias":
        lock["sequence_to_group"]["inner"] = "train_group"
    elif damage == "missing_exclusion":
        job = replace(job, excluded_groups=("outer_group",))
    elif damage == "outer":
        recipe["outer_excluded_sequences"].append("inner")
    elif damage == "fit_token":
        recipe["ordered_tokens"].append("inner0")
    else:
        reader.identity = "changed"
    with pytest.raises(ValueError):
        module.validate_inner_view(job, reader, ("inner0", "inner1"), lock)


def test_missing_new_inner_endpoint_barrier_prevents_observation(fixture, tmp_path):
    job, reader, _, _ = fixture
    reader.observation = lambda token: pytest.fail(
        "count read before all new inner endpoints froze"
    )
    with pytest.raises(FileNotFoundError):
        module.infer_initial_endpoint(
            job,
            barrier_path=tmp_path / "missing.json",
            lock_path=tmp_path / "lock.json",
            reader=reader,
            tokens=("inner0", "inner1"),
            normalization=None,
            output=tmp_path,
            owner=SimpleNamespace(check=lambda: None),
        )
