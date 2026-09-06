"""Fixed diagnostic sampling, finite health summaries and source rejection."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from e_jepa_ttc.artifacts.risk_geometry_v10 import binding
from e_jepa_ttc.evaluation.supervised_checkpoint_diagnostics_v11 import (
    diagnose_supervised_checkpoint,
    diagnostic_tokens,
    embedding_health,
)
from e_jepa_ttc.training.stage70_update_ledger import FitSpec


@pytest.mark.parametrize("endpoint", [8000, 11000])
def test_series_observes_all_fixed_updates(tmp_path, monkeypatch, endpoint):
    from e_jepa_ttc.evaluation import supervised_checkpoint_diagnostics_v11 as module

    expected = [0, 1000, 2000, 4000, 8000] + ([11000] if endpoint == 11000 else [])
    root = tmp_path / "supervised"
    root.mkdir()
    for update in expected:
        (root / f"checkpoint_{update:06d}.pt").write_bytes(b"source fixture")
    seen = []

    def observe(checkpoint, *args, **kwargs):
        seen.append(int(checkpoint["path"].split("checkpoint_")[-1].split(".")[0]))
        return {"recipe": kwargs["recipe_sha256"]}

    monkeypatch.setattr(module, "diagnose_supervised_checkpoint", observe)
    result = module.diagnose_supervised_series(
        tmp_path,
        None,
        None,
        None,
        recipe_sha256="a" * 64,
        endpoint=endpoint,
        check=lambda: None,
    )
    assert seen == expected + [endpoint] * 7
    assert result == [{"recipe": "a" * 64}] * (len(expected) + 7)


def test_sampling_is_independent_of_source_order():
    tokens = tuple(f"token_{i}" for i in range(100))
    selected = diagnostic_tokens(tokens)
    assert selected == diagnostic_tokens(tuple(reversed(tokens)))
    assert len(set(selected)) == 16
    assert set(selected) <= set(tokens)


@pytest.mark.parametrize("tokens", [tuple(map(str, range(15))), ("duplicate",) * 16])
def test_sampling_rejects_small_or_duplicate_population(tokens):
    with pytest.raises(ValueError, match="unique"):
        diagnostic_tokens(tokens)


def test_health_identifies_collapse_and_rank_one():
    collapsed = embedding_health(np.ones((16, 96)))
    assert collapsed == {
        "mean_std": 0.0,
        "fraction_std_below_0p001": 1.0,
        "effective_rank": 0.0,
    }
    rank_one = embedding_health(np.arange(16)[:, None] * np.ones((1, 4)))
    assert rank_one["effective_rank"] == pytest.approx(1.0)
    assert rank_one["fraction_std_below_0p001"] == 0.0
    assert all(value is None for value in embedding_health(np.full((2, 4), np.nan)).values())


@pytest.mark.parametrize("change", ["recipe", "view", "normalization", "tokens", "update"])
def test_source_mismatch_rejected_before_model_or_output(tmp_path, change):
    tokens = tuple(map(str, range(16)))
    spec = FitSpec("diagnostic", "S72", "DIRECT", 0, 7, 8000, "a" * 64, "b" * 64)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"spec": asdict(spec), "update": 250 if change == "update" else 1000}, checkpoint)
    view = SimpleNamespace(identity="c" * 64 if change == "view" else "b" * 64, tokens=tokens)
    norm = SimpleNamespace(training_view_sha256="c" * 64 if change == "normalization" else "b" * 64)
    supervision = SimpleNamespace(tokens=() if change == "tokens" else tokens)
    with pytest.raises(ValueError, match="source identity"):
        diagnose_supervised_checkpoint(
            binding(checkpoint),
            view,
            norm,
            supervision,
            recipe_sha256="c" * 64 if change == "recipe" else "a" * 64,
            output=tmp_path / "observations",
            check=lambda: None,
            device="cpu",
        )
    assert not (tmp_path / "observations").exists()
