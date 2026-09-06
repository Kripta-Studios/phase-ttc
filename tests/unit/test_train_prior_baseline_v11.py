"""Frozen prior source isolation and phase-median reporting without target fitting."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.evaluation.architecture_diagnostics_v11 import DiagnosticExport
from e_jepa_ttc.evaluation.train_prior_baseline_v11 import append_prior_baseline, frozen_prior
from e_jepa_ttc.training.stage73_factorial import PlannedFit


def test_prior_reuses_exact_registered_population_for_new_seed(tmp_path):
    supervision = {
        "training_view_sha256": "a" * 64,
        "tokens": ["train_only"],
        "prior_bin_mass": (np.ones(108) / 108).tolist(),
    }
    path = tmp_path / "SUPERVISION.json"
    atomic_json(path, supervision)
    source = {
        "outer_fold": 1,
        "include_expansion": True,
        "source_role_sha256": "b" * 64,
        "training_view_sha256": "a" * 64,
        "ordered_tokens": ["train_only"],
        "supervision": binding(path),
    }
    source["artifact_sha256"] = sha = object_digest(source)
    source_path = tmp_path / "SOURCE_VIEW.json"
    atomic_json(source_path, source)
    original = PlannedFit("S73", "S73-DIRECT-D", 1, 7)
    lock = {
        "initial_source_manifest_by_job": {original.job_id: sha},
        "authorized_source_manifests": {sha: binding(source_path)},
        "source_role_sha256": "b" * 64,
    }
    logits, references = frozen_prior(PlannedFit("S74", "S73-DIRECT-D", 1, 23), lock)
    expected = (
        (torch.tensor(supervision["prior_bin_mass"], dtype=torch.float64) + 1e-3)
        .log()
        .float()
        .numpy()
    )
    np.testing.assert_array_equal(logits, expected)
    assert references["supervision"] == binding(path)
    changed = json.loads(path.read_text())
    changed["tokens"] = ["held_out"]
    atomic_json(path, changed)
    with pytest.raises(ValueError):
        frozen_prior(original, lock)


def test_prior_predictions_do_not_change_when_evaluation_targets_change():
    metadata = pd.DataFrame(
        {
            "outer_fold": np.repeat(np.arange(9) % 3, 4),
            "sequence_id": np.repeat([f"s{i}" for i in range(9)], 4),
            "target_ttc_s": np.tile([2.0, 4.0, 8.0, -2.0], 9),
        }
    )
    prior = {fold: np.linspace(-fold, fold + 1, 108).astype(np.float32) for fold in range(3)}
    first = DiagnosticExport(metadata.copy(), {}, {})
    append_prior_baseline(first, metadata, prior)
    changed = metadata.copy()
    changed["target_ttc_s"] = np.tile([1.0, 5.0, 9.0, -3.0], 9)
    second = DiagnosticExport(changed.copy(), {}, {})
    append_prior_baseline(second, changed, prior)
    np.testing.assert_array_equal(first.rows.train_prior_ttc_s, second.rows.train_prior_ttc_s)
    assert (
        first.summary["train_prior_baseline"]["score"]
        != second.summary["train_prior_baseline"]["score"]
    )
    assert first.arrays["train_prior_posterior108_by_fold"].shape == (3, 108)
