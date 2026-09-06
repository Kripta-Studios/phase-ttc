"""Frozen TRAIN-bin prior baseline, with no outer-target fitting or selection."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest, verify
from e_jepa_ttc.evaluation.architecture_diagnostics_v11 import DiagnosticExport
from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass
from e_jepa_ttc.models.phase_distribution_v11 import phase_edges, phase_to_finite_ttc, quantile
from e_jepa_ttc.training.stage73_factorial import PlannedFit, treatment


def frozen_prior(plan: PlannedFit, lock: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover the already-fitted fold TRAIN prior from its prospectively bound source."""
    stage = "S72" if plan.treatment_name.startswith("S72") else "S73"
    original = PlannedFit(stage, plan.treatment_name, plan.outer_fold, 7)
    sha = lock["initial_source_manifest_by_job"][original.job_id]
    source_binding = lock["authorized_source_manifests"][sha]
    source = json.loads(verify(source_binding).read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in source.items() if key != "artifact_sha256"}
    if (
        object_digest(unsigned) != sha
        or source.get("artifact_sha256") != sha
        or source.get("outer_fold") != plan.outer_fold
        or source.get("include_expansion") != treatment(plan.treatment_name).expanded
        or source.get("source_role_sha256") != lock["source_role_sha256"]
    ):
        raise ValueError("prior source differs from its frozen fold or role")
    supervision = json.loads(verify(source["supervision"]).read_text(encoding="utf-8"))
    prior = np.asarray(supervision["prior_bin_mass"], dtype=np.float64)
    if (
        supervision.get("training_view_sha256") != source["training_view_sha256"]
        or supervision.get("tokens") != source["ordered_tokens"]
        or prior.shape != (108,)
        or not np.isfinite(prior).all()
        or (prior < 0).any()
        or abs(float(prior.sum()) - 1) > 1e-12
    ):
        raise ValueError("prior does not describe its complete frozen TRAIN population")
    # Match initialize_train_prior: log in float64, then copy to the FP32 head bias.
    logits = (torch.from_numpy(prior) + 1e-3).log().float().numpy()
    return logits, {"source_manifest": source_binding, "supervision": source["supervision"]}


def append_prior_baseline(
    diagnostic: DiagnosticExport, metadata: pd.DataFrame, logits_by_fold: dict[int, np.ndarray]
) -> None:
    """Score the fixed prior under the unchanged original strict metric and preserve every row."""
    if set(logits_by_fold) != {0, 1, 2}:
        raise ValueError("prior baseline requires all three fixed TRAIN producers")
    logits = torch.from_numpy(np.stack([logits_by_fold[fold] for fold in range(3)]))
    if (
        logits.shape != (3, 108)
        or logits.dtype != torch.float32
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("prior requires three finite FP32 logit vectors")
    with torch.inference_mode():
        posterior = logits.softmax(-1).numpy()
        phase = quantile(logits, phase_edges()).numpy()
        ttc = phase_to_finite_ttc(torch.from_numpy(phase)).numpy()
    folds = metadata.outer_fold.to_numpy()
    if set(folds) != {0, 1, 2} or not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("prior evaluation changed the original fold assignments")
    target = benchmark_phase(metadata.target_ttc_s.to_numpy(np.float64))
    prediction = ttc[folds].astype(np.float64)
    cost = 10000 * abs(benchmark_phase(prediction) - target)
    mass = strict_macro_mass(
        metadata.target_ttc_s.to_numpy(np.float64), metadata.sequence_id.to_numpy()
    )
    diagnostic.rows["train_prior_ttc_s"] = prediction
    diagnostic.rows["train_prior_phase"] = phase[folds]
    diagnostic.rows["train_prior_mid_loss"] = cost
    diagnostic.arrays["train_prior_posterior108_by_fold"] = posterior
    diagnostic.arrays["train_prior_logits_by_fold"] = logits.numpy()
    diagnostic.summary["train_prior_baseline"] = {
        "score": float(mass @ cost),
        "finite_fraction": 1.0,
        "prior_fitted_on": "each frozen fold TRAIN population only",
        "calibration_status": "not_calibrated",
        "new_optimizer_updates": 0,
        "outer_selection": False,
    }
