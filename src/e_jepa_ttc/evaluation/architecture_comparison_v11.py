"""Paired loss-based standalone decisions, preserving the historical nine-sequence estimand."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.evaluation.exact_sequence_v10 import exact_sequence_diagnostic
from e_jepa_ttc.evaluation.risk_geometry_v10 import hierarchical_losses
from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass
from e_jepa_ttc.evaluation.stage70_76 import Evidence, decide


def aligned_loss_vectors(
    metadata: pd.DataFrame, rows: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray] | None:
    """Require complete identities; a failed row blocks evidence, never changes its universe."""
    columns = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc_s"]
    if rows.sample_token.duplicated().any() or not metadata[columns].reset_index(drop=True).equals(
        rows[columns].reset_index(drop=True)
    ):
        raise ValueError("comparison rows differ from the complete canonical target universe")
    flags = rows[["finite", "failure"]].to_numpy()
    if not all(isinstance(value, (bool, np.bool_)) for value in flags.flat):
        raise ValueError("comparison requires explicit boolean failure flags")
    if not rows.finite.all() or rows.failure.any():
        return None
    target = benchmark_phase(metadata.target_ttc_s.to_numpy(np.float64))
    prediction = benchmark_phase(rows.prediction_ttc_s.to_numpy(np.float64))
    loss = 10000 * abs(prediction - target)
    if not np.allclose(loss, rows.mid_loss.to_numpy(np.float64), rtol=0, atol=1e-10):
        raise ValueError("stored per-row model losses differ from benchmark-phase recomputation")
    return loss, (np.sign(prediction) != np.sign(target)).astype(np.float64)


def compare_standalone_rows(
    metadata: pd.DataFrame,
    model_rows: Mapping[str, pd.DataFrame],
    risk17_ttc: np.ndarray,
    old_expert_ttc: np.ndarray,
    *,
    output: Path,
    check: Callable[[], None],
) -> dict[str, Any]:
    """Materialize one shared bootstrap, exact sequence uncertainty and unchanged S74 gates.

    Call only after the source-bound initial-stage evaluator has verified every
    endpoint and assembled all prescribed rows. This statistical primitive does
    not grant data access or authorize a fit by itself.
    """
    if output.exists():
        raise ValueError("comparison output already exists; immutable results need verification")
    sequences = metadata.sequence_id.astype(str).to_numpy()
    names = sorted(np.unique(sequences))
    if len(names) != 9 or metadata.sample_token.duplicated().any() or not model_rows:
        raise ValueError("standalone development comparison requires nine complete sequences")
    target = benchmark_phase(metadata.target_ttc_s.to_numpy(np.float64))
    risk = benchmark_phase(risk17_ttc)
    old = benchmark_phase(old_expert_ttc)
    if risk.shape != target.shape or old.shape != (len(target), 3):
        raise ValueError("RISK17 and three original experts require exact row alignment")
    mass = strict_macro_mass(metadata.target_ttc_s.to_numpy(np.float64), sequences)
    risk_loss = 10000 * abs(risk - target)
    risk_sign = (np.sign(risk) != np.sign(target)).astype(np.float64)
    old_oracle = (10000 * abs(old - target[:, None])).min(axis=1)
    vectors = {name: aligned_loss_vectors(metadata, rows) for name, rows in model_rows.items()}
    valid = {name: value for name, value in vectors.items() if value is not None}
    model_names = sorted(valid)
    losses = np.column_stack([risk_loss, *(valid[name][0] for name in model_names)])
    output.mkdir(parents=True)
    sampled, bootstrap = hierarchical_losses(metadata, losses, output, check)
    comparisons: dict[str, Any] = {}
    crucial = (metadata.target_ttc_s.to_numpy() > 0) & (metadata.target_ttc_s.to_numpy() <= 3)
    for name in model_rows:
        value = vectors[name]
        if value is None:
            comparisons[name] = {
                "technically_valid": False,
                "evidence": None,
                "decision": None,
                "reason": "complete endpoint has failed rows",
            }
            continue
        loss, wrong_sign = value
        delta = loss - risk_loss
        per_sequence = np.array(
            [9 * float(mass[sequences == seq] @ delta[sequences == seq]) for seq in names]
        )
        exact = exact_sequence_diagnostic(per_sequence)
        draws = sampled[:, 1 + model_names.index(name)] - sampled[:, 0]
        evidence = Evidence(
            delta_vs_risk17=float(mass @ delta),
            hierarchical_high=float(np.quantile(draws, 0.975)),
            sequence_high=exact["ci95"][1],
            sequence_wins=exact["sequence_wins"],
            sequence_count=9,
            sign_error_increase=float(mass @ (wrong_sign - risk_sign)),
            crucial_delta=float(mass[crucial] @ delta[crucial] / 0.5),
            finite_fraction=1.0,
            new_oracle_gain=float(mass @ (old_oracle - np.minimum(old_oracle, loss))),
        )
        comparisons[name] = {
            "technically_valid": True,
            "score": float(mass @ loss),
            "evidence": asdict(evidence),
            "decision": decide(evidence),
            "exact_sequence": exact,
            "hierarchical_ci95": [float(np.quantile(draws, 0.025)), evidence.hierarchical_high],
            "hierarchical_fraction_negative": float(np.mean(draws < 0)),
            "hierarchical_bootstrap_mean": float(draws.mean()),
            "hierarchical_bootstrap_median": float(np.median(draws)),
            "sequence_order": names,
            "per_sequence_delta": per_sequence.tolist(),
        }
    joint_oracle = np.min(
        np.column_stack([old_oracle, *(valid[n][0] for n in model_names)]), axis=1
    )
    result = {
        "kind": "STANDALONE_PAIRED_DEVELOPMENT_COMPARISON",
        "comparisons": comparisons,
        "risk17_score": float(mass @ risk_loss),
        "old_selector_oracle": float(mass @ old_oracle),
        "all_supplied_models_joint_oracle_gain": float(mass @ (old_oracle - joint_oracle)),
        "joint_oracle_model_names": model_names,
        "bootstrap": bootstrap,
        "bootstrap_columns": ["RISK17", *model_names],
        "bootstrap_loss_array": binding(output / "BOOTSTRAP_LOSSES.npy"),
        "bootstrap_draw_journal": binding(output / "HIERARCHICAL_DRAWS.jsonl"),
        "independent_confirmation": False,
        "historical_stage67_gate_changed": False,
        "fraction_negative_meaning": "paired bootstrap fraction, not Bayesian posterior",
    }
    result["artifact_sha256"] = object_digest(result)
    atomic_json(output / "COMPARISON.json", result)
    return result
