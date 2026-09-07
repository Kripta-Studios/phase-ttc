"""Paired loss-based standalone decisions, preserving the historical nine-sequence estimand."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.evaluation.capacity_metrics import present_bucket_mass
from e_jepa_ttc.evaluation.exact_sequence_v10 import exact_sequence_diagnostic
from e_jepa_ttc.evaluation.risk_geometry_v10 import hierarchical_losses
from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass
from e_jepa_ttc.evaluation.stage70_76 import Evidence, decide

INITIAL_CONTRASTS = (
    ("S72-PPS-TRAJ", "S72-DIRECT"),
    ("S72-PPS-TRAJ", "S72-PPS-PAIR"),
    ("S72-PPS-TRAJ", "S72-PPS-TIMEPERM"),
    ("S73-PPS-A", "S72-PPS-TRAJ"),
    ("S73-PPS-D", "S72-PPS-TRAJ"),
    ("S73-PPS-AD", "S72-PPS-TRAJ"),
    ("S73-PPS-AD", "S73-PPS-D"),
    ("S73-PPS-AD", "S73-PPS-A"),
    ("S73-PPS-A", "S73-PPS-EXTRA"),
    ("S73-PPS-EXTRA", "S72-PPS-TRAJ"),
    ("S73-DIRECT-D", "S72-DIRECT"),
)


def paired_contrasts(
    metadata: pd.DataFrame,
    losses: np.ndarray,
    sampled_losses: np.ndarray,
    names: list[str],
    pairs: tuple[tuple[str, str], ...],
    *,
    expanded: bool = False,
) -> dict[str, Any]:
    """Use the same paired draws for prospectively named mechanism/compute contrasts."""
    losses = np.asarray(losses, dtype=np.float64)
    sampled_losses = np.asarray(sampled_losses, dtype=np.float64)
    if (
        losses.shape != (len(metadata), len(names))
        or sampled_losses.ndim != 2
        or sampled_losses.shape[1] != len(names)
        or sampled_losses.shape[0] < 2
        or len(set(names)) != len(names)
        or not np.isfinite(losses).all()
        or not np.isfinite(sampled_losses).all()
        or len(set(pairs)) != len(pairs)
        or any(left == right for left, right in pairs)
    ):
        raise ValueError("contrasts require aligned finite losses and shared paired draws")
    sequence = metadata.sequence_id.astype(str).to_numpy()
    sequences = sorted(set(sequence))
    if not expanded and len(sequences) != 9:
        raise ValueError("initial contrasts retain exactly nine original sequences")
    if expanded and len(sequences) < 3:
        raise ValueError("expanded contrasts require at least three sequences")
    mass_fn = present_bucket_mass if expanded else strict_macro_mass
    mass = mass_fn(metadata.target_ttc_s.to_numpy(np.float64), sequence)
    result = {}
    for left, right in pairs:
        key = f"{left}_minus_{right}"
        record: dict[str, Any] = {
            "candidate": left,
            "reference": right,
            "negative_favors_candidate": True,
            "confirmatory": False,
            "selects_candidate_identity": False,
        }
        if left not in names or right not in names:
            result[key] = {**record, "status": "UNAVAILABLE_OR_INVALID_ENDPOINT"}
            continue
        li, ri = names.index(left), names.index(right)
        delta = losses[:, li] - losses[:, ri]
        draws = sampled_losses[:, li] - sampled_losses[:, ri]
        per_sequence = np.array(
            [
                len(sequences) * float(mass[sequence == seq] @ delta[sequence == seq])
                for seq in sequences
            ]
        )
        result[key] = {
            **record,
            "status": "COMPLETE",
            "delta": float(mass @ delta),
            ("acquisition_ci95" if expanded else "hierarchical_ci95"): np.quantile(
                draws, [0.025, 0.975]
            ).tolist(),
            (
                "acquisition_fraction_negative" if expanded else "hierarchical_fraction_negative"
            ): float(np.mean(draws < 0)),
            "exact_sequence": None if expanded else exact_sequence_diagnostic(per_sequence),
            "sequence_order": sequences,
            "per_sequence_delta": per_sequence.tolist(),
        }
    return result


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
        "prospective_initial_contrasts": paired_contrasts(
            metadata, losses, sampled, ["RISK17", *model_names], INITIAL_CONTRASTS
        ),
        "bootstrap_loss_array": binding(output / "BOOTSTRAP_LOSSES.npy"),
        "bootstrap_draw_journal": binding(output / "HIERARCHICAL_DRAWS.jsonl"),
        "independent_confirmation": False,
        "historical_stage67_gate_changed": False,
        "fraction_negative_meaning": "paired bootstrap fraction, not Bayesian posterior",
    }
    result["artifact_sha256"] = object_digest(result)
    atomic_json(output / "COMPARISON.json", result)
    return result
