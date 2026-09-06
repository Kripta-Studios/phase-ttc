"""Row-preserving phase, posterior, geometry and old-hull diagnostics in float64."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.evaluation.stage63_65 import BUCKETS, benchmark_phase, strict_macro_mass
from e_jepa_ttc.models.phase_distribution_v11 import crps, quantile, validate_edges


@dataclass(frozen=True)
class DiagnosticExport:
    """Aligned tables and safe numeric arrays; invalid model rows are never omitted."""

    rows: pd.DataFrame
    arrays: dict[str, np.ndarray]
    summary: dict[str, Any]


def posterior_diagnostics(output: Mapping[str, np.ndarray], edges: np.ndarray) -> DiagnosticExport:
    """Summarize inference fields without accessing targets or assigning calibration success."""
    expected = {
        "logits",
        "phase",
        "ttc",
        "direct_logits",
        "score_correction",
        "evidence",
        "observability",
    }
    if set(output) != expected:
        raise ValueError("inference diagnostic field schema mismatch")
    arrays = {key: np.asarray(value).copy() for key, value in output.items()}
    n = len(arrays["ttc"])
    shapes = {
        "logits": (n, 108),
        "phase": (n,),
        "ttc": (n,),
        "direct_logits": (n, 108),
        "score_correction": (n, 108),
        "evidence": (n, 4, 108),
        "observability": (n, 4, 108),
    }
    if n == 0 or any(arrays[key].shape != shape for key, shape in shapes.items()):
        raise ValueError("108-bin row-aligned inference arrays required")
    if arrays["observability"].dtype != bool:
        raise ValueError("geometry observability must remain an explicit boolean mask")
    for key in expected - {"observability"}:
        arrays[key] = arrays[key].astype(np.float64)
    edge_tensor = torch.as_tensor(np.asarray(edges, dtype=np.float64))
    validate_edges(edge_tensor)
    if edge_tensor.shape != (109,):
        raise ValueError("exactly109 phase edges required")
    valid = np.ones(n, dtype=bool)
    for key in expected - {"observability"}:
        valid &= np.isfinite(arrays[key].reshape(n, -1)).all(axis=1)
    valid &= (arrays["ttc"] < 0) | (arrays["ttc"] > 0.1)
    valid &= np.abs(arrays["ttc"]) <= 60.0
    valid &= (arrays["phase"] >= float(edge_tensor[0])) & (
        arrays["phase"] <= float(edge_tensor[-1])
    )
    rows = pd.DataFrame(
        {
            "finite": valid,
            "failure": ~valid,
            "prediction_ttc_s": arrays["ttc"],
            "raw_posterior_median_phase": arrays["phase"],
            "calibration_status": "not_calibrated",
        }
    )
    names = [
        "prediction_phase",
        "negative_phase_probability",
        "posterior_entropy_normalized",
        "left_tail_mass",
        "right_tail_mass",
        "geometric_validity_fraction",
        "direct_logits_rms",
        "score_correction_rms",
        "geometric_evidence_rms",
        "phase_median_float64",
        "median_rounding_delta",
    ]
    levels = {
        "phase_q025": 0.025,
        "phase_q10": 0.1,
        "phase_q25": 0.25,
        "phase_q75": 0.75,
        "phase_q90": 0.9,
        "phase_q975": 0.975,
    }
    for name in names + list(levels):
        rows[name] = np.nan
    probability = np.full((n, 108), np.nan, dtype=np.float64)
    if valid.any():
        logits = torch.from_numpy(arrays["logits"][valid])
        p = logits.softmax(-1).numpy()
        probability[valid] = p
        edge = edge_tensor.numpy()
        rows.loc[valid, "prediction_phase"] = benchmark_phase(arrays["ttc"][valid])
        fraction_negative = np.clip((0 - edge[:-1]) / np.diff(edge), 0, 1)
        rows.loc[valid, "negative_phase_probability"] = p @ fraction_negative
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy_terms = np.where(p > 0, p * np.log(p), 0)
        rows.loc[valid, "posterior_entropy_normalized"] = -entropy_terms.sum(axis=1) / np.log(108)
        rows.loc[valid, "left_tail_mass"] = p[:, :4].sum(axis=1)
        rows.loc[valid, "right_tail_mass"] = p[:, -8:].sum(axis=1)
        rows.loc[valid, "geometric_validity_fraction"] = arrays["observability"][valid].mean(
            axis=(1, 2)
        )
        for field, name in (
            ("direct_logits", "direct_logits_rms"),
            ("score_correction", "score_correction_rms"),
            ("evidence", "geometric_evidence_rms"),
        ):
            rows.loc[valid, name] = np.sqrt(
                np.mean(arrays[field][valid].reshape(valid.sum(), -1) ** 2, axis=1)
            )
        median = quantile(logits, edge_tensor).numpy()
        rows.loc[valid, "phase_median_float64"] = median
        rows.loc[valid, "median_rounding_delta"] = arrays["phase"][valid] - median
        for name, level in levels.items():
            rows.loc[valid, name] = quantile(logits, edge_tensor, level).numpy()
    arrays["posterior108"] = probability
    arrays["phase_edges"] = edge_tensor.numpy().copy()
    centers = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2
    observable = arrays["observability"] & np.isfinite(arrays["evidence"])
    peaks = np.argmax(np.where(observable, arrays["evidence"], -np.inf), axis=2)
    peak_phase = np.where(observable.any(axis=2), centers[peaks], np.nan)
    arrays["pair_evidence_peak_phase"] = peak_phase
    for pair in range(4):
        rows[f"pair{pair}_observable_fraction"] = observable[:, pair].mean(axis=1)
        rows[f"pair{pair}_evidence_peak_phase"] = peak_phase[:, pair]
    disagreements = []
    for left in range(4):
        for right in range(left + 1, 4):
            delta = np.abs(peak_phase[:, left] - peak_phase[:, right])
            rows[f"pair{left}_{right}_peak_disagreement"] = delta
            disagreements.append(delta)
    arrays["pair_disagreement"] = np.stack(disagreements, axis=1)
    return DiagnosticExport(
        rows,
        arrays,
        {
            "rows": n,
            "finite_rows": int(valid.sum()),
            "failure_rows": int((~valid).sum()),
            "calibration_status": "not_calibrated",
            "reporting_precision": "float64",
            "geometry_mask_meaning": (
                "physical depth and crop-overlap validity; not foreground truth"
            ),
            "interval_domain": "phase; signed TTC inversion is discontinuous at zero",
        },
    )


def evaluate_rows(
    metadata: pd.DataFrame,
    output: Mapping[str, np.ndarray],
    edges: np.ndarray,
    old_predictions: pd.DataFrame,
) -> DiagnosticExport:
    """Join exact token/target identity, retain failures and score the full original estimand."""
    columns = ("sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc_s")
    if not set(columns) <= set(metadata) or metadata[list(columns)].isna().to_numpy().any():
        raise ValueError("complete canonical target/identity metadata required")
    if metadata.sample_token.duplicated().any() or not metadata.index.is_unique:
        raise ValueError("duplicate diagnostic row identity")
    if tuple(old_predictions.columns) != ("A5", "C2F", "PAIR") or not old_predictions.index.equals(
        pd.Index(metadata.sample_token)
    ):
        raise ValueError("old expert predictions must be indexed in exact canonical token order")
    diagnostics = posterior_diagnostics(output, edges)
    if len(diagnostics.rows) != len(metadata):
        raise ValueError("prediction and target row counts differ")
    rows = pd.concat([metadata[list(columns)].reset_index(drop=True), diagnostics.rows], axis=1)
    target = benchmark_phase(rows.target_ttc_s.to_numpy())
    mass = strict_macro_mass(rows.target_ttc_s.to_numpy(), rows.sequence_id.to_numpy())
    old_ttc = old_predictions.to_numpy(dtype=np.float64)
    old_phase = benchmark_phase(old_ttc)
    old_cost = 10000 * abs(old_phase - target[:, None])
    point = rows.prediction_phase.to_numpy()
    cost = 10000 * abs(point - target)
    lo, hi = old_phase.min(axis=1), old_phase.max(axis=1)
    rows["target_phase"], rows["global_mass"] = target, mass
    rows["bucket"] = "unassigned"
    for label, lower, upper, _ in BUCKETS:
        rows.loc[(rows.target_ttc_s > lower) & (rows.target_ttc_s <= upper), "bucket"] = str(label)
    rows["signed_phase_error"] = point - target
    rows["finite_clip_phase_delta"] = point - rows.raw_posterior_median_phase
    rows["signed_ttc_error_s"] = rows.prediction_ttc_s - rows.target_ttc_s
    rows["mid_loss"] = cost
    rows["sign_error"] = np.where(rows.finite, np.sign(point) != np.sign(target), np.nan)
    rows["old_selector_oracle_loss"] = old_cost.min(axis=1)
    rows["old_hull_oracle_loss"] = 10000 * np.maximum(np.maximum(lo - target, target - hi), 0)
    rows["old_plus_new_oracle_loss"] = np.minimum(old_cost.min(axis=1), cost)
    rows["old_hull_low"], rows["old_hull_high"] = lo, hi
    rows["hull_escape"] = np.where(rows.finite, (point < lo) | (point > hi), np.nan)
    rows["escape_beats_old_selector"] = np.where(
        rows.finite, ((point < lo) | (point > hi)) & (cost < old_cost.min(axis=1)), np.nan
    )
    rows["target_outside_phase_support"] = (target < edges[0]) | (target > edges[-1])
    valid = rows.finite.to_numpy(dtype=bool)
    rows["phase_crps"] = np.nan
    rows["phase_pit"] = np.nan
    rows["phase_density_nll"] = np.nan
    if valid.any():
        rows.loc[valid, "phase_crps"] = crps(
            torch.from_numpy(diagnostics.arrays["logits"][valid]),
            torch.from_numpy(target[valid]),
            torch.as_tensor(edges, dtype=torch.float64),
        ).numpy()
        probability = diagnostics.arrays["posterior108"][valid]
        fractions = np.clip((target[valid, None] - edges[:-1]) / np.diff(edges), 0, 1)
        rows.loc[valid, "phase_pit"] = (probability * fractions).sum(axis=1)
        bins = np.clip(np.searchsorted(edges, target[valid], side="right") - 1, 0, 107)
        density = probability[np.arange(valid.sum()), bins] / np.diff(edges)[bins]
        inside = (target[valid] >= edges[0]) & (target[valid] <= edges[-1])
        density[~inside] = 0
        with np.errstate(divide="ignore"):
            rows.loc[valid, "phase_density_nll"] = -np.log(density)
    for coverage, lower, upper in (
        (50, "phase_q25", "phase_q75"),
        (80, "phase_q10", "phase_q90"),
        (95, "phase_q025", "phase_q975"),
    ):
        rows[f"phase_coverage_{coverage}"] = np.where(
            valid, (target >= rows[lower]) & (target <= rows[upper]), np.nan
        )
        rows[f"phase_width_{coverage}"] = rows[upper] - rows[lower]
    for index, name in enumerate(old_predictions.columns):
        rows[f"{name}_ttc_s"] = old_ttc[:, index]
        rows[f"{name}_phase"] = old_phase[:, index]
        rows[f"{name}_oracle_loss"] = old_cost[:, index]
    summary = dict(diagnostics.summary)
    summary.update(
        {
            "metric": "strict_sequence_macro_signed_garl_release_MiD",
            "score": float(mass @ cost) if valid.all() else None,
            "score_status": "complete" if valid.all() else "invalid_model_rows_no_omission",
            "old_selector_oracle": float(mass @ old_cost.min(axis=1)),
            "old_hull_oracle": float(mass @ rows.old_hull_oracle_loss.to_numpy()),
            "old_plus_new_oracle": float(mass @ rows.old_plus_new_oracle_loss.to_numpy())
            if valid.all()
            else None,
            "outside_phase_support_mass": float(
                mass @ rows.target_outside_phase_support.to_numpy()
            ),
            "weighted_sign_error": float(mass @ rows.sign_error.to_numpy())
            if valid.all()
            else None,
            "weighted_hull_escape": float(mass @ rows.hull_escape.to_numpy())
            if valid.all()
            else None,
            "phase_interval_coverage": {
                str(level): float(mass @ rows[f"phase_coverage_{level}"].to_numpy())
                if valid.all()
                else None
                for level in (50, 80, 95)
            },
        }
    )
    summary["phase_nll_infinite_rows"] = int(np.isinf(rows.phase_density_nll.to_numpy()).sum())
    summary["phase_nll_mean"] = (
        float(mass @ rows.phase_density_nll.to_numpy())
        if valid.all() and np.isfinite(rows.phase_density_nll.to_numpy()).all()
        else None
    )
    summary["group_statistics"] = {}
    for fields in (("sequence_id",), ("sequence_id", "track_id"), ("sequence_id", "bucket")):
        groups = []
        for identity, group in rows.groupby(list(fields), sort=True):
            indices = group.index.to_numpy()
            local_mass = mass[indices] / mass[indices].sum()
            groups.append(
                {
                    "identity": list(identity) if isinstance(identity, tuple) else [identity],
                    "rows": len(group),
                    "failure_rows": int(group.failure.sum()),
                    "population_mass": float(mass[indices].sum()),
                    "conditional_weighted_MiD": float(local_mass @ cost[indices])
                    if group.finite.all()
                    else None,
                    "unweighted_MiD": float(cost[indices].mean()) if group.finite.all() else None,
                }
            )
        summary["group_statistics"]["/".join(fields)] = groups
    summary["signed_phase_error_correlation"] = {
        name: float(np.corrcoef(point - target, old_phase[:, index] - target)[0, 1])
        if valid.all() and np.std(point - target) > 0 and np.std(old_phase[:, index] - target) > 0
        else None
        for index, name in enumerate(old_predictions.columns)
    }
    diagnostics.arrays.update(
        old_expert_ttc=old_ttc,
        old_expert_phase=old_phase,
        old_expert_loss=old_cost,
        target_phase=target,
        global_mass=mass,
    )
    return DiagnosticExport(rows, diagnostics.arrays, summary)
