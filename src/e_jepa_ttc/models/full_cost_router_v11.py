"""Four/five-expert full-cost ridge; historical PHASE17 and FullRegretRidge remain intact."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from e_jepa_ttc.models.full_regret_router import FullRegretRidge
from e_jepa_ttc.models.three_expert_router import PHASE17_FEATURES

DIAGNOSTICS = (
    "negative_phase_probability",
    "posterior_entropy_normalized",
    "phase_q10",
    "phase_q90",
    "geometric_validity_fraction",
    "score_correction_rms",
)


def validate_experts(experts: tuple[str, ...]) -> None:
    """Only old experts plus the actually valid canonical DIRECT/PPS models are eligible."""
    if experts not in (
        ("A5", "C2F", "PAIR", "DIRECT"),
        ("A5", "C2F", "PAIR", "PPS"),
        ("A5", "C2F", "PAIR", "DIRECT", "PPS"),
    ):
        raise ValueError("full-cost router requires four/five experts in frozen order")


def feature_names(experts: tuple[str, ...]) -> tuple[str, ...]:
    """Preserve PHASE17, then new phases, signed differences, absolute differences, diagnostics."""
    validate_experts(experts)
    new = experts[3:]
    differences = tuple(
        f"{experts[right].lower()}_minus_{experts[left].lower()}_phase"
        for right in range(3, len(experts))
        for left in range(right)
    )
    return (
        PHASE17_FEATURES
        + tuple(f"{name.lower()}_benchmark_phase" for name in new)
        + differences
        + tuple(f"abs_{name}" for name in differences)
        + tuple(f"{name.lower()}_{diagnostic}" for name in new for diagnostic in DIAGNOSTICS)
    )


def build_features(old: pd.DataFrame, new: pd.DataFrame, experts: tuple[str, ...]) -> pd.DataFrame:
    """Require aligned producer rows and inference-only fields; never accept a target column."""
    names = feature_names(experts)
    if (
        tuple(old.columns) != PHASE17_FEATURES
        or not old.index.equals(new.index)
        or not old.index.is_unique
    ):
        raise ValueError("PHASE17 order or producer row identity mismatch")
    required = tuple(f"{name.lower()}_benchmark_phase" for name in experts[3:]) + tuple(
        f"{name.lower()}_{diagnostic}" for name in experts[3:] for diagnostic in DIAGNOSTICS
    )
    if not new.columns.is_unique or set(new.columns) != set(required):
        raise ValueError(
            "new expert inference fields missing or extra target/identity fields supplied"
        )
    result = old.copy()
    for expert in experts[3:]:
        name = f"{expert.lower()}_benchmark_phase"
        result[name] = new[name].to_numpy(dtype=np.float64)
    for right in range(3, len(experts)):
        for left in range(right):
            a, b = experts[left].lower(), experts[right].lower()
            result[f"{b}_minus_{a}_phase"] = (
                result[f"{b}_benchmark_phase"] - result[f"{a}_benchmark_phase"]
            )
    differences = [
        name
        for name in names[len(PHASE17_FEATURES) + len(experts) - 3 :]
        if name.startswith("abs_")
    ]
    for name in differences:
        result[name] = result[name[4:]].abs()
    for name in required:
        if name not in result:
            result[name] = new[name].to_numpy(dtype=np.float64)
    result = result.loc[:, list(names)].astype(np.float64)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("nonfinite router features are an evidence failure")
    return result


@dataclass(frozen=True)
class FullCostRouterV11(FullRegretRidge):
    """Generalize the proven weighted ridge and reuse its cost prediction and tie selection."""

    experts: tuple[str, ...] = ("A5", "C2F", "PAIR", "DIRECT", "PPS")

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        losses: np.ndarray,
        mass: np.ndarray,
        *,
        ridge: float = 0.01,
        experts: tuple[str, ...] = ("A5", "C2F", "PAIR", "DIRECT", "PPS"),
    ) -> FullCostRouterV11:
        """Fit E−1 relative costs in one float64 solve on exclusively nested TRAIN producers."""
        expected_features = len(feature_names(experts))
        x, costs, weight = (
            np.asarray(value, dtype=np.float64) for value in (features, losses, mass)
        )
        if (
            x.ndim != 2
            or x.shape[1] != expected_features
            or len(x) == 0
            or costs.shape != (len(x), len(experts))
        ):
            raise ValueError("full-cost feature/loss schema mismatch")
        if (
            weight.shape != (len(x),)
            or np.any(weight < 0)
            or weight.sum() <= 0
            or np.any(costs < 0)
        ):
            raise ValueError("full-cost training masses/costs invalid")
        if ridge != 0.01 or any(not np.isfinite(value).all() for value in (x, costs, weight)):
            raise ValueError("finite inputs and the fixed .01 ridge are required")
        weight = weight / weight.sum()
        mean = np.sum(weight[:, None] * x, axis=0)
        scale = np.maximum(np.sqrt(np.sum(weight[:, None] * np.square(x - mean), axis=0)), 1e-8)
        z = (x - mean) / scale
        targets = (costs[:, 1:] - costs[:, :1]) / 100.0
        intercept = np.sum(weight[:, None] * targets, axis=0)
        lhs = z.T @ (weight[:, None] * z) + ridge * np.eye(expected_features)
        rhs = z.T @ (weight[:, None] * (targets - intercept))
        coefficient = np.linalg.solve(lhs, rhs)
        if not np.allclose(lhs @ coefficient - rhs, 0.0, rtol=0, atol=1e-10):
            raise ArithmeticError("full-cost ridge solve residual exceeds1e-10")
        return cls(mean, scale, coefficient, intercept, ridge, experts)
