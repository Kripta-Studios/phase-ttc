"""Synthetic algebra and inference-schema checks; these are not scientific fits."""

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.models.full_cost_router_v11 import (
    DIAGNOSTICS,
    FullCostRouterV11,
    build_features,
    feature_names,
)
from e_jepa_ttc.models.three_expert_router import PHASE17_FEATURES


@pytest.mark.parametrize("new_experts", [("DIRECT",), ("PPS",), ("DIRECT", "PPS")])
def test_full_cost_solve_matches_unpenalized_intercept_system(new_experts):
    experts = ("A5", "C2F", "PAIR") + new_experts
    rng = np.random.default_rng(123)
    x = rng.normal(size=(90, len(feature_names(experts))))
    x[:, 0] = 3.0  # Constant feature must not penalize the intercept.
    costs = rng.uniform(0, 300, (90, len(experts)))
    mass = rng.uniform(0, 1, 90)
    mass[:5] = 0
    fitted = FullCostRouterV11.fit(x, costs, mass, experts=experts)
    design = np.column_stack((np.ones(90), (x - fitted.mean) / fitted.scale))
    weights = mass / mass.sum()
    penalty = np.eye(design.shape[1]) * 0.01
    penalty[0, 0] = 0
    target = (costs[:, 1:] - costs[:, :1]) / 100
    reference = np.linalg.solve(
        design.T @ (weights[:, None] * design) + penalty,
        design.T @ (weights[:, None] * target),
    )
    np.testing.assert_allclose(fitted.predict_regret(x)[:, 1:], design @ reference, atol=1e-12)
    shifted = FullCostRouterV11.fit(x, costs + 77, mass * 9, experts=experts)
    np.testing.assert_allclose(shifted.predict_regret(x), fitted.predict_regret(x), atol=1e-12)


def test_constant_costs_select_new_expert_and_ties_keep_frozen_order():
    x = np.zeros((8, 45))
    mass = np.ones(8)
    costs = np.tile([100, 90, 80, 70, 10], (8, 1))
    assert (FullCostRouterV11.fit(x, costs, mass).select(x) == 4).all()
    assert (FullCostRouterV11.fit(x, np.ones((8, 5)), mass).select(x) == 0).all()


def test_features_keep_history_and_recompute_new_differences():
    experts = ("A5", "C2F", "PAIR", "DIRECT", "PPS")
    old = pd.DataFrame(np.arange(34).reshape(2, 17), columns=PHASE17_FEATURES, index=["a", "b"])
    new = pd.DataFrame(index=old.index)
    for expert, phase in (("direct", -3.0), ("pps", 4.0)):
        new[f"{expert}_benchmark_phase"] = phase
        for diagnostic in DIAGNOSTICS:
            new[f"{expert}_{diagnostic}"] = 0.25
    result = build_features(old, new, experts)
    assert tuple(result.columns) == feature_names(experts)
    assert result.shape == (2, 45)
    np.testing.assert_array_equal(result.iloc[:, :17], old)
    assert (result["pps_minus_direct_phase"] == 7).all()
    np.testing.assert_array_equal(
        result["abs_direct_minus_a5_phase"], abs(-3 - old["a5_benchmark_phase"])
    )
    for invalid in (
        new.assign(target=1),
        new.iloc[::-1],
        pd.concat([new, new.iloc[:, :1]], axis=1),
    ):
        with pytest.raises(ValueError):
            build_features(old, invalid, experts)


@pytest.mark.parametrize("fault", ["ridge", "mass", "cost", "nonfinite", "experts"])
def test_invalid_fit_inputs_fail_closed(fault):
    x, costs, mass = np.zeros((8, 45)), np.ones((8, 5)), np.ones(8)
    kwargs = {}
    if fault == "ridge":
        kwargs["ridge"] = 0.1
    elif fault == "mass":
        mass[:] = 0
    elif fault == "cost":
        costs[0, 0] = -1
    elif fault == "nonfinite":
        x[0, 0] = np.nan
    else:
        kwargs["experts"] = ("A5", "C2F", "PAIR", "PPS", "DIRECT")
    with pytest.raises(ValueError):
        FullCostRouterV11.fit(x, costs, mass, **kwargs)
