"""Permutation moves actual new actions consistently while leaving targets and old experts fixed."""

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.data.new_expert_tables_v11 import full_cost_targets, router_inputs
from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase
from e_jepa_ttc.models.full_cost_router_v11 import DIAGNOSTICS
from e_jepa_ttc.models.three_expert_router import PHASE17_FEATURES


@pytest.fixture
def tables():
    experts = ("A5", "C2F", "PAIR", "DIRECT", "PPS")
    index = pd.Index([f"t{i}" for i in range(8)])
    predictions = pd.DataFrame(np.arange(40).reshape(8, 5) / 10 + 1, columns=experts, index=index)
    phases = benchmark_phase(predictions.to_numpy())
    old = pd.DataFrame(np.zeros((8, 17)), columns=PHASE17_FEATURES, index=index)
    for i, expert in enumerate(experts[:3]):
        old[f"{expert.lower()}_benchmark_phase"] = phases[:, i]
    new = pd.DataFrame(index=index)
    for i, expert in enumerate(experts[3:], start=3):
        new[f"{expert.lower()}_benchmark_phase"] = phases[:, i]
        for diagnostic in DIAGNOSTICS:
            new[f"{expert.lower()}_{diagnostic}"] = np.arange(8) / 8
    metadata = pd.DataFrame(
        dict(sequence_id=["s0"] * 4 + ["s1"] * 4, track_id=["a", "a", "b", "b"] * 2), index=index
    )
    return old, new, predictions, metadata, experts


def test_control_moves_complete_new_bundle_and_recomputes_costs(tables):
    old, new, predictions, metadata, experts = tables
    control = router_inputs(old, new, predictions, metadata, experts=experts, permute_new=True)
    donor = control.donor_indices
    assert np.all(donor != np.arange(8))
    assert np.array_equal(metadata.sequence_id.to_numpy(), metadata.sequence_id.to_numpy()[donor])
    assert np.all(metadata.track_id.to_numpy() != metadata.track_id.to_numpy()[donor])
    pd.testing.assert_frame_equal(control.features.iloc[:, :17], old)
    pd.testing.assert_frame_equal(control.expert_ttc.iloc[:, :3], predictions.iloc[:, :3])
    np.testing.assert_array_equal(control.expert_ttc.iloc[:, 3:], predictions.iloc[donor, 3:])
    for name in new:
        np.testing.assert_array_equal(control.features[name], new[name].to_numpy()[donor])
    np.testing.assert_array_equal(
        control.features.direct_minus_a5_phase,
        new.direct_benchmark_phase.to_numpy()[donor] - old.a5_benchmark_phase.to_numpy(),
    )
    target = pd.Series(np.linspace(1, 2, 8), index=old.index)
    losses = full_cost_targets(control, target)
    np.testing.assert_array_equal(
        losses,
        10000
        * abs(
            benchmark_phase(control.expert_ttc.to_numpy())
            - benchmark_phase(target.to_numpy())[:, None]
        ),
    )
    plain = router_inputs(old, new, predictions, metadata, experts=experts, permute_new=False)
    assert not np.array_equal(losses[:, 3:], full_cost_targets(plain, target)[:, 3:])
    np.testing.assert_array_equal(predictions.to_numpy(), np.arange(40).reshape(8, 5) / 10 + 1)


@pytest.mark.parametrize("fault", ["order", "phase", "target", "single_track"])
def test_invalid_control_inputs_fail_without_search_or_fallback(tables, fault):
    old, new, predictions, metadata, experts = tables
    if fault == "order":
        new = new.iloc[::-1]
    elif fault == "phase":
        new.loc[new.index[0], "direct_benchmark_phase"] += 0.001
    elif fault == "target":
        new["target_ttc"] = 1.0
    else:
        metadata["track_id"] = "same"
    with pytest.raises(ValueError):
        router_inputs(old, new, predictions, metadata, experts=experts, permute_new=True)
