"""Physical nested producer roles cannot be replaced with pooled outer prediction rows."""

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.data.frozen_expert_tables_v10 import create_record, load_table
from e_jepa_ttc.data.nested_router_inputs_v11 import assemble_router_inputs
from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass
from e_jepa_ttc.models.full_cost_router_v11 import DIAGNOSTICS
from e_jepa_ttc.models.three_expert_router import PHASE17_FEATURES


@pytest.fixture
def fixture(tmp_path):
    target = np.tile([2.0, 4.0, 8.0, -2.0], 2)
    meta = pd.DataFrame(
        {
            "sample_token": [f"r{i}" for i in range(8)],
            "sequence_id": ["s0"] * 4 + ["s1"] * 4,
            "track_id": ["a", "a", "b", "b"] * 2,
            "target_ttc": target,
            "outer_fold": 0,
            "inner_fold": [0] * 4 + [1] * 4,
            "role": "inner_oof",
        }
    )
    expert = np.column_stack([target + 0.01, target + 0.02, target + 0.03])
    phase = benchmark_phase(expert)
    features = np.zeros((8, 17), np.float64)
    for index, name in enumerate(("a5", "c2f", "pair")):
        features[:, PHASE17_FEATURES.index(f"{name}_benchmark_phase")] = phase[:, index]
    path = tmp_path / "table.npz"
    np.savez(
        path,
        features17=features,
        expert_phase=phase,
        expert_ttc=expert,
        target_phase=benchmark_phase(target),
        global_mass=strict_macro_mass(target, meta.sequence_id.to_numpy()),
    )
    meta_path = tmp_path / "table.csv"
    meta.to_csv(meta_path, index=False)
    table = load_table(create_record(path, meta_path, 0, "inner_oof"), role="inner_oof")
    new = meta.rename(columns={"target_ttc": "target_ttc_s"}).assign(
        prediction_ttc_s=target + 0.04,
        prediction_phase=benchmark_phase(target + 0.04),
        finite=True,
        failure=False,
        prediction_role="genuinely_nested_inner_oof",
    )
    for diagnostic in DIAGNOSTICS:
        new[diagnostic] = np.arange(8) / 8
    return table, new


def test_adapter_keeps_targets_outside_inputs_and_permutes_complete_new_bundle(fixture):
    table, rows = fixture
    experts = ("A5", "C2F", "PAIR", "PPS")
    ordinary = assemble_router_inputs(
        table, {"PPS": rows}, experts=experts, role="inner_oof", permute_new=False
    )
    control = assemble_router_inputs(
        table, {"PPS": rows}, experts=experts, role="inner_oof", permute_new=True
    )
    assert ordinary.features.shape == (8, 30)
    assert not any("target" in name or "token" in name for name in ordinary.features)
    np.testing.assert_array_equal(
        control.expert_ttc.PPS, ordinary.expert_ttc.PPS.to_numpy()[control.donor_indices]
    )
    np.testing.assert_array_equal(control.features.iloc[:, :17], ordinary.features.iloc[:, :17])


@pytest.mark.parametrize(
    "damage", ["outer_oof", "inner_fold", "target", "failed", "phase", "dropped"]
)
def test_adapter_rejects_changed_or_failed_nested_producers(fixture, damage):
    table, rows = fixture
    if damage == "outer_oof":
        rows["prediction_role"] = "outer_oof"
    elif damage == "inner_fold":
        rows["inner_fold"] = 2
    elif damage == "target":
        rows.loc[0, "target_ttc_s"] += 0.1
    elif damage == "failed":
        rows.loc[0, "failure"] = True
    elif damage == "phase":
        rows.loc[0, "prediction_phase"] += 0.001
    else:
        rows = rows.iloc[:-1]
    with pytest.raises(ValueError):
        assemble_router_inputs(
            table,
            {"PPS": rows},
            experts=("A5", "C2F", "PAIR", "PPS"),
            role="inner_oof",
            permute_new=False,
        )
