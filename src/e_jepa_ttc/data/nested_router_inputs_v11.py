"""Assemble four/five-expert inference fields using physically validated nested table roles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas as pd

from e_jepa_ttc.data.frozen_expert_tables_v10 import FrozenTable
from e_jepa_ttc.data.new_expert_tables_v11 import RouterInputs, router_inputs
from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase
from e_jepa_ttc.models.full_cost_router_v11 import DIAGNOSTICS, validate_experts
from e_jepa_ttc.models.three_expert_router import PHASE17_FEATURES


def assemble_router_inputs(
    table: FrozenTable,
    new_rows: Mapping[str, pd.DataFrame],
    *,
    experts: tuple[str, ...],
    role: str,
    permute_new: bool,
) -> RouterInputs:
    """Require exact original tokens/targets and keep labels outside all inference features."""
    table.validate(role)
    validate_experts(experts)
    if set(new_rows) != set(experts[3:]):
        raise ValueError(
            "routing requires every technically valid canonical expert and no controls"
        )
    metadata = table.metadata.set_index("sample_token")
    index = metadata.index
    old = pd.DataFrame(table.inputs.features, columns=PHASE17_FEATURES, index=index)
    predictions = pd.DataFrame(table.inputs.expert_ttc, columns=experts[:3], index=index)
    new = pd.DataFrame(index=index)
    for expert in experts[3:]:
        supplied = new_rows[expert]
        if supplied.sample_token.duplicated().any() or not set(index) <= set(supplied.sample_token):
            raise ValueError("new expert rows omit or duplicate the required prediction universe")
        rows = supplied.set_index("sample_token").loc[index]
        if (
            not np.array_equal(rows.sequence_id.to_numpy(), metadata.sequence_id.to_numpy())
            or not np.array_equal(rows.track_id.to_numpy(), metadata.track_id.to_numpy())
            or not np.allclose(
                rows.target_ttc_s.to_numpy(np.float64),
                metadata.target_ttc.to_numpy(np.float64),
                rtol=0,
                atol=1e-13,
            )
        ):
            raise ValueError("new expert rows changed target/track identity")
        if role == "inner_oof" and (
            len(supplied) != len(metadata)
            or "prediction_role" not in rows
            or not rows.prediction_role.eq("genuinely_nested_inner_oof").all()
            or not np.array_equal(rows.inner_fold.to_numpy(), metadata.inner_fold.to_numpy())
        ):
            raise ValueError(
                "router TRAIN requires the exact new inner producers, never pooled outer-OOF"
            )
        flags = rows[["finite", "failure"]].to_numpy()
        if (
            not all(isinstance(value, (bool, np.bool_)) for value in flags.flat)
            or not rows.finite.all()
            or rows.failure.any()
        ):
            raise ValueError("failed canonical producer rows block their routing evidence")
        prediction = rows.prediction_ttc_s.to_numpy(np.float64)
        phase = benchmark_phase(prediction)
        if not np.array_equal(phase, rows.prediction_phase.to_numpy(np.float64)):
            raise ValueError("new expert phase differs from its emitted TTC")
        predictions[expert] = prediction
        new[f"{expert.lower()}_benchmark_phase"] = phase
        for diagnostic in DIAGNOSTICS:
            new[f"{expert.lower()}_{diagnostic}"] = rows[diagnostic].to_numpy(np.float64)
    return router_inputs(
        old,
        new,
        predictions,
        cast(pd.DataFrame, metadata[["sequence_id", "track_id"]]),
        experts=experts,
        permute_new=permute_new,
        seed=7,
    )
