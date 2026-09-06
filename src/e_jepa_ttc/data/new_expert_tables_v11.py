"""Consistent new-expert permutation control, reusing the proven cross-track derangement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase
from e_jepa_ttc.models.full_cost_router_v11 import build_features
from e_jepa_ttc.training.stage62_local_field import cross_track_derangement_indices


@dataclass(frozen=True)
class RouterInputs:
    """Inference fields and donor audit only; no target or true cost reaches the router."""

    features: pd.DataFrame
    expert_ttc: pd.DataFrame
    donor_indices: np.ndarray


def router_inputs(
    old_features: pd.DataFrame,
    new_features: pd.DataFrame,
    expert_ttc: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    experts: tuple[str, ...],
    permute_new: bool,
    seed: int = 7,
) -> RouterInputs:
    """Move new TTC, phase and all six diagnostics together; retain old inputs and row identities.

    The caller must separately validate nested producer ancestry for a fitting
    table. This function never substitutes pooled outer-OOF predictions for it.
    """
    if type(permute_new) is not bool or seed != 7:
        raise ValueError("new-expert control is a fixed seed7 transformation")
    index = old_features.index
    if (
        not index.is_unique
        or len(index) == 0
        or any(not frame.index.equals(index) for frame in (new_features, expert_ttc, metadata))
        or tuple(expert_ttc.columns) != experts
        or not {"sequence_id", "track_id"} <= set(metadata)
        or metadata[["sequence_id", "track_id"]].isna().to_numpy().any()
    ):
        raise ValueError("router tables must share exact unique token/producer row order")
    phases = benchmark_phase(expert_ttc.to_numpy(dtype=np.float64))
    # Build before mutation to reject extras, targets and invalid feature schemas.
    build_features(old_features, new_features, experts)
    for column, expert in enumerate(experts):
        source = old_features if column < 3 else new_features
        if not np.array_equal(
            source[f"{expert.lower()}_benchmark_phase"].to_numpy(), phases[:, column]
        ):
            raise ValueError("expert phase differs from the actually emitted TTC")
    donor = np.arange(len(index), dtype=np.int64)
    new, predictions = new_features.copy(), expert_ttc.copy()
    if permute_new:
        donor = cross_track_derangement_indices(
            sequence_ids=metadata.sequence_id.astype(str).tolist(),
            track_ids=metadata.track_id.astype(str).tolist(),
            seed=seed,
        )
        new.iloc[:, :] = new_features.to_numpy()[donor]
        predictions.iloc[:, 3:] = expert_ttc.iloc[:, 3:].to_numpy()[donor]
    return RouterInputs(build_features(old_features, new, experts), predictions, donor)


def full_cost_targets(inputs: RouterInputs, target_ttc: pd.Series) -> np.ndarray:
    """Recompute control costs against ORIGINAL targets, outside all inference features."""
    if not target_ttc.index.equals(inputs.features.index):
        raise ValueError("router supervision target identity differs")
    target = benchmark_phase(target_ttc.to_numpy(dtype=np.float64))
    prediction = benchmark_phase(inputs.expert_ttc.to_numpy(dtype=np.float64))
    return 10000 * abs(prediction - target[:, None])
