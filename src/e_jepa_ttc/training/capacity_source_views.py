"""Compile shared capacity TRAIN views through the existing normalization and source contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.data.event_context128_reader import CountSource, TrainingView
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.training.resolved_architecture_sources_v11 import load_architecture_sources
from e_jepa_ttc.training.stage72_supervision import train_supervision


def capacity_target_rows(
    cohort: dict[str, Any], metadata: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Join only the selected, clock-validated TRAIN rows for all capacity consumers."""
    unsigned = {k: v for k, v in cohort.items() if k != "artifact_sha256"}
    if object_digest(unsigned) != cohort.get("artifact_sha256"):
        raise ValueError("capacity target cohort signature changed")
    tokens = cohort["ordered_tokens"]
    for frame in (metadata, labels):
        if frame.sample_token.duplicated().any() or set(frame.sample_token) != set(tokens):
            raise ValueError("capacity target inputs must match every selected token exactly")
    rows = metadata.set_index("sample_token").loc[tokens].reset_index()
    truth = labels.set_index("sample_token").loc[tokens].reset_index()
    for name in ("sequence_id", "timestamp_us"):
        if not np.array_equal(rows[name].to_numpy(), truth[name].to_numpy()):
            raise ValueError("capacity target identity or time anchor changed")
    if set(rows.sequence_id) != set(cohort["selected_sequences"]):
        raise ValueError("capacity target rows contain a different acquisition pool")
    target = truth.ttc.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(target) & ((target < 0) | (target > 0.1))):
        raise ValueError("capacity targets contain invalid signed TTC")
    if any(
        np.asarray(pair).shape != (2,) or float(pair[1]) != target[i]
        for i, pair in enumerate(truth.frame_ttc)
    ):
        raise ValueError("capacity target is not attached to the later annotation")
    result = rows[["sample_token", "sequence_id", "track_id", "timestamp_us"]].copy()
    result["outer_fold"] = result.sequence_id.map(cohort["sequence_to_outer_fold"])
    result["acquisition_group"] = result.sequence_id.map(cohort["sequence_to_group"])
    if result[["outer_fold", "acquisition_group"]].isna().any().any():
        raise ValueError("capacity target acquisition or fold is unresolved")
    if "outer_fold" in rows and not np.array_equal(rows.outer_fold, result.outer_fold):
        raise ValueError("capacity target metadata changed its shared fold")
    result["target_ttc"] = target
    result["bucket"] = np.select(
        [target < 0, target <= 3, target <= 6], ["negative", "crucial", "small"], "large"
    )
    return result


def compile_capacity_outer_view(
    reader: CountSource,
    labels: pd.DataFrame,
    roles: TrainRoles,
    producer_recipe: dict[str, Any],
    *,
    outer_fold: int,
    source_role_sha256: str,
    source_bindings: dict[str, Any],
    output: Path,
    check: Callable[[], None],
) -> dict[str, Any]:
    """Exclude diagnostic/outer acquisitions before reading normalization observations.

    This compiles the subset view shared by DIRECT/PPS and all capacities. A uses
    the same TRAIN view. Diversity sources require their own resolved branch and
    are never silently appended to this subset-only producer.
    """
    check()
    unsigned = {k: v for k, v in producer_recipe.items() if k != "artifact_sha256"}
    if (
        object_digest(unsigned) != producer_recipe.get("artifact_sha256")
        or producer_recipe.get("kind") != "CAPACITY_NESTED_PRODUCER_ROLES"
    ):
        raise ValueError("capacity producer roles changed")
    matches = [row for row in producer_recipe["folds"] if row["outer_fold"] == outer_fold]
    if len(matches) != 1:
        raise ValueError("one resolved outer producer required")
    recipe = matches[0]
    excluded = tuple(recipe["all_fit_ancestor_exclusions"])
    fit_sequences = set(recipe["fit_sequences"])
    if (
        fit_sequences & set(excluded)
        or not set(recipe["diagnostic_sequences"]) <= set(excluded)
        or not set(recipe["outer_held_sequences"]) <= set(excluded)
        or set(recipe["normalization_and_pretext_exclusions"]) != set(excluded)
    ):
        raise ValueError("diagnostic or held acquisition leaked into producer roles")
    metadata = reader.metadata
    actual_held = set(metadata.loc[metadata.outer_fold == outer_fold, "sequence_id"].astype(str))
    actual_train = (
        set(metadata.sequence_id.astype(str)) - actual_held - set(recipe["diagnostic_sequences"])
    )
    if actual_held != set(recipe["outer_held_sequences"]) or actual_train != fit_sequences:
        raise ValueError("producer partition differs from the physical cohort cache")
    tokens = tuple(
        metadata.loc[metadata.sequence_id.isin(fit_sequences), "sample_token"].astype(str)
    )
    view = TrainingView(reader, tokens, roles, excluded_sequences=excluded, include_expansion=False)
    if output.resolve() != output.absolute() or output.is_symlink():
        raise ValueError("source publication path redirected")
    path = output / "SOURCE.json"
    if path.exists():
        loaded = load_architecture_sources(
            binding(path),
            reader,
            roles,
            expected_fold=outer_fold,
            expected_expansion=False,
            source_role_sha256=source_role_sha256,
            producer_recipe=producer_recipe,
        )
        if (
            loaded.view.identity != view.identity
            or loaded.source["source_bindings"] != source_bindings
        ):
            raise ValueError("existing capacity source belongs to another resolved input")
        return binding(path)
    if output.exists() and any(output.iterdir()):
        raise ValueError("partial source publication requires a separate visible attempt")
    # Target membership is checked before expensive normalization, and targets are
    # never supplied to the count reader or pretext objective.
    supervision = train_supervision(view, labels, expanded=True)
    normalization = view.fit_normalization(check)
    check()
    mapping = dict(roles.sequence_to_group)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "NORMALIZATION.json", asdict(normalization))
    atomic_json(
        output / "SUPERVISION.json",
        {
            "training_view_sha256": view.identity,
            "identity_sha256": supervision.identity_sha256,
            "tokens": list(tokens),
            "target_phase": supervision.target_phase.tolist(),
            "mass": supervision.mass.tolist(),
            "prior_bin_mass": supervision.prior_bin_mass.tolist(),
            "outside_support_mass": supervision.outside_support_mass,
            "estimand": "present_bucket_normalized_TRAIN",
        },
    )
    source = {
        "kind": "RESOLVED_ARCHITECTURE_TRAIN_SOURCE_VIEW",
        "outer_fold": outer_fold,
        "training_view_sha256": view.identity,
        "ordered_tokens": list(tokens),
        "allowed_train_groups": sorted({mapping[s] for s in fit_sequences}),
        "excluded_groups": sorted({mapping[s] for s in excluded}),
        "train_sequences": sorted(fit_sequences),
        "excluded_sequences": list(excluded),
        "include_expansion": False,
        "source_role_sha256": source_role_sha256,
        "producer_roles_sha256": producer_recipe["artifact_sha256"],
        "source_bindings": source_bindings,
        "normalization": binding(output / "NORMALIZATION.json"),
        "supervision": binding(output / "SUPERVISION.json"),
        "optimizer_updates": 0,
        "scientific_model_fit": False,
    }
    atomic_json(path, {**source, "artifact_sha256": object_digest(source)})
    load_architecture_sources(
        binding(path),
        reader,
        roles,
        expected_fold=outer_fold,
        expected_expansion=False,
        source_role_sha256=source_role_sha256,
        producer_recipe=producer_recipe,
    )
    return binding(path)
