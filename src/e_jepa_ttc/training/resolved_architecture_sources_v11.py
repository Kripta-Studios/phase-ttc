"""Rehydrate exact frozen TRAIN adapters, normalization and supervision for fixed jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest, verify
from e_jepa_ttc.data.event_context128 import RateNormalization
from e_jepa_ttc.data.event_context128_reader import CountSource, TrainingView
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.training.stage72_supervision import Supervision


@dataclass(frozen=True)
class ResolvedArchitectureSources:
    """Loaded TRAIN tensors and exact manifests; no outer rows or labels enter a batch."""

    source_sha256: str
    source: dict[str, Any]
    view: TrainingView
    normalization: RateNormalization
    supervision: Supervision


def load_architecture_sources(
    source_binding: dict[str, Any],
    reader: CountSource,
    roles: TrainRoles,
    *,
    expected_fold: int,
    expected_expansion: bool,
    source_role_sha256: str,
    nested_recipe: dict[str, Any] | None = None,
    producer_recipe: dict[str, Any] | None = None,
) -> ResolvedArchitectureSources:
    """Validate actual arrays against their signed fold source before reserving any fit."""
    source = json.loads(verify(source_binding).read_text(encoding="utf-8"))
    signature = source.pop("artifact_sha256", None)
    if (
        object_digest(source) != signature
        or source.get("kind") != "RESOLVED_ARCHITECTURE_TRAIN_SOURCE_VIEW"
        or source.get("outer_fold") != expected_fold
        or source.get("include_expansion") is not expected_expansion
        or source.get("source_role_sha256") != source_role_sha256
    ):
        raise ValueError("TRAIN source signature, role or prospective treatment fold changed")
    tokens = tuple(source["ordered_tokens"])
    excluded = tuple(source["excluded_sequences"])
    mapping = dict(roles.sequence_to_group)
    original_mask = reader.metadata.sequence_id.map(mapping).isin(roles.original)
    original = reader.metadata.loc[original_mask]
    held = original.loc[original.outer_fold == expected_fold]
    required_train = set(original.loc[original.outer_fold != expected_fold, "sample_token"])
    diagnostic_sequences: set[str] = set()
    if producer_recipe is not None:
        unsigned = {k: v for k, v in producer_recipe.items() if k != "artifact_sha256"}
        if (
            object_digest(unsigned) != producer_recipe.get("artifact_sha256")
            or producer_recipe.get("kind") != "CAPACITY_NESTED_PRODUCER_ROLES"
            or source.get("producer_roles_sha256") != producer_recipe["artifact_sha256"]
        ):
            raise ValueError("TRAIN source differs from its frozen internal diagnostic roles")
        matches = [r for r in producer_recipe["folds"] if r["outer_fold"] == expected_fold]
        if len(matches) != 1:
            raise ValueError("missing or duplicate outer diagnostic role")
        diagnostic_sequences = set(matches[0]["diagnostic_sequences"])
        diagnostic_tokens = set(
            original.loc[original.sequence_id.isin(diagnostic_sequences), "sample_token"]
        )
        if not diagnostic_sequences <= set(excluded) or diagnostic_tokens & set(tokens):
            raise ValueError("internal diagnostic acquisition leaked into optimization")
        required_train -= diagnostic_tokens
    elif source.get("producer_roles_sha256") is not None:
        raise ValueError("capacity source requires its prospectively frozen diagnostic roles")
    if nested_recipe is not None:
        recipe_unsigned = {
            key: value for key, value in nested_recipe.items() if key != "artifact_sha256"
        }
        if (
            object_digest(recipe_unsigned) != nested_recipe.get("artifact_sha256")
            or source.get("nested_recipe_sha256") != nested_recipe["artifact_sha256"]
            or source.get("inner_fold") != nested_recipe["inner_fold"]
            or nested_recipe.get("outer_fold") != expected_fold
            or nested_recipe.get("cache_sha256") != reader.identity
        ):
            raise ValueError("nested TRAIN source differs from its prospective recipe")
        for field in (
            "ordered_tokens",
            "excluded_sequences",
            "train_sequences",
            "allowed_train_groups",
            "excluded_groups",
            "training_view_sha256",
            "include_expansion",
        ):
            if source.get(field) != nested_recipe.get(field):
                raise ValueError(f"nested TRAIN source changed its prospective {field}")
        if not set(held.sequence_id.astype(str)) <= set(nested_recipe["outer_excluded_sequences"]):
            raise ValueError("nested recipe omits actual outer exclusions")
        if set(nested_recipe["inner_prediction_tokens"]) & set(tokens):
            raise ValueError("nested TRAIN source exposed inner prediction rows")
        required_train = set(
            original.loc[
                original.sequence_id.isin(nested_recipe["train_sequences"]), "sample_token"
            ]
        )
    elif source.get("inner_fold") is not None or source.get("nested_recipe_sha256") is not None:
        raise ValueError("nested TRAIN source requires its prospectively authorized recipe")
    if (
        held.empty
        or not required_train <= set(tokens)
        or set(held.sample_token) & set(tokens)
        or not set(held.sequence_id.astype(str)) <= set(excluded)
    ):
        raise ValueError(
            "initial TRAIN source omitted original rows or exposed its actual outer fold"
        )
    view = TrainingView(
        reader, tokens, roles, excluded_sequences=excluded, include_expansion=expected_expansion
    )
    if view.identity != source["training_view_sha256"]:
        raise ValueError("TRAIN view differs from the count cache, role map or original row order")
    sequence_index = reader.metadata.set_index("sample_token")
    sequences = sorted(set(sequence_index.loc[list(tokens), "sequence_id"].astype(str)))
    if (
        sequences != sorted(source["train_sequences"])
        or sorted({mapping[sequence] for sequence in sequences})
        != sorted(source["allowed_train_groups"])
        or sorted({mapping[sequence] for sequence in excluded}) != sorted(source["excluded_groups"])
    ):
        raise ValueError("TRAIN acquisition lineage differs from the resolved source manifest")
    norm = json.loads(verify(source["normalization"]).read_text(encoding="utf-8"))
    normalization = RateNormalization(**norm)
    if (
        normalization.training_view_sha256 != view.identity
        or len(normalization.mean) != 2
        or len(normalization.std) != 2
    ):
        raise ValueError("normalization does not describe this TRAIN polarity view")
    supervision = json.loads(verify(source["supervision"]).read_text(encoding="utf-8"))
    expected_estimand = (
        "present_bucket_normalized_TRAIN"
        if expected_expansion or producer_recipe is not None
        else "strict_original_TRAIN"
    )
    if (
        supervision.get("training_view_sha256") != view.identity
        or tuple(supervision["tokens"]) != tokens
        or supervision.get("estimand") != expected_estimand
    ):
        raise ValueError("supervision changed its TRAIN rows or named mass estimand")
    phase = torch.tensor(supervision["target_phase"], dtype=torch.float32)
    mass = torch.tensor(supervision["mass"], dtype=torch.float64)
    prior = torch.tensor(supervision["prior_bin_mass"], dtype=torch.float64)
    if (
        phase.shape != (len(tokens),)
        or mass.shape != phase.shape
        or prior.shape != (108,)
        or not all(bool(torch.isfinite(value).all()) for value in (phase, mass, prior))
        or not bool((mass > 0).all())
        or not bool((prior >= 0).all())
        or abs(float(mass.sum()) - 1) > 1e-12
        or abs(float(prior.sum()) - 1) > 1e-12
        or not 0 <= supervision["outside_support_mass"] <= 1
    ):
        raise ValueError("supervised phase, population mass or108-bin prior schema is invalid")
    identity = supervision["identity_sha256"]
    if len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity):
        raise ValueError("supervision lacks its exact original float64 source identity")
    return ResolvedArchitectureSources(
        signature,
        source,
        view,
        normalization,
        Supervision(tokens, phase, mass, prior, supervision["outside_support_mass"], identity),
    )
