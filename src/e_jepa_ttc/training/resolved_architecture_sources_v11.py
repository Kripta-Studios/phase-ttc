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
        "present_bucket_normalized_TRAIN" if expected_expansion else "strict_original_TRAIN"
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
