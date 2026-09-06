"""Prospective nested role recipes; no normalization, labels or neural fitting before S75 gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.data.canonical_token_identity import hash_sorted_token_strings
from e_jepa_ttc.data.event_context128_reader import CountObservation, TrainingView
from e_jepa_ttc.data.stage70_roles import TrainRoles


@dataclass
class CountMetadataCatalog:
    """Verified cache identity and row metadata, deliberately incapable of reading events."""

    metadata: pd.DataFrame
    identity: str

    def __post_init__(self) -> None:
        if (
            set(self.metadata.columns) != {"sample_token", "sequence_id", "track_id", "outer_fold"}
            or self.metadata.sample_token.duplicated().any()
            or self.metadata.isna().to_numpy().any()
            or len(self.identity) != 64
        ):
            raise ValueError("nested catalog requires unique target-free count metadata")

    def observation(self, token: str) -> CountObservation:
        """Event reads require the actual verified CountReader after conditional authorization."""
        raise PermissionError("prospective nested recipe inventory cannot read event observations")


def nested_source_recipe(
    catalog: CountMetadataCatalog,
    original: pd.DataFrame,
    parent: dict[str, Any],
    protocol: dict[str, Any],
    roles: TrainRoles,
    *,
    outer: int,
    inner: int,
    expanded: bool,
) -> dict[str, Any]:
    """Preserve each historical four-train/two-dev split and safe fixed expansion additions."""
    if object_digest({k: v for k, v in parent.items() if k != "artifact_sha256"}) != parent.get(
        "artifact_sha256"
    ):
        raise ValueError("parent source manifest signature changed")
    if protocol.get("outer_fold") != outer or protocol.get("inner_fold") != inner:
        raise ValueError("historical nested protocol belongs to another producer")
    splits = protocol.get("folds", [])
    if len(splits) != 1:
        raise ValueError("nested protocol must describe exactly one original split")
    split = splits[0]
    train, dev = set(split["train_sequence_ids"]), set(split["dev_sequence_ids"])
    outer_dev = set(original.loc[original.outer_fold == outer, "sequence_id"])
    original_sequences = set(original.sequence_id)
    if (
        len(train) != 4
        or len(dev) != 2
        or train & dev
        or (train | dev) & outer_dev
        or train | dev | outer_dev != original_sequences
    ):
        raise ValueError("nested recipe changed the original four/two/outer sequence partition")
    for role, sequences in (("train", train), ("dev", dev)):
        rows = original.loc[original.sequence_id.isin(sequences)]
        if (
            len(rows) != split[f"{role}_rows"]
            or hash_sorted_token_strings(rows.sample_token) != split[f"{role}_sample_tokens_sha256"]
        ):
            raise ValueError("nested recipe token membership differs from its historical producer")
    if parent.get("outer_fold") != outer or parent.get("include_expansion") is not expanded:
        raise ValueError("nested recipe inherited a different canonical outer TRAIN pool")
    parent_tokens = tuple(parent["ordered_tokens"])
    index = catalog.metadata.set_index("sample_token")
    if len(set(parent_tokens)) != len(parent_tokens) or not set(parent_tokens) <= set(index.index):
        raise ValueError("parent TRAIN rows are duplicated or absent from the fixed cache")
    mapping = dict(roles.sequence_to_group)
    parent_sequences = set(index.loc[list(parent_tokens), "sequence_id"].astype(str))
    roles.check_fit_sequences(
        tuple(parent_sequences), excluded_sequences=tuple(outer_dev), include_expansion=expanded
    )
    allowed = set(train)
    if expanded:
        allowed |= {
            sequence for sequence in parent_sequences if mapping[sequence] in roles.expansion
        }
    tokens = tuple(
        token for token in parent_tokens if str(index.loc[token, "sequence_id"]) in allowed
    )
    if not set(original.loc[original.sequence_id.isin(train), "sample_token"]) <= set(tokens):
        raise ValueError("nested source omitted prescribed original TRAIN rows")
    excluded = tuple(sorted(outer_dev | dev))
    view = TrainingView(
        catalog, tokens, roles, excluded_sequences=excluded, include_expansion=expanded
    )
    return {
        "kind": "PROSPECTIVE_NESTED_NEW_ARCHITECTURE_SOURCE_RECIPE",
        "outer_fold": outer,
        "inner_fold": inner,
        "include_expansion": expanded,
        "ordered_tokens": list(tokens),
        "training_view_sha256": view.identity,
        "cache_sha256": catalog.identity,
        "train_sequences": sorted(allowed),
        "excluded_sequences": list(excluded),
        "allowed_train_groups": sorted({mapping[sequence] for sequence in allowed}),
        "excluded_groups": sorted({mapping[sequence] for sequence in excluded}),
        "inner_prediction_tokens": original.loc[
            original.sequence_id.isin(dev), "sample_token"
        ].tolist(),
        "inner_prediction_sequences": sorted(dev),
        "outer_excluded_sequences": sorted(outer_dev),
        "parent_source_sha256": parent["artifact_sha256"],
        "normalization_fitted": False,
        "prior_fitted": False,
        "optimizer_updates": 0,
        "materialization_requires_stage75_headroom_authorization": True,
        "pretext_source_rule": (
            "all texture and normalization ancestors exclude outer AND inner groups"
        ),
    }
