"""Target-free development selection and acquisition-grouped capacity partitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.data.event_context128_reader import CountObservation, CountSource
from e_jepa_ttc.data.stage70_roles import SamplingRow, sample_expansion, stable_rank


@dataclass(frozen=True)
class CohortRecipe:
    """Prospective common sampling;27 is a ceiling, never a prerequisite."""

    requested_sequences: int = 27
    sample_limit: int = 8192
    spacing_us: int = 20_000
    prefix: str = "stage70_76_capacity_v1"

    def __post_init__(self) -> None:
        if (
            self.requested_sequences < 3
            or self.sample_limit < self.requested_sequences
            or self.spacing_us < 20_000
            or not self.prefix
        ):
            raise ValueError("invalid prospective cohort recipe")


def grouped_folds(
    sequences: Sequence[str], mapping: Mapping[str, str], *, prefix: str
) -> dict[str, int]:
    """Balance sequence counts across three folds without splitting an acquisition."""
    if len(set(sequences)) != len(sequences) or not set(sequences) <= mapping.keys():
        raise ValueError("duplicate or unresolved sequence")
    groups: dict[str, list[str]] = {}
    for sequence in sequences:
        groups.setdefault(mapping[sequence], []).append(sequence)
    if len(groups) < 3:
        raise ValueError("three folds require at least three distinct acquisition groups")
    sizes = [0, 0, 0]
    assignment = {}
    for group in sorted(groups, key=lambda g: (-len(groups[g]), stable_rank(prefix, g), g)):
        fold = min(range(3), key=lambda i: (sizes[i], i))
        assignment.update((sequence, fold) for sequence in groups[group])
        sizes[fold] += len(groups[group])
    return assignment


def build_cohort(
    rows: Sequence[SamplingRow],
    *,
    sequence_to_group: Mapping[str, str],
    confirmation_sequences: Sequence[str],
    protected_sequences: Sequence[str],
    recipe: CohortRecipe,
) -> dict[str, Any]:
    """Select by availability and stable identities, never labels or model outcomes.

    Related sequences form indivisible units. Confirmation identities are inherited,
    not reassigned. The caller must bind the inventory and acquisition-audit evidence.
    """
    mapping = dict(sequence_to_group)
    reserved = set(confirmation_sequences) | set(protected_sequences)
    if not reserved <= mapping.keys() or any(row.sequence not in mapping for row in rows):
        raise ValueError("source roles contain unresolved acquisitions")
    forbidden = {mapping[s] for s in reserved}
    available = {row.sequence for row in rows if mapping[row.sequence] not in forbidden}
    # Do not admit a partially available related acquisition as independent evidence.
    related: dict[str, set[str]] = {}
    for sequence, group in mapping.items():
        if group not in forbidden:
            related.setdefault(group, set()).add(sequence)
    units = {g: s for g, s in related.items() if s <= available}
    selected: list[str] = []
    for group in sorted(units, key=lambda g: (stable_rank(recipe.prefix, g), g)):
        if len(selected) + len(units[group]) <= recipe.requested_sequences:
            selected.extend(sorted(units[group]))
    outer = grouped_folds(selected, mapping, prefix=recipe.prefix + "/outer")
    selected_set = set(selected)
    tokens, rejected = sample_expansion(
        [row for row in rows if row.sequence in selected_set],
        sequence_to_group=mapping,
        limit=recipe.sample_limit,
        prefix=recipe.prefix + "/common_tokens",
        spacing_us=recipe.spacing_us,
    )
    by_token = {row.token: row for row in rows}
    if len(by_token) != len(rows) or {by_token[t].sequence for t in tokens} != selected_set:
        raise ValueError("duplicate token or common sample pool omitted a selected sequence")
    folds = []
    for fold in range(3):
        held = sorted(s for s in selected if outer[s] == fold)
        train = sorted(selected_set - set(held))
        inner = (
            grouped_folds(train, mapping, prefix=f"{recipe.prefix}/outer{fold}/inner")
            if len({mapping[s] for s in train}) >= 3
            else None
        )
        folds.append(
            {
                "outer_fold": fold,
                "train_sequences": train,
                "held_sequences": held,
                "train_tokens": [t for t in tokens if by_token[t].sequence in train],
                "held_tokens": [t for t in tokens if by_token[t].sequence in held],
                "inner_sequence_to_fold": inner,
                "nested_prerequisite": "AVAILABLE" if inner is not None else "TOO_FEW_TRAIN_GROUPS",
            }
        )
    external = sorted(set().union(*units.values()) - selected_set)
    result = {
        "kind": "CAPACITY_DEVELOPMENT_COHORT",
        "recipe": asdict(recipe),
        "selected_sequences": sorted(selected),
        "actual_sequence_count": len(selected),
        "external_sequences": external,
        "confirmation_sequences": sorted(confirmation_sequences),
        "protected_sequences": sorted(protected_sequences),
        "sequence_to_group": dict(sorted(mapping.items())),
        "sequence_to_outer_fold": outer,
        "ordered_tokens": tokens,
        "rejected_tokens": rejected,
        "folds": folds,
        "targets_or_predictions_used_for_selection": False,
        "historical_nine_sequence_results_modified": False,
    }
    return {**result, "artifact_sha256": object_digest(result)}


class RepartitionedCountReader:
    """Bind new token/fold metadata while reusing verified immutable count payloads."""

    def __init__(self, source: CountSource, cohort: dict[str, Any]) -> None:
        unsigned = {k: v for k, v in cohort.items() if k != "artifact_sha256"}
        if object_digest(unsigned) != cohort.get("artifact_sha256"):
            raise ValueError("capacity cohort signature changed")
        tokens = cohort["ordered_tokens"]
        indexed = source.metadata.set_index("sample_token", drop=False)
        if indexed.index.duplicated().any() or not set(tokens) <= set(indexed.index):
            raise ValueError("cohort tokens are absent or duplicated in verified count sources")
        self.metadata = indexed.loc[tokens].reset_index(drop=True).copy()
        folds = self.metadata.sequence_id.map(cohort["sequence_to_outer_fold"])
        if folds.isna().any() or not folds.isin([0, 1, 2]).all():
            raise ValueError("cohort contains an unresolved development fold")
        self.metadata["outer_fold"] = folds.astype(int)
        self.source = source
        self._tokens = frozenset(tokens)
        self.identity = object_digest(
            {
                "operation": "new_grouped_folds_immutable_count_payloads",
                "source_cache_sha256": source.identity,
                "cohort_sha256": cohort["artifact_sha256"],
                "metadata": self.metadata.to_dict(orient="records"),
            }
        )

    metadata: pd.DataFrame

    def observation(self, token: str) -> CountObservation:
        """Reject rows outside the new frozen cohort before accessing their bytes."""
        if token not in self._tokens:
            raise ValueError("token outside the capacity development cohort")
        return self.source.observation(token)
