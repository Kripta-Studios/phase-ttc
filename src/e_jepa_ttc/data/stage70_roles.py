"""Prospective TRAIN roles and label-free, bounded expansion sampling."""

from __future__ import annotations

import hashlib
import stat
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


def stable_rank(prefix: str, identity: str) -> bytes:
    """Rank an identity without targets or scientific model outputs."""
    return hashlib.sha256(f"{prefix}|{identity}".encode()).digest()


@dataclass(frozen=True)
class TrainRoles:
    """Fixed acquisition roles; sequence fallback must be explicitly documented."""

    sequence_to_group: tuple[tuple[str, str], ...]
    original: tuple[str, ...]
    historical_exposed: tuple[str, ...]
    protected: tuple[str, ...]
    confirmation: tuple[str, ...]
    expansion: tuple[str, ...]
    role_prefix: str

    def __post_init__(self) -> None:
        mapping = dict(self.sequence_to_group)
        if len(mapping) != len(self.sequence_to_group) or not self.role_prefix:
            raise ValueError("duplicate sequence mapping or missing role prefix")
        original, expansion, confirmation = map(
            set, (self.original, self.expansion, self.confirmation)
        )
        if (original & expansion) or (original & confirmation) or (expansion & confirmation):
            raise ValueError("TRAIN roles overlap")
        if (expansion | confirmation) & (set(self.historical_exposed) | set(self.protected)):
            raise ValueError("fresh role contains a documented exposed acquisition")
        if original & set(self.protected):
            raise ValueError("original role overlaps protected acquisition")
        if not original or not (original | expansion | confirmation) <= set(mapping.values()):
            raise ValueError("role refers to an unresolved acquisition")

    def allowed_sequences(self, *, include_expansion: bool) -> frozenset[str]:
        """Return model-development sources; locked confirmation is always excluded."""
        groups = set(self.original)
        if include_expansion:
            groups.update(self.expansion)
        return frozenset(seq for seq, group in self.sequence_to_group if group in groups)

    def check_fit_sequences(
        self,
        sequences: Sequence[str],
        *,
        excluded_sequences: Sequence[str],
        include_expansion: bool,
    ) -> None:
        """Check acquisition exclusions, including aliases of outer/inner sequences."""
        mapping = dict(self.sequence_to_group)
        requested, excluded = set(sequences), set(excluded_sequences)
        if not requested or not requested <= self.allowed_sequences(
            include_expansion=include_expansion
        ):
            raise ValueError("fit source is outside its fixed TRAIN role")
        if not excluded <= mapping.keys():
            raise ValueError("unresolved excluded sequence")
        held_groups = {mapping[seq] for seq in excluded}
        if held_groups & {mapping[seq] for seq in requested}:
            raise ValueError("held-out acquisition exposed through a sequence alias")


def assign_train_roles(
    sequence_to_group: Mapping[str, str],
    *,
    original_sequences: Sequence[str],
    historically_exposed_sequences: Sequence[str],
    protected_sequences: Sequence[str],
    role_prefix: str,
) -> TrainRoles:
    """Apply the fixed six-group holdout rule to an audited public-TRAIN inventory.

    The caller supplies evidence of historical exposure and group membership.
    This function does not infer freshness from an absent in-memory read record.
    """
    mapping = dict(sequence_to_group)
    if (
        not mapping
        or not role_prefix
        or any(not seq or not group for seq, group in mapping.items())
    ):
        raise ValueError("explicit nonempty TRAIN identities and role prefix required")
    for values in (original_sequences, historically_exposed_sequences):
        if not set(values) <= mapping.keys():
            raise ValueError("original/exposed identity is outside public TRAIN inventory")
    original = {mapping[seq] for seq in original_sequences}
    exposed = {mapping[seq] for seq in historically_exposed_sequences}
    protected = {mapping[seq] for seq in protected_sequences if seq in mapping}
    if not original or original & protected:
        raise ValueError("original development acquisition overlaps protected source")
    eligible = sorted(
        set(mapping.values()) - original - exposed - protected,
        key=lambda group: (stable_rank(role_prefix, group), group),
    )
    confirmation = eligible[:6] if len(eligible) >= 12 else []
    expansion = eligible[len(confirmation) :]
    return TrainRoles(
        tuple(sorted(mapping.items())),
        tuple(sorted(original)),
        tuple(sorted(exposed)),
        tuple(sorted(protected)),
        tuple(confirmation),
        tuple(expansion),
        role_prefix,
    )


@dataclass(frozen=True)
class SamplingRow:
    """Metadata-only expansion candidate; there is deliberately no target field."""

    token: str
    sequence: str
    track: str
    anchor_us: int


def sample_expansion(
    rows: Sequence[SamplingRow],
    *,
    sequence_to_group: Mapping[str, str],
    limit: int,
    prefix: str,
    spacing_us: int = 20_000,
) -> tuple[list[str], dict[str, str]]:
    """Round robin groups, sequences and tracks, with stable target-free ranking.

    Spacing rejection is permanent for this recipe; rejected identities and their
    reasons are returned. The caller retains all old outer-train tokens separately.
    """
    if limit < 0 or spacing_us < 20_000 or not prefix:
        raise ValueError("invalid expansion budget/spacing/prefix")
    if len({row.token for row in rows}) != len(rows):
        raise ValueError("duplicate expansion token")
    track_rows: dict[tuple[str, str], list[SamplingRow]] = {}
    for row in rows:
        if row.sequence not in sequence_to_group:
            raise ValueError("unresolved acquisition identity")
        if isinstance(row.anchor_us, bool) or not isinstance(row.anchor_us, int):
            raise ValueError("sampling anchors must be integer microseconds")
        track_rows.setdefault((row.sequence, row.track), []).append(row)
    rejected: dict[str, str] = {}
    pools: dict[tuple[str, str], deque[SamplingRow]] = {}
    for key, candidates in sorted(track_rows.items()):
        chosen: list[SamplingRow] = []
        for row in sorted(candidates, key=lambda row: (stable_rank(prefix, row.token), row.token)):
            if any(abs(row.anchor_us - previous.anchor_us) < spacing_us for previous in chosen):
                rejected[row.token] = "anchor_spacing_under_20ms"
            else:
                chosen.append(row)
        pools[key] = deque(chosen)
    tracks: dict[str, deque[tuple[str, str]]] = {}
    for key in sorted(pools, key=lambda key: (stable_rank(prefix, "|".join(key)), key)):
        if pools[key]:
            tracks.setdefault(key[0], deque()).append(key)
    sequences: dict[str, deque[str]] = {}
    for seq in sorted(tracks, key=lambda seq: (stable_rank(prefix, seq), seq)):
        sequences.setdefault(sequence_to_group[seq], deque()).append(seq)
    groups = deque(sorted(sequences, key=lambda group: (stable_rank(prefix, group), group)))
    result: list[str] = []
    while groups and len(result) < limit:
        group = groups.popleft()
        seq = sequences[group].popleft()
        track = tracks[seq].popleft()
        result.append(pools[track].popleft().token)
        if pools[track]:
            tracks[seq].append(track)
        if tracks[seq]:
            sequences[group].append(seq)
        if sequences[group]:
            groups.append(group)
    for pool in pools.values():
        rejected.update((row.token, "registered_row_cap") for row in pool)
    return result, rejected


def resolve_train_event(
    raw_train_root: Path, sequence: str, metadata_path: str, allowed_sequences: frozenset[str]
) -> Path:
    """Resolve only an allowed TRAIN event file; reject traversal and role aliases."""
    if sequence not in allowed_sequences or any(char in sequence for char in "/\\:"):
        raise ValueError("event sequence is not allowed for this producer")
    text = metadata_path.replace("\\", "/")
    if text not in (
        f"{sequence}/events.h5",
        f"train/{sequence}/events.h5",
        f"data/train/{sequence}/events.h5",
    ):
        raise ValueError("event metadata path does not name the exact TRAIN source")
    root = raw_train_root.resolve(strict=True)
    lexical = root / sequence / "events.h5"
    for component in (lexical.parent, lexical):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError("indirect event path may alias a different source role")
    path = lexical.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("event file escapes the canonical TRAIN root")
    return path
