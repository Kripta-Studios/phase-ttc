"""Seeded selection of the timed query population (pure numpy, label-free)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

SEED = 20261119
N_SEQUENCES = 12
PER_SEQUENCE = 50
WARMUP = 10


@dataclass(frozen=True)
class Population:
    """Row indices: one first call, the warm-up rows and the measured rows (execution order)."""

    sequences: tuple[str, ...]
    first_call: int
    warmup: tuple[int, ...]
    measured: tuple[int, ...]

    def as_json(self) -> dict[str, Any]:
        """JSON-safe description."""
        return {
            "sequences": list(self.sequences),
            "first_call": self.first_call,
            "warmup": list(self.warmup),
            "measured": list(self.measured),
        }


def choose(
    groups: Sequence[str] | NDArray[Any],
    eligible: Sequence[bool] | NDArray[Any],
    *,
    seed: int = SEED,
    n_sequences: int = N_SEQUENCES,
    per_sequence: int = PER_SEQUENCE,
    warmup: int = WARMUP,
) -> Population:
    """Pick ``n_sequences`` sequences, ``per_sequence`` measured rows in each and, from the
    remaining eligible rows of those sequences, one first-call row plus ``warmup`` rows.

    The measured rows are returned in a seeded random execution order.  Only sequence identity
    and the eligibility mask are read.
    """
    names = np.asarray(groups).astype(str)
    mask = np.asarray(eligible, dtype=bool)
    if names.shape != mask.shape or names.ndim != 1:
        raise ValueError("groups and eligibility must be aligned vectors")
    rng = np.random.default_rng(seed)
    candidates = sorted(set(names[mask].tolist()))
    if len(candidates) < n_sequences:
        raise ValueError("Not enough eligible sequences")
    chosen = sorted(rng.choice(candidates, size=n_sequences, replace=False).tolist())
    measured: list[int] = []
    for sequence in chosen:
        rows = np.flatnonzero(mask & (names == sequence))
        if len(rows) < per_sequence:
            raise ValueError(f"{sequence}: fewer than {per_sequence} eligible rows")
        measured.extend(int(v) for v in rng.choice(rows, size=per_sequence, replace=False))
    taken = set(measured)
    pool = [int(v) for v in np.flatnonzero(mask & np.isin(names, chosen)) if int(v) not in taken]
    if len(pool) < warmup + 1:
        raise ValueError("Not enough eligible rows for the first call and warm-up")
    extra = [int(v) for v in rng.choice(pool, size=warmup + 1, replace=False)]
    order = [int(v) for v in rng.permutation(np.asarray(measured, dtype=np.int64))]
    return Population(tuple(chosen), extra[0], tuple(extra[1:]), tuple(order))


def pipeline_orders(count: int, names: Sequence[str], seed: int = SEED + 1) -> list[list[str]]:
    """A seeded random permutation of the pipeline names for each of ``count`` queries."""
    rng = np.random.default_rng(seed)
    base = list(names)
    return [[base[i] for i in rng.permutation(len(base))] for _ in range(count)]


__all__ = [
    "N_SEQUENCES",
    "PER_SEQUENCE",
    "SEED",
    "WARMUP",
    "Population",
    "choose",
    "pipeline_orders",
]
