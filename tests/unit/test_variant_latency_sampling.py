"""Seeded, disjoint, label-free selection of the timed query population."""

from __future__ import annotations

import json

import numpy as np
import pytest

from operational.variant_latency import sampling


def _groups() -> tuple[np.ndarray, np.ndarray]:
    groups = np.repeat([f"S{i:02d}" for i in range(20)], 80)
    eligible = np.ones(len(groups), dtype=bool)
    eligible[::7] = False
    return groups, eligible


def test_population_is_deterministic_disjoint_and_eligible() -> None:
    groups, eligible = _groups()
    a = sampling.choose(groups, eligible, n_sequences=5, per_sequence=20, warmup=4)
    b = sampling.choose(groups, eligible, n_sequences=5, per_sequence=20, warmup=4)
    assert a == b
    assert len(a.sequences) == 5 and len(a.measured) == 100 and len(a.warmup) == 4
    rows = [a.first_call, *a.warmup, *a.measured]
    assert len(set(rows)) == len(rows)
    assert all(eligible[r] for r in rows)
    assert set(groups[list(a.measured)]) == set(a.sequences)
    for sequence in a.sequences:
        assert int((groups[list(a.measured)] == sequence).sum()) == 20
    assert {groups[r] for r in (a.first_call, *a.warmup)} <= set(a.sequences)
    assert list(a.measured) != sorted(a.measured)  # shuffled execution order


def test_different_seed_changes_population_and_errors_are_explicit() -> None:
    groups, eligible = _groups()
    a = sampling.choose(groups, eligible, seed=1, n_sequences=5, per_sequence=20)
    b = sampling.choose(groups, eligible, seed=2, n_sequences=5, per_sequence=20)
    assert a.measured != b.measured
    with pytest.raises(ValueError, match="sequences"):
        sampling.choose(groups, eligible, n_sequences=21)
    with pytest.raises(ValueError, match="fewer"):
        sampling.choose(groups, eligible, n_sequences=2, per_sequence=79)
    with pytest.raises(ValueError, match="aligned"):
        sampling.choose(groups, eligible[:-1])


def test_pipeline_orders_are_seeded_permutations() -> None:
    names = ["A", "B:cold", "B:persistent"]
    orders = sampling.pipeline_orders(50, names)
    assert orders == sampling.pipeline_orders(50, names)
    assert all(sorted(o) == sorted(names) for o in orders)
    assert len({tuple(o) for o in orders}) > 1
    population = sampling.choose(*_groups(), n_sequences=2, per_sequence=3, warmup=1)
    decoded = json.loads(json.dumps(population.as_json()))
    assert decoded["measured"] == list(population.measured)
