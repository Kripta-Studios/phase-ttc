"""Protect denominator, parameter accounting and native-output comparison."""

import numpy as np
import pytest
import torch

from operational.local_comparison.core import (
    assert_parity,
    parameter_inventory,
    summarize_records,
    without_decoders,
)


def test_failed_queries_stay_in_denominator_and_matched_population():
    rows = [
        dict(
            model=m,
            query=q,
            repeat=0,
            status=status,
            inference_ms=cost,
            preparation_ms=2.0,
            total_ms=cost + 2,
        )
        for m, q, status, cost in [
            ("phase", "a", "ok", 10.0),
            ("phase", "b", "ok", 30.0),
            ("garl", "a", "ok", 20.0),
            ("garl", "b", "unavailable", 0.0),
        ]
    ]
    result = summarize_records(rows)
    assert result["phase"]["attempts"] == 2
    assert result["garl"]["coverage"] == 0.5
    assert result["phase"]["matched_success"]["inference_ms"]["mean"] == 10.0
    assert result["phase"]["successful"]["inference_ms"]["mean"] == 20.0


def test_duplicate_trials_and_bad_timings_are_rejected():
    row = dict(
        model="m", query="a", repeat=0, status="ok", inference_ms=1, preparation_ms=0, total_ms=1
    )
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_records([row, row])
    with pytest.raises(ValueError, match="Invalid"):
        summarize_records([{**row, "inference_ms": float("nan")}])


def test_inventory_distinguishes_unused_weights_and_deduplicates():
    model = torch.nn.ModuleDict({"active": torch.nn.Linear(3, 2), "unused": torch.nn.Linear(8, 4)})
    inventory = parameter_inventory({"m": model}, lambda: model["active"](torch.ones(1, 3)))
    assert inventory["total_parameters"] == 44
    assert inventory["executed_parameters"] == 8
    assert inventory["parameter_bytes"] == 44 * 4


def test_decoder_switch_restores_after_exception():
    model = torch.nn.Linear(2, 2)
    model.with_decoder = True
    with pytest.raises(RuntimeError), without_decoders(model):
        assert not model.with_decoder
        raise RuntimeError()
    assert model.with_decoder


def test_parity_requires_matching_availability_and_finite_values():
    assert_parity(1.0, 1.0)
    assert_parity(None, None)
    for lhs, rhs in [(None, 1.0), (np.nan, np.nan), (1.0, 1.1)]:
        with pytest.raises(ValueError, match="parity"):
            assert_parity(lhs, rhs)
