"""Feature-assembly parity between eAP training rows and FCWD queries (synthetic inputs)."""

from __future__ import annotations

import numpy as np
import pytest

from operational.event_geometry import history_inputs as hi
from operational.event_geometry.evaluate_eap import clipped_edges, select_primary
from operational.event_geometry.fcwd_eval import edge_count_fcwd, verdict
from operational.event_geometry.variants import (
    ANCHOR_COLUMN,
    VARIANTS,
    design,
    scale_event10,
    stack_columns,
    stacked_anchors,
)
from operational.phase_geometry import geometry
from operational.phase_geometry.variants import scaled_phase20
from operational.second_benchmark import fcwd


def _synthetic(n: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(0.0, 0.03, (n, 20)).astype(np.float32)
    features[:, 15:19] = rng.uniform(0.0, 0.02, (n, 4))
    features[:, 19] = 0.875
    raw = np.zeros((n, len(geometry.RAW_FIELDS)))
    for i in range(n):
        times = np.arange(7) * 100_000
        scale = 1.0 + rng.uniform(0.0, 0.3) * np.arange(7) / 6
        boxes = [[600 - 50 * s, 300 - 30 * s, 600 + 50 * s, 300 + 30 * s] for s in scale]
        values = geometry.window_features(times, boxes, int(times[-1]))
        raw[i] = [values[k] for k in geometry.RAW_FIELDS]
    return features, raw, rng.normal(0.0, 1.0, (n, 2)).astype(np.float32)


def test_event_scaling_matches_released_phase20_scaling() -> None:
    features, _, _ = _synthetic()
    np.testing.assert_array_equal(
        scale_event10(features[:, 10:20]), scaled_phase20(features)[:, 10:20]
    )


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_stacked_training_columns_equal_shared_design(name: str) -> None:
    features, raw, width = _synthetic()
    variant = VARIANTS[name]
    x_full = np.concatenate((scaled_phase20(features), geometry.learned_inputs(raw)), 1)
    x, anchor, w = design(variant, features[:, 10:20], raw, features[:, :10], width)
    np.testing.assert_array_equal(x, x_full[:, stack_columns(variant)])
    np.testing.assert_array_equal(
        anchor, stacked_anchors(features, raw)[:, ANCHOR_COLUMN[variant.anchor]]
    )
    assert x.shape[1] == variant.inputs
    assert (w == width).all() if variant.uses_rgb_width else not w.any()


@pytest.mark.parametrize("name", ["EVT_GEO_BAND", "EVT_GEO_BAND_GA"])
def test_fcwd_block_assembly_equals_training_fields(name: str) -> None:
    """FCWD reads fields from the frozen 170-D E_CTX state; training reads PHASE20 10-19."""
    features, raw, _ = _synthetic()
    block = np.random.default_rng(1).normal(size=(len(features), 170)).astype(np.float32)
    block[:, 160:170] = features[:, 10:20]
    train = design(VARIANTS[name], features[:, 10:20], raw)
    deploy = design(VARIANTS[name], hi.event10_from_block(block), raw)
    for a, b in zip(train, deploy, strict=True):
        np.testing.assert_array_equal(a, b)


def test_fcwd_variants_ignore_timing_and_support_fields() -> None:
    features, raw, _ = _synthetic()
    shifted = features.copy()
    shifted[:, 15:20] = 0.0
    for name in ("EVT_GEO_BAND", "EVT_GEO_BAND_GA"):
        a = design(VARIANTS[name], features[:, 10:20], raw)
        b = design(VARIANTS[name], shifted[:, 10:20], raw)
        for left, right in zip(a, b, strict=True):
            np.testing.assert_array_equal(left, right)


def test_rgb_variants_require_rgb_inputs() -> None:
    features, raw, _ = _synthetic()
    with pytest.raises(ValueError):
        design(VARIANTS["REL_E5_GEO_BAND"], features[:, 10:20], raw)


def test_fcwd_edge_count_convention() -> None:
    boxes = np.asarray([[1, 1, 1281, 721], [10, 1, 500, 400], [100, 100, 200, 200]], float)
    assert edge_count_fcwd(boxes).tolist() == [4, 1, 0]
    tx, ty = fcwd.truncation_fcwd(boxes)
    assert bool(tx[0] and ty[0]) and not tx[1] and ty[1]


def test_eap_clipped_edges_are_counted_not_ored() -> None:
    assert clipped_edges(np.asarray([0, 0, 1279, 719])) == 4
    assert clipped_edges(np.asarray([0, 10, 500, 719])) == 2
    assert clipped_edges(np.asarray([5, 10, 500, 700])) == 0


def test_selection_rule_rejects_undefined_tie_break() -> None:
    table = {"EVT_GEO_BAND": {"overall_mean": 40.0}, "EVT_GEO_BAND_GA": {"overall_mean": 40.5}}
    with pytest.raises(ValueError):
        select_primary(table, {"EVT_GEO_BAND": float("nan"), "EVT_GEO_BAND_GA": float("nan")})


def test_selection_rule_breaks_ties_on_clipped_edges() -> None:
    table = {"EVT_GEO_BAND": {"overall_mean": 40.0}, "EVT_GEO_BAND_GA": {"overall_mean": 40.8}}
    tie = select_primary(table, {"EVT_GEO_BAND": 60.0, "EVT_GEO_BAND_GA": 55.0})
    assert tie["primary"] == "EVT_GEO_BAND_GA"
    table["EVT_GEO_BAND_GA"]["overall_mean"] = 42.0
    clear = select_primary(table, {"EVT_GEO_BAND": 60.0, "EVT_GEO_BAND_GA": 55.0})
    assert clear["primary"] == "EVT_GEO_BAND"


def test_verdict_rule() -> None:
    base = {
        "seed_mean_macro_rte": 25.0,
        "geo_learned_seed_mean_macro_rte": 33.4,
        "per_seed_delta_vs_geo_learned": {"7": -8.0, "13": -9.0, "21": -7.0},
        "seed_mean_delta_vs_garl_local": -1.8,
    }
    assert verdict(base) == "SUPPORTED"
    assert (
        verdict({**base, "seed_mean_macro_rte": 29.0, "seed_mean_delta_vs_garl_local": 2.2})
        == "PARTIAL"
    )
    worse = {**base, "per_seed_delta_vs_geo_learned": {"7": -1.0, "13": 0.5, "21": -2.0}}
    assert verdict(worse) == "NOT_SUPPORTED"
