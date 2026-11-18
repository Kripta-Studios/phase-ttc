"""Saturation-scale arms, design matrix, fixed expert and FCWD stratum statistics."""

from __future__ import annotations

import numpy as np
import pytest

from operational.geometry_robust import robust
from operational.phase_geometry import geometry
from operational.saturation_scale import content_scale as cs
from operational.saturation_scale import variants
from operational.saturation_scale.fcwd_eval import stratum_delta_ci


def test_arm_columns_respect_their_declared_information() -> None:
    for name, arm in variants.ARMS.items():
        assert len(set(arm.columns)) == len(arm.columns)
        assert max(arm.columns) < 56 + len(cs.C_FIELDS)
        assert not set(arm.columns) & set(range(10, 20))  # no PHASE event fields anywhere
        assert not set(arm.columns) & set(range(43, 56))  # no 2 s memory block
        if name.startswith("GEO_"):
            assert min(arm.columns) >= 20 and not arm.use_width
        assert (set(variants.C_COLS) <= set(arm.columns)) == name.endswith("_CS")
    primary, base = variants.ARMS[variants.PRIMARY], variants.ARMS[variants.BASELINE]
    assert set(primary.columns) - set(base.columns) == set(variants.C_COLS)
    assert primary.anchor == base.anchor and primary.augmented and base.augmented


def test_full_design_appends_content_inputs() -> None:
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "as_tensor"):
        pytest.skip("the released PHASE20 scaling needs a working torch")
    n = 4
    g = np.zeros((n, len(geometry.RAW_FIELDS)))
    r = np.zeros((n, len(robust.RAW_R_FIELDS)))
    c = np.zeros((n, len(cs.RAW_C_FIELDS)))
    c[2, cs.RAW_C_FIELDS.index("valid")] = 1.0
    x, anchors, widths = variants.full_design(
        np.zeros((n, 20), np.float32), np.zeros((n, 2), np.float32), g, r, c
    )
    assert x.shape == (n, 56 + len(cs.C_FIELDS)) and anchors.shape == (n, 4)
    assert x[2, variants.C_COLS[cs.C_FIELDS.index("valid")]] == 1.0
    assert widths.shape == (n, 2)


def test_content_fixed_falls_back_to_robust_phase() -> None:
    r = np.zeros((2, len(robust.RAW_R_FIELDS)))
    r[:, robust.RAW_R_FIELDS.index("phi_robust")] = (0.01, 0.02)
    c = np.zeros((2, len(cs.RAW_C_FIELDS)))
    c[0, cs.RAW_C_FIELDS.index("valid")] = 1.0
    c[0, cs.RAW_C_FIELDS.index("phi_content")] = 0.07
    c[1, cs.RAW_C_FIELDS.index("phi_content")] = 0.09  # invalid: ignored
    assert np.allclose(variants.content_fixed_phase(r, c), (0.07, 0.02))


def test_stratum_delta_ci_pools_rows_and_brackets_point() -> None:
    order = ["A", "B", "C"]
    sequences = np.asarray(["A", "A", "B", "C", "C", "C"])
    diff = np.asarray([1.0, 3.0, -2.0, 0.0, 0.0, 6.0])
    mask = np.asarray([True, True, True, False, True, True])
    draws = np.random.default_rng(0).integers(0, 3, (2000, 3))
    out = stratum_delta_ci(diff, mask, sequences, order, draws)
    assert out["delta"] == pytest.approx(8.0 / 5.0) and out["rows"] == 5
    assert out["ci95"][0] <= out["delta"] <= out["ci95"][1]
