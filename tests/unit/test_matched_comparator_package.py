"""Fact sheets, arm names and package expansion of the matched comparators (numpy only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from operational.matched_comparator import final_package
from operational.matched_comparator.run_cv import arm_name


def _parameters(inputs: int) -> int:
    """``ResidualPhaseHead(d)``: Linear(d,32)+LN+Linear(32,32)+LN+Linear(32,3)."""
    return (inputs * 32 + 32) + 64 + (32 * 32 + 32) + 64 + (32 * 3 + 3)


def test_arm_names_follow_the_protocol() -> None:
    assert arm_name("FULL", "GEO_R") == "GARL_FULL_GEO_BAND"
    assert arm_name("EVENT", "R") == "GARL_EVENT_BAND"


def test_fact_sheets_state_modalities_explicitly() -> None:
    a = final_package.FACT_SHEETS["A"]
    assert a["uses_event"] is False and a["uses_image"] is False
    assert "supplied regions only" in a["method_name"]
    assert final_package.FACT_SHEETS["B_FULL"]["uses_image"] is True
    assert final_package.FACT_SHEETS["B_EVENT"]["uses_image"] is False
    assert f"{_parameters(10):,}" in a["note"]
    assert f"{_parameters(13):,}" in final_package.FACT_SHEETS["B_FULL"]["note"]


def test_package_expands_aliases_in_official_order(tmp_path: Path) -> None:
    test = {
        "sample_token": np.asarray(["p0", "p1"]),
        "official_token": np.asarray(["t0", "t1", "t2"]),
        "official_canonical": np.asarray(["p1", "p0", "p1"]),
    }
    docs = tmp_path / "docs"
    docs.mkdir()
    path, ttc = final_package.package(
        "A", "X", np.asarray([2.0, -3.0]), test, tmp_path / "out", docs
    )
    np.testing.assert_array_equal(ttc, [-3.0, 2.0, -3.0])
    assert (docs / path.name).read_bytes() == path.read_bytes()
    np.testing.assert_array_equal(final_package.official_vector(path, ["t0", "t1", "t2"]), ttc)
    with pytest.raises(ValueError):
        final_package.official_vector(path, ["t1", "t0", "t2"])
