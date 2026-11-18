"""Label-free projections, 12-sequence resolution and source pins (numpy only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from operational.matched_comparator import projection, sources


def _toy(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    clusters = np.repeat([f"s{i:02d}" for i in range(20)], 40)
    bands = np.tile(np.arange(4), 200)
    mid = rng.gamma(2.0, 20.0, size=800)
    return mid, bands, clusters


def test_multiplicative_identity_returns_the_official_anchor() -> None:
    mid, bands, clusters = _toy()
    result = projection.multiplicative(mid, mid, bands, clusters, "RGB_GEO_BAND", replicates=50)
    assert result["point"] == pytest.approx(23.47, abs=0.01)
    assert result["ci95"][0] == pytest.approx(result["ci95"][1], abs=1e-9)


def test_multiplicative_scales_each_band() -> None:
    mid, bands, clusters = _toy()
    scaled = mid * np.where(bands == 0, 2.0, 1.0)
    result = projection.multiplicative(scaled, mid, bands, clusters, "REL", replicates=50)
    official = projection.OFFICIAL["REL"]
    expected = 0.5 * 2 * official["c"] + 0.3 * official["s"] + 0.1 * official["l"]
    expected += 0.1 * official["n"]
    assert result["point"] == pytest.approx(expected, rel=1e-9)
    assert result["ratio_point"]["c"] == pytest.approx(2.0)


def test_additive_projection_is_anchor_when_predictions_coincide() -> None:
    rng = np.random.default_rng(1)
    truth = rng.choice([-5.0, 2.0, 4.0, 8.0], size=400) * rng.uniform(0.8, 1.2, size=400)
    clusters = np.repeat([f"s{i}" for i in range(10)], 40)
    bands = np.select(
        [(truth > 0) & (truth <= 3), (truth > 3) & (truth <= 6), (truth > 6) & (truth <= 10)],
        [0, 1, 2],
        default=3,
    )
    fold = truth * 1.1
    test_pred = rng.uniform(1, 9, size=50)
    result = projection.additive_disagreement(
        fold, fold, truth, bands, clusters, test_pred, test_pred, np.ones(50), replicates=20
    )
    assert result["point"] == pytest.approx(23.47, abs=0.01)


def test_resolution12_is_zero_for_a_constant_difference() -> None:
    _, bands, clusters = _toy()
    out = projection.resolution12(np.full(800, 3.0), bands, clusters, replicates=200)
    assert out["sd"] == pytest.approx(0.0, abs=1e-12)
    assert out["q025"] == pytest.approx(3.0)


def test_predicted_group_matches_level_gap_grouping() -> None:
    ttc = np.asarray([1.0, 4.0, 7.0, 20.0, -3.0, -30.0])
    assert projection.predicted_group(ttc).tolist() == [0, 1, 2, 2, 3, 3]


def test_missing_location_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(sources.ENV_V12_ARTIFACTS, raising=False)
    with pytest.raises(sources.MissingLocationError, match=sources.ENV_V12_ARTIFACTS):
        sources.pin_path("garl_full_checkpoint")


def test_pins_are_relative_and_hashed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for spec in sources.PINS.values():
        assert not Path(spec["path"]).is_absolute()
        assert len(spec["sha256"]) == 64
        assert spec["location"] in sources.LOCATIONS
    target = tmp_path / sources.PINS["garl_full_config"]["path"]
    target.parent.mkdir(parents=True)
    target.write_text("x", encoding="utf-8")
    monkeypatch.setenv(sources.ENV_GARLTTC_BENCHMARK, str(tmp_path))
    with pytest.raises(ValueError, match="pinned SHA-256"):
        sources.verified("garl_full_config")
