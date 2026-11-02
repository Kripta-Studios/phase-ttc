from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest


def _module() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "build_r0_monitor_curve.py"
    spec = importlib.util.spec_from_file_location("r0_curve", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load curve script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _monitor(update: int, ap: float, recall: float, xy: float) -> dict[str, Any]:
    return {
        "update": update,
        "protocol_sha256": "protocol",
        "profile": "local",
        "official_eap_score": False,
        "sample_count": 1024,
        "population_identity_sha256": "population",
        "eligible_ground_truth_count": 100,
        "dev_ap101_supported_classes": ap,
        "class_agnostic_recall": {threshold: recall for threshold in ("0.5", "1.0", "2.0", "4.0")},
        "errors": {
            "xy_median_m": xy,
            "ego_z_median_m": 0.1,
            "dimension_lwh_mean_abs_median_m": 0.2,
            "yaw_median_rad": 0.3,
        },
        "per_class": {
            name: {"ap101": ap, "eligible_ground_truth_count": 1}
            for name in cast(Any, _module()).CLASSES
        },
    }


def test_summary_detects_localization_pattern_without_overfit_claim() -> None:
    module = cast(Any, _module())
    monitors = [
        _monitor(1024, 0.04, 0.5, 1.0),
        _monitor(2048, 0.06, 0.6, 0.8),
        _monitor(3072, 0.08, 0.7, 0.7),
        _monitor(4096, 0.03, 0.3, 1.4),
    ]
    result = module.build_summary(monitors)
    assert result["interpretation"]["dominant_observed_pattern"] == (
        "CENTER_LOCALIZATION_DIAGNOSTIC_DEGRADATION"
    )
    assert "does not prove overfitting" in result["interpretation"]["not_proven"][0]
    assert result["curve_integrity"]["best_retained_monitor_update"] == 3072


def test_summary_rejects_population_change() -> None:
    module = cast(Any, _module())
    monitors = [
        _monitor(1024, 0.04, 0.5, 1.0),
        _monitor(2048, 0.06, 0.6, 0.8),
        _monitor(3072, 0.08, 0.7, 0.7),
        _monitor(4096, 0.03, 0.3, 1.4),
    ]
    monitors[-1]["population_identity_sha256"] = "different"
    with pytest.raises(ValueError, match="population_identity_sha256"):
        module.build_summary(monitors)


def test_median_errors_labels_ego_z_and_parses_dimensions() -> None:
    result = cast(Any, _module())._median_errors(
        [
            {
                "errors": {
                    "xy_m": 2.0,
                    "z_m": 0.5,
                    "dimensions_lwh_m": "1.0 2.0 3.0",
                    "yaw_rad": 0.25,
                }
            }
        ]
    )
    assert result["xy_median_m"] == 2.0
    assert result["ego_z_median_m"] == 0.5
    assert result["dimension_lwh_mean_abs_median_m"] == 2.0
