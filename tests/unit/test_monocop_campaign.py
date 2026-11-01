"""Protect the registered population and budget of the separate quality screen."""

from dataclasses import asdict, replace

import pytest

from e_jepa_ttc.detection.monocop_campaign import GATE_SHA256, PANEL_SHA256, MonoCoPScreenConfig


def config() -> MonoCoPScreenConfig:
    return MonoCoPScreenConfig(
        "phase_monocop_screen_v1",
        "fixture",
        "technical",
        "a" * 64,
        "gate",
        GATE_SHA256,
        "panel",
        PANEL_SHA256,
        2048,
        [0, 512, 1024, 1536, 2048],
        128,
        7200.0,
        4.0,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"updates": 4096},
        {"maximum_seconds": 14400},
        {"population_sha256": "a" * 64},
        {"technical_gate_sha256": "b" * 64},
        {"evaluation_updates": [0, 2048]},
        {"checkpoint_interval": True},
    ],
)
def test_screen_rejects_budget_population_or_gate_drift(change: dict) -> None:
    with pytest.raises(ValueError):
        replace(config(), **change).validate()


def test_screen_lineage_changes_with_technical_inputs() -> None:
    original = config()
    original.validate()
    changed = replace(original, technical_config_sha256="b" * 64)
    assert original.sha256 != changed.sha256
    assert asdict(original)["updates"] == 2048
