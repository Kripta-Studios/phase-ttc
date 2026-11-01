"""The new range must not relax the historical producer or change other settings."""

import json
from dataclasses import asdict, replace

import pytest

from e_jepa_ttc.detection.long_range_campaign import LongRangeConfig
from e_jepa_ttc.detection.range_campaign import RangeConfig
from tests.unit.test_detection_range_campaign import config


def test_long_range_has_distinct_contract_and_rejects_other_changes(tmp_path):
    values = asdict(config()) | {"schema": "phase_eap_long_range_campaign_v1", "depth_max_m": 300.0}
    values["model"] = config().model
    cfg = LongRangeConfig(**values)
    cfg.validate()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(cfg)))
    assert LongRangeConfig.load(path) == cfg
    with pytest.raises(ValueError):
        RangeConfig.load(path)
    for invalid in (
        replace(cfg, depth_max_m=400.0),
        replace(cfg, schema="phase_eap_range_campaign_v1"),
        replace(cfg, evaluation_profile_sha256="a" * 64),
        replace(cfg, batch_size=True),
    ):
        with pytest.raises(ValueError):
            invalid.validate()
