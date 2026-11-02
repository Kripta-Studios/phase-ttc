"""Protect external provenance and keep executed historical contracts immutable."""

import json
from dataclasses import asdict, replace

import pytest

from e_jepa_ttc.detection.pretrained_campaign import PretrainedConfig, verify_pretrained_artifacts
from e_jepa_ttc.detection.pretrained_initialization import POLICY
from e_jepa_ttc.detection.range_campaign import RangeConfig, digest
from tests.unit.test_detection_range_campaign import config


def external_config(tmp_path):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"not loaded during provenance checks")
    receipt = tmp_path / "conversion.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "phase_monodgp_pretrained_conversion_v1",
                "status": "PASS",
                "repo": "FanqiPu/MonoDGP",
                "revision": "de9f0ad875450f41f9d74edb4fabff8dc911960b",
                "model_state_sha256": digest(weights),
            }
        )
    )
    return PretrainedConfig(
        **(
            config().__dict__
            | {
                "schema": "phase_eap_pretrained_campaign_v1",
                "depth_max_m": 300.0,
                "initialization": POLICY,
            }
        ),
        pretrained_weights=str(weights),
        pretrained_weights_sha256=digest(weights),
        conversion_receipt=str(receipt),
        conversion_receipt_sha256=digest(receipt),
    )


def test_external_config_stays_separate_and_rejects_wrong_policy(tmp_path):
    cfg = external_config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(cfg)))
    assert PretrainedConfig.load(path) == cfg
    with pytest.raises(ValueError):
        RangeConfig.load(path)
    for invalid in (
        replace(cfg, depth_max_m=400),
        replace(cfg, initialization="random"),
        replace(cfg, pretrained_weights_sha256="z" * 64),
    ):
        with pytest.raises(ValueError):
            invalid.validate()


def test_external_bytes_and_receipt_must_agree_before_model_loading(tmp_path):
    from pathlib import Path

    cfg = external_config(tmp_path)
    assert verify_pretrained_artifacts(cfg)["status"] == "PASS"
    Path(cfg.pretrained_weights).write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance"):
        verify_pretrained_artifacts(cfg)
    with pytest.raises(ValueError, match="receipt changed"):
        verify_pretrained_artifacts(replace(cfg, conversion_receipt_sha256="0" * 64))
