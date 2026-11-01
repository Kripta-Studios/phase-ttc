"""Separate, byte-bound KITTI initialization experiment on the fixed 300m recipe."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch

from e_jepa_ttc.detection.pretrained_initialization import POLICY, transfer_kitti_weights
from e_jepa_ttc.detection.range_campaign import RangeConfig, digest


@dataclass(frozen=True)
class PretrainedConfig(RangeConfig):
    """Bind the safely converted external model and its verified intake receipt."""

    pretrained_weights: str
    pretrained_weights_sha256: str
    conversion_receipt: str
    conversion_receipt_sha256: str

    @classmethod
    def load(cls, path: Path) -> PretrainedConfig:
        """Retain strict field parsing, including explicit external provenance."""
        return cast(PretrainedConfig, super().load(path))

    def validate(self) -> None:
        """Accept only the registered initialization and unchanged range/geometry."""
        if self.schema != "phase_eap_pretrained_campaign_v1":
            raise ValueError("Unknown pretrained campaign schema")
        if self.depth_max_m != 300 or self.initialization != POLICY:
            raise ValueError("Pretrained comparison requires its policy and 300m range")
        for name in ("pretrained_weights", "conversion_receipt"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"Missing {name}")
            value = getattr(self, name + "_sha256")
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"Invalid {name} SHA-256")
        RangeConfig.validate(
            replace(
                self,
                schema="phase_eap_range_campaign_v1",
                depth_max_m=150.0,
                initialization="dfull_reset_depth_classifier_keep_physical_embeddings_v1",
            )
        )


def verify_pretrained_artifacts(cfg: PretrainedConfig) -> dict[str, Any]:
    """Check both byte identities and their cross-binding before allocating CUDA."""
    cfg.validate()
    receipt_path = Path(cfg.conversion_receipt)
    weights = Path(cfg.pretrained_weights)
    if digest(receipt_path) != cfg.conversion_receipt_sha256:
        raise ValueError("External conversion receipt changed")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema") != "phase_monodgp_pretrained_conversion_v1"
        or receipt.get("status") != "PASS"
        or receipt.get("repo") != "FanqiPu/MonoDGP"
        or receipt.get("revision") != "de9f0ad875450f41f9d74edb4fabff8dc911960b"
        or receipt.get("model_state_sha256") != cfg.pretrained_weights_sha256
        or digest(weights) != cfg.pretrained_weights_sha256
    ):
        raise ValueError("External weights lack the admitted conversion provenance")
    return receipt


def initialize_pretrained(model: torch.nn.Module, cfg: PretrainedConfig) -> dict[str, Any]:
    """Transfer only model tensors; all eAP training state starts afresh."""
    receipt = verify_pretrained_artifacts(cfg)
    state = torch.load(cfg.pretrained_weights, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Converted external payload must contain only model tensors")
    if (
        len(state) != receipt["tensor_count"]
        or sum(v.numel() for v in state.values()) != receipt["tensor_elements"]
    ):
        raise ValueError("External tensor inventory differs")
    policy = transfer_kitti_weights(model, state, seed=cfg.seed)
    return {
        **policy,
        "pretrained_weights_sha256": cfg.pretrained_weights_sha256,
        "conversion_receipt_sha256": cfg.conversion_receipt_sha256,
        "external_repo": receipt["repo"],
        "external_revision": receipt["revision"],
        "upstream_checkpoint_sha256": receipt["upstream_checkpoint_sha256"],
        "new_optimizer": True,
        "historical_resume": False,
    }
