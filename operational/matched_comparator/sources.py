"""Pinned external inputs of the matched-comparator study, resolved from environment variables.

Locations already defined by ``operational.phase_geometry.data`` are reused
(``PHASE_TTC_POD_RETURN_ROOT``, ``PHASE_TTC_GARLTTC_DATASET``, ``PHASE_TTC_GARLTTC_BENCHMARK``).
Two new ones are added the same way (no defaults; a missing variable stops with its name):

- ``PHASE_TTC_V12_ARTIFACTS``: the ``artifacts`` directory of the sibling worktree
  ``e-jepa-ttc-v12-efficient-context`` that holds the released Garl checkpoints and the frozen
  event-only Garl predictions (TRAIN40 and Test12).
- ``PHASE_TTC_EAP_ROOT``: the public eAP media root holding ``data/train`` and ``data/test``
  (``events.h5`` and ``rgb_shards``), used only to run the released Garl-TTC model.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

ENV_V12_ARTIFACTS = "PHASE_TTC_V12_ARTIFACTS"
ENV_EAP_ROOT = "PHASE_TTC_EAP_ROOT"
ENV_GARLTTC_BENCHMARK = "PHASE_TTC_GARLTTC_BENCHMARK"
ENV_GARLTTC_DATASET = "PHASE_TTC_GARLTTC_DATASET"
LOCATIONS: dict[str, str] = {
    ENV_V12_ARTIFACTS: (
        "the artifacts directory of the e-jepa-ttc-v12-efficient-context worktree "
        "(train40_system_20261005/, garl_checkpoint_comparison_20261010/, sota_campaign_20261008/)"
    ),
    ENV_EAP_ROOT: "the public eAP media root holding data/train and data/test",
    ENV_GARLTTC_BENCHMARK: "the Garl-TTC release repository root (garl_ttc/, configs/)",
    ENV_GARLTTC_DATASET: "the GarlTTC dataset root holding data/train.parquet",
}
PINS: dict[str, dict[str, str]] = {
    "garl_event_train40_predictions": {
        "location": ENV_V12_ARTIFACTS,
        "path": "train40_system_20261005/PUBLIC_GARL_TRAIN_PREDICTIONS.npz",
        "sha256": "3f237b8508f9cc88133afa1ca9cc82397f2233f6d00d4d8f9c73fed01682cf24",
    },
    "train40_rows": {
        "location": ENV_V12_ARTIFACTS,
        "path": "train40_system_20261005/TRAIN40_ROWS.parquet",
        "sha256": "0e59ce2629f512f68dff6ab77beef68090816adac7adf21994cb29fa6954859c",
    },
    "garl_event_test12_predictions": {
        "location": ENV_V12_ARTIFACTS,
        "path": "garl_checkpoint_comparison_20261010/test12_event/TEST_PREDICTIONS.csv",
        "sha256": "a010db07319d68406f6808819a8ac2b4b76a00293328ffc4abfad3e79c00ea0d",
    },
    "garl_event_test12_zip": {
        "location": ENV_GARLTTC_DATASET,
        "path": "Garl_event_only_submission.zip",
        "sha256": "da87e60b0ffd2849555467efb05066932896e4cf87c14e628d87d8e2c1ba9171",
    },
    "garl_event_checkpoint": {
        "location": ENV_V12_ARTIFACTS,
        "path": "train40_system_20261005/public_garl/paper_event_only_lhr.pth",
        "sha256": "fcaf9be47e2dafc6f73c6c3ebd102595ae06119dcae78aea698a42627b2b4fef",
    },
    "garl_event_config": {
        "location": ENV_V12_ARTIFACTS,
        "path": "train40_system_20261005/public_garl/configs/ablation/event_lhr.yaml",
        "sha256": "601ecb74f1ed2f04921f8014ae648aeb8d111ca311e456be20af2d60b63be676",
    },
    "garl_full_checkpoint": {
        "location": ENV_V12_ARTIFACTS,
        "path": "sota_campaign_20261008/dev32_expanded_rgb/public_garl/paper_ours_full.pth",
        "sha256": "e96a613a4fb877a1969d57ab562cadba89961fb202f5f2f2f0658f333a0d443e",
    },
    "garl_full_config": {
        "location": ENV_GARLTTC_BENCHMARK,
        "path": "configs/garl_ttc_eventdecoder.yaml",
        "sha256": "ed7e062f0e86166be17273ac6952bb4916bb29b164bdeb8e218627deb10d2842",
    },
    "garl_ttc_network": {
        "location": ENV_GARLTTC_BENCHMARK,
        "path": "garl_ttc/models/ttc_network.py",
        "sha256": "7927946d4b9af608f1ed2e57c6cc8b498d3c4be8b9c84305647199ef812d9c14",
    },
    "garl_ttc_dataset": {
        "location": ENV_GARLTTC_BENCHMARK,
        "path": "garl_ttc/datasets/ttc_dataset.py",
        "sha256": "d21eaf3928528c2d5b9a7c2787a64e248bbf49bc60756a3156034b9d8ac5b592",
    },
    "garl_ttc_inference": {
        "location": ENV_GARLTTC_BENCHMARK,
        "path": "garl_ttc/engine/inference.py",
        "sha256": "78f30f84bb771c278e2b692412dd07beee8b4f12fd500be3b8e9e9003482dfa8",
    },
    "garl_train_inputs": {
        "location": ENV_GARLTTC_DATASET,
        "path": "data/train.parquet",
        "sha256": "03dd3022db4b5f43bb10244fc8778476d74351e764f73a90c8566af949c17fd6",
    },
}


class MissingLocationError(RuntimeError):
    """An external input location is not configured."""


def sha256_file(path: Path) -> str:
    """SHA-256 of a regular file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def location_root(variable: str) -> Path:
    """Directory named by ``variable``; fail with an actionable message if unset or missing."""
    value = os.environ.get(variable, "").strip()
    if not value:
        raise MissingLocationError(
            f"Environment variable {variable} is not set. Set it to {LOCATIONS[variable]} "
            "(see docs/sota_20261118/matched_comparator/REPORT.md, Reproducción)."
        )
    path = Path(value)
    if not path.is_dir():
        raise MissingLocationError(f"{variable}={value} is not an existing directory")
    return path


def pin_path(name: str) -> Path:
    """Resolve a pinned file against its configured location (no hash check)."""
    spec = PINS[name]
    return location_root(spec["location"]) / spec["path"]


def verified(name: str) -> Path:
    """Resolve a pinned file and check its SHA-256."""
    path = pin_path(name)
    digest = sha256_file(path)
    if digest != PINS[name]["sha256"]:
        raise ValueError(f"{name} differs from its pinned SHA-256: {path} ({digest})")
    return path


__all__ = [
    "ENV_EAP_ROOT",
    "ENV_V12_ARTIFACTS",
    "LOCATIONS",
    "PINS",
    "MissingLocationError",
    "location_root",
    "pin_path",
    "sha256_file",
    "verified",
]
