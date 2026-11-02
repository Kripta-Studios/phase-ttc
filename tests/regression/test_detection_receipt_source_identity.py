"""Keep the eight formatter exceptions tied to published detection receipts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
EVIDENCE = REPOSITORY / "docs/detection_sota_20261029/evidence"

# Digests recorded before the CI repair at commit 5a563243, not regenerated
# from working source. A successor needs a new identity and its own receipts.
RECEIPT_BOUND_SOURCES = [
    (
        "scripts/audit_monocop_gpu_smoke.py",
        "d30be3b36f449b0202eec2687f275f3718b61f42bf25bf7f94453c20adb4e287",
        "monocop_gpu_execution_20261101/monocop_gpu_smoke_recovery_v1/GATE.json",
    ),
    (
        "scripts/preflight_detection_rgb_assets.py",
        "38dafa09b69b0de4cdeda921e1462996781fa06d556e00e5ba702a4c85b46f22",
        "current_v2/manifest.json",
    ),
    (
        "scripts/probe_detection_event_representation.py",
        "afd163cd9091e06ce0e968374d7126c3df5f4dc1a3483953643da18c52ab978f",
        "events_v3/probe.json",
    ),
    (
        "scripts/profile_detection_evaluation_memory.py",
        "6c98dc6c501c45044b226f620b322357a8fa67005acdbb1ff718717883fcb676",
        "current_v2/manifest.json",
    ),
    (
        "scripts/revise_detection_campaign_media.py",
        "285bc8b431eb23ced325045e5c3bb9ca1b7af3274df64d85d0baf2fedb6a1ea0",
        "current_v2/manifest.json",
    ),
    (
        "scripts/run_detection_campaign_queue.py",
        "98303ce7d90a22832e3cbc15e755c0be8f31b17f7413921c84511884ccb2dc69",
        "r0_r2_seed7_20261031/comparison.json",
    ),
    (
        "src/e_jepa_ttc/detection/event_fusion.py",
        "bd2663f322725644b61bc8806fb9cc4a83df467c39968b6844f547f431e78ffc",
        "post_dfull_20261031/event_gpu_smoke.json",
    ),
    (
        "src/e_jepa_ttc/detection/temporal_fusion.py",
        "308849fac626e9833b2de39d6a06cffde344c6ff305582b361816f00cd612f94",
        "post_dfull_20261031/event_local_gpu_parity.json",
    ),
]


@pytest.mark.parametrize(
    ("relative_path", "published_sha256", "receipt"),
    RECEIPT_BOUND_SOURCES,
    ids=[Path(path).stem for path, _, _ in RECEIPT_BOUND_SOURCES],
)
def test_detection_source_matches_published_receipt(
    relative_path: str, published_sha256: str, receipt: str
) -> None:
    """Detect source changes without rewriting any historical evidence."""
    assert published_sha256 in (EVIDENCE / receipt).read_text(encoding="utf-8")
    assert hashlib.sha256((REPOSITORY / relative_path).read_bytes()).hexdigest() == published_sha256
