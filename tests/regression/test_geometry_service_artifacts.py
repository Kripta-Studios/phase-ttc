"""The P5 CPU service evidence remains internally complete and hash-bound."""

from __future__ import annotations

from pathlib import Path

from operational.geometry_service.verify import verify_bundle


def test_long_cpu_clean_bundle_records_failed_strict_gate_and_exact_parity() -> None:
    archive = Path("docs/sota_20261121/geometry_service/long_cpu_clean_traces.zip")
    result = verify_bundle(archive)
    assert result == {
        "archive_sha256": "66ded9ca55ba0498c1843989689f73d99a5d441148bce976c47007286f3656eb",
        "source_bytes_rehashed": False,
        "frozen_input_bytes_rehashed": False,
        "cells": 8,
        "offered": 24_000,
        "valid": 23_995,
        "dropped": 5,
        "under_20ms": 23_992,
        "parity_objects": 59_980,
        "max_parity_abs": 0.0,
    }
