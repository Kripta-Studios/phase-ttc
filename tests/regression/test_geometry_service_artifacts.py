"""The P5 CPU service evidence remains internally complete and hash-bound."""

from __future__ import annotations

import json
import zipfile
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


def test_rgb_long_contended_bundle_is_portable_and_explicitly_invalidated() -> None:
    archive = Path("docs/sota_20261121/geometry_service/rgb_gpu_long_contended_traces.zip")
    result = verify_bundle(archive)
    assert result == {
        "archive_sha256": "d202daf41921ee37c57ec635977df1188d5804f8681949ed5d62fc98e2e03134",
        "source_bytes_rehashed": False,
        "frozen_input_bytes_rehashed": False,
        "cells": 8,
        "offered": 24_000,
        "valid": 1_886,
        "dropped": 22_114,
        "under_20ms": 0,
        "parity_objects": 3_236,
        "max_parity_abs": 0.0,
    }
    with zipfile.ZipFile(archive) as bundle:
        results = json.loads(bundle.read("RESULTS.json"))
    assert results["strict_service_gate"] == {
        "capacity_decision": "FAIL",
        "effective_decision": "INVALIDATED",
        "passing_cells": 0,
        "total_cells": 8,
    }
    assert results["gpu_compute_processes"]["audit"]["foreign_python_pids"] == [37868]
