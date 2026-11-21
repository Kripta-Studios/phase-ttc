"""The expanded-paper follow-up tables are deterministic data projections."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper/from_motion_to_time_20261102"
SCRIPT = PAPER / "scripts/make_followup_tables.py"

EXPECTED_TABLE_SHA256 = {
    "tables/followup_fcwd.tex": "ffe39fb54029c1edd1172d4c46ee160799595b175d6ee25f4711286d8f9d52e6",
    "tables/followup_latency.tex": (
        "c5135fb46a3883f6cd6424bf37adde6d4cce756822b0ab3b63da3e197318332f"
    ),
    "tables/followup_literature.tex": (
        "1b63c936f5c9e92d48bd8e38116a35f599f218161bc50e1ccbeccdffcf46853b"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_followup_tables_match_versioned_inputs_and_golden_bytes() -> None:
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--check"],
        cwd=PAPER,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for name in EXPECTED_TABLE_SHA256:
        assert b"\\\\n" not in (PAPER / name).read_bytes()
    assert {name: _sha256(PAPER / name) for name in EXPECTED_TABLE_SHA256} == (
        EXPECTED_TABLE_SHA256
    )


def test_followup_input_manifest_covers_every_packaged_input() -> None:
    manifest = json.loads((PAPER / "data/followup/source_manifest.json").read_text())
    expected = {
        "event_geometry_fcwd.csv",
        "geometry_robust_fcwd.csv",
        "literature_main.csv",
        "saturation_scale_fcwd.csv",
        "variant_latency_results.json",
    }
    assert manifest["schema"] == "phase_ttc_followup_table_sources_v1"
    assert set(manifest["files"]) == expected
    for name, expected_hash in manifest["files"].items():
        assert _sha256(PAPER / "data/followup" / name) == expected_hash
