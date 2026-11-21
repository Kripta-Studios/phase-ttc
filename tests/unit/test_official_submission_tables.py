"""Official-submission tables are deterministic projections of bundled evidence."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper/from_motion_to_time_20261102"
SCRIPT = PAPER / "scripts/make_official_submission_tables.py"


def test_official_submission_tables_match_bundled_records() -> None:
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--check"],
        cwd=PAPER,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    long_table = (PAPER / "tables/official_submissions.tex").read_text()
    short_table = (PAPER / "tables/official_submissions_short.tex").read_text()
    assert "PHASE-TTC original" in long_table
    assert "1016512" in long_table
    assert "35.80" in long_table
    assert "PHASE-TTC-Geo" in long_table
    assert "1058935" in long_table
    assert "23.47" in long_table
    assert "45.03 / 53.13, 37.58, 40.61, 31.29" in short_table
    assert "23.47 / 24.37, 23.95, 22.89, 18.13" in short_table


def test_official_submission_evidence_retains_full_precision_and_source_hashes() -> None:
    evidence = json.loads(
        (PAPER / "data/official_submissions/records.json").read_text(encoding="utf-8")
    )
    assert evidence["schema"] == "phase_ttc_official_submissions_v1"
    assert evidence["reference"]["scores"]["overall_MiD"] == 45.03
    assert evidence["original"]["scores"]["overall_MiD"] == 35.80145158082389
    assert evidence["geometry"]["scores"]["overall_MiD"] == 23.472377543653998
    assert len(evidence["original"]["source_sha256"]) == 64
    assert len(evidence["geometry"]["source_sha256"]) == 64
