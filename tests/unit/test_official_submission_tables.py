"""Official-submission tables are deterministic projections of bundled evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper/from_motion_to_time_20261102"
SCRIPT = PAPER / "scripts/make_official_submission_tables.py"


def load_generator_module():
    spec = importlib.util.spec_from_file_location("official_submission_generator", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert "Box-history comparator" in long_table
    assert "1063261" in long_table
    assert "27.98" in long_table
    assert "45.03 / 53.13, 37.58, 40.61, 31.29" in short_table
    assert "23.47 / 24.37, 23.95, 22.89, 18.13" in short_table
    assert "27.98 / 26.39, 28.00, 32.94, 30.89" in short_table


def test_official_submission_evidence_retains_full_precision_and_source_hashes() -> None:
    evidence = json.loads(
        (PAPER / "data/official_submissions/records.json").read_text(encoding="utf-8")
    )
    assert evidence["schema"] == "phase_ttc_official_submissions_v3"
    assert evidence["reference"]["scores"]["overall_MiD"] == 45.03
    assert evidence["original"]["scores"]["overall_MiD"] == 35.80145158082389
    assert evidence["geometry"]["scores"]["overall_MiD"] == 23.472377543653998
    assert evidence["box_only"]["scores"]["overall_MiD"] == 27.975630860024584
    assert len(evidence["original"]["source_sha256"]) == 64
    assert len(evidence["geometry"]["source_sha256"]) == 64
    assert len(evidence["box_only"]["source_sha256"]) == 64
    assert len(evidence["reference"]["source_lf_sha256"]) == 64
    assert len(evidence["original"]["source_lf_sha256"]) == 64
    assert len(evidence["geometry"]["source_lf_sha256"]) == 64
    assert len(evidence["box_only"]["source_lf_sha256"]) == 64


def test_source_hashes_bind_canonical_checkout_and_exact_git_blobs() -> None:
    evidence = json.loads(
        (PAPER / "data/official_submissions/records.json").read_text(encoding="utf-8")
    )
    paths = {
        "reference": "paper/data/official_benchmark.json",
        "original": "paper/data/official_benchmark.json",
        "geometry": "docs/sota_20261118/leaderboard/submission_1058935/RECORD.json",
        "box_only": "docs/sota_20261118/leaderboard/submission_1063261/RECORD.json",
    }
    for key, relative in paths.items():
        checkout = (REPO / relative).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(checkout).hexdigest() == evidence[key]["source_lf_sha256"]
        blob = subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "blob", f"HEAD:{relative}"],
            capture_output=True,
            check=True,
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == evidence[key]["source_sha256"]


def test_lf_source_bundle_validates_without_git_binary(tmp_path, monkeypatch) -> None:
    module = load_generator_module()
    bundle = tmp_path / "source_bundle"
    bundle_paper = bundle / "paper/from_motion_to_time_20261102"
    sources = (
        "paper/data/official_benchmark.json",
        "docs/sota_20261118/leaderboard/submission_1058935/RECORD.json",
        "docs/sota_20261118/leaderboard/submission_1063261/RECORD.json",
    )
    for relative in sources:
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / relative).read_bytes().replace(b"\r\n", b"\n"))
    evidence = bundle_paper / "data/official_submissions/records.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PAPER / "data/official_submissions/records.json", evidence)

    module.REPO = bundle
    module.ROOT = bundle_paper
    module.EVIDENCE = evidence

    def reject_git_call(*args, **kwargs):
        raise AssertionError("source bundles without .git must not invoke Git")

    monkeypatch.setattr(module.subprocess, "run", reject_git_call)
    loaded = module.load()
    module.validate_repository_sources(loaded)
