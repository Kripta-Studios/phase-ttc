from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "docs/sota_20261121/lineage/INDEX.json"
VERIFY = ROOT / "scripts/verify_submission_lineage.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_lineage_index_binds_git_blobs_and_versioned_artifacts() -> None:
    payload = json.loads(INDEX.read_text(encoding="utf-8"))

    assert payload["schema"] == "phase_ttc_submission_lineage_v1"
    assert payload["submission_history"]["owner_declared_upload_count"] == 2
    assert payload["submission_history"]["matched_a_status"] == "prepared_not_uploaded"
    assert payload["claims"]["end_to_end_geo_reproduction"] is False

    for binding in payload["train40_parent"]["preserved_source_blob_bindings"]:
        actual = subprocess.run(
            ["git", "rev-parse", f"{binding['commit']}:{binding['path']}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert actual == binding["git_blob_sha1"]

    for artifact in payload["versioned_artifacts"]:
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]

    for name, result in payload["reproduction_evidence"].items():
        if not isinstance(result, dict) or "result" not in result:
            continue
        result_path = (INDEX.parent / result["result"]).resolve()
        assert _sha256(result_path) == result["result_sha256"], name

    portable = payload["reproduction_evidence"]["portable_bundle"]
    read_only = (INDEX.parent / portable["read_only_test"]).resolve()
    assert _sha256(read_only) == portable["read_only_test_sha256"]


def test_lineage_verifier_accepts_repository_only_mode() -> None:
    result = subprocess.run(
        [sys.executable, str(VERIFY), "--repo", str(ROOT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "repository checks: PASS" in result.stdout
    assert "external checks: NOT RUN" in result.stdout
