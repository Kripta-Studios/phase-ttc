from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "docs/sota_20261121/distribution"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hf_asset_diff_matches_exact_five_files() -> None:
    manifest = json.loads((DIST / "HF_ASSET_DIFF.json").read_text(encoding="utf-8"))
    assets = manifest["assets"]
    assert len(assets) == 5
    assert {asset["remote_path"] for asset in assets} == {
        "README.md",
        "index.html",
        "gradio_app.zip",
    }
    for asset in assets:
        path = DIST / asset["local_path"]
        assert path.stat().st_size == asset["after_bytes"]
        assert _sha256(path) == asset["after_sha256"]


def test_cards_space_and_gradio_zip_record_third_submission() -> None:
    paths = [
        DIST / "hf/PHASE-TTC/README.md",
        DIST / "hf/PHASE-TTC-Adapt/README.md",
        DIST / "space/README.md",
        DIST / "space/index.html",
    ]
    required = ("1063261", "27.97563", "4.50325", "26.38540", "30.89121")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert all(value in text for value in required), path
        assert (
            "public leaderboard display after the third submission has not been archived"
            in text.lower()
            or "display policy after the third submission is unknown" in text.lower()
        )

    archive_path = DIST / "space/gradio_app.zip"
    with zipfile.ZipFile(archive_path) as archive:
        archived_app = archive.read("app.py")
    source_app = (DIST / "space/gradio_app/app.py").read_bytes()
    assert archived_app == source_app
    archived_text = archived_app.decode("utf-8")
    assert all(value in archived_text for value in required)
