"""Verify the versioned and optionally external PHASE-TTC submission lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    """Return a lowercase SHA-256 for one file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def require_hash(path: Path, expected: str) -> None:
    """Fail if a file is absent or differs from its expected SHA-256."""
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def verify_repository(repo: Path, index: dict[str, Any]) -> None:
    """Verify Git blob bindings and all small/versioned evidence."""
    for binding in index["train40_parent"]["preserved_source_blob_bindings"]:
        revision = f"{binding['commit']}:{binding['path']}"
        actual = subprocess.run(
            ["git", "rev-parse", revision],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != binding["git_blob_sha1"]:
            raise ValueError(f"Git blob mismatch for {revision}: {actual}")
    for artifact in index["versioned_artifacts"]:
        require_hash(repo / artifact["path"], artifact["sha256"])
    geo = index["geo_candidate"]
    receipt = read_json(repo / geo["code_receipt_path"])
    commit = geo["first_source_commit"]
    for name, expected in receipt["code_sha256"].items():
        revision = f"{commit}:operational/phase_geometry/{name}"
        source = subprocess.run(
            ["git", "show", revision],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        actual = hashlib.sha256(source).hexdigest()
        if actual != expected:
            raise ValueError(f"Geo source mismatch for {revision}: {actual} != {expected}")


def verify_pod(pod_root: Path, index: dict[str, Any]) -> None:
    """Verify the preserved TRAIN40 sources, weights, recipe, seal and ZIP."""
    parent = index["train40_parent"]
    archive = parent["preserved_source_archive"]
    require_hash(pod_root / archive["external_relative_path"], archive["sha256"])
    recipe = parent["recipe"]
    require_hash(pod_root / recipe["external_relative_path"], recipe["file_sha256"])
    candidate = pod_root / "FINAL_CANDIDATE"
    require_hash(candidate / "MODEL_SEAL.json", parent["model_seal_sha256"])
    require_hash(candidate / "PREDICTION_RECEIPT.json", parent["prediction_receipt_sha256"])
    require_hash(
        candidate / "PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.zip",
        parent["submitted_parent_zip_sha256"],
    )
    for name, expected in parent["checkpoint_sha256"].items():
        require_hash(candidate / "checkpoints" / name, expected)
    for name, expected in parent["normalizer_sha256"].items():
        require_hash(candidate / name, expected)


def main() -> int:
    """Run repository checks and optional external checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--pod-root", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    index = read_json(repo / "docs/sota_20261121/lineage/INDEX.json")
    verify_repository(repo, index)
    print("repository checks: PASS")
    if args.pod_root is None:
        print("external checks: NOT RUN (pass --pod-root to hash the pod return)")
    else:
        verify_pod(args.pod_root.resolve(), index)
        print("external checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
