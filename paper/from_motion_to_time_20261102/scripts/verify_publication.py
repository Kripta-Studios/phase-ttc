"""Bind a manuscript delivery to the exact inputs used for its successful build."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEM = "PHASE_TTC_From_Motion_to_Time_Expanded"
CONTRACT = "revision/publication_contract.json"
SOURCE_DIRS = (
    "sections",
    "tables",
    "figures",
    "data",
    "research",
    "scripts",
    "audit_20261104",
    "updates/evidence",
    "updates/figures",
)
SOURCE_SUFFIXES = {
    ".tex",
    ".sty",
    ".bib",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".json",
    ".csv",
    ".npz",
    ".py",
    ".xml",
    ".log",
    ".txt",
    ".md",
    ".yaml",
    ".yml",
}
ARTIFACTS = (STEM + ".pdf", STEM + ".tex", "build/validation.json")


def digest(path: Path) -> str:
    """Return the SHA-256 of the exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    """Inventory scientific inputs while excluding build and review receipts."""
    files = {
        root / name
        for name in (
            "main.tex",
            "paper_style.sty",
            "references.bib",
            "updates/evidence_tables.tex",
            "updates/build_evidence.py",
            "revision/deployment_contract.json",
            "revision/generated_figures.json",
            "revision/derived_transfer_analysis.json",
            "revision/bootstrap_differences.csv",
        )
        if (root / name).is_file()
    }
    for directory in SOURCE_DIRS:
        files.update(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and path.suffix in SOURCE_SUFFIXES
            and "__pycache__" not in path.parts
            # TeX figure logs are compilation products, not source evidence.
            and not (directory == "figures" and path.suffix == ".log")
        )
    return {path.relative_to(root).as_posix(): digest(path) for path in sorted(files)}


def record_build(root: Path, before: dict[str, str]) -> None:
    """Seal artifacts only if scientific inputs stayed unchanged during compilation."""
    if source_hashes(root) != before:
        raise ValueError("Manuscript inputs changed during compilation; rebuild before packaging")
    receipt = {
        "schema": "phase_ttc_publication_binding_v1",
        "scope": "Scientific inputs and compiled delivery; no new training or sensor inference",
        "sources": before,
        "artifacts": {name: digest(root / name) for name in ARTIFACTS},
    }
    (root / CONTRACT).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def verify(root: Path) -> dict[str, object]:
    """Reject stale sources, missing bindings, or modified compiled artifacts."""
    path = root / CONTRACT
    if not path.is_file():
        raise ValueError("Missing publication binding; rebuild with scripts/build_expanded.py")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "phase_ttc_publication_binding_v1":
        raise ValueError("Unknown publication binding; rebuild before packaging")
    current = source_hashes(root)
    previous = receipt.get("sources", {})
    changed = sorted(
        name for name in current.keys() | previous.keys() if current.get(name) != previous.get(name)
    )
    if changed:
        raise ValueError(f"Sources changed since compilation; rebuild: {changed}")
    if set(receipt.get("artifacts", {})) != set(ARTIFACTS):
        raise ValueError("Incomplete artifact binding; rebuild before packaging")
    for name in ARTIFACTS:
        if not (root / name).is_file() or digest(root / name) != receipt["artifacts"][name]:
            raise ValueError(f"Compiled artifact changed; rebuild: {name}")
    return {"status": "PASS", "scientific_inputs": len(current), "artifacts": len(ARTIFACTS)}


if __name__ == "__main__":
    print(json.dumps(verify(ROOT), indent=2))
