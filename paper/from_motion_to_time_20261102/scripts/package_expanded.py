"""Package the full expanded article and record continuity with the original."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import verify_publication

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent
REPO = ORIGINAL.parent
STEM = "PHASE_TTC_From_Motion_to_Time_Expanded"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publication_files() -> list[Path]:
    """Match the publication inventory, excluding local compilation products."""
    return [
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and not any(part in {"__pycache__", ".ruff_cache"} for part in path.relative_to(ROOT).parts)
        and (
            "build" not in path.relative_to(ROOT).parts
            or path.relative_to(ROOT).as_posix() == "build/validation.json"
        )
        and (
            path.suffix
            not in {".aux", ".bbl", ".log", ".out", ".blg", ".pyc", ".toc", ".fls", ".fdb_latexmk"}
            or path.relative_to(ROOT).as_posix()
            in {"data/verification_architecture.log", "data/verification_protocol.log"}
            or path.relative_to(ROOT).parts[:2] == ("audit_20261104", "sources")
        )
        and not path.name.endswith(".synctex.gz")
        and path.name
        not in {
            "GUIA_REESCRITURA_NATURAL_PAPEL.tex",
            "GUIA_REESCRITURA_NATURAL_PAPEL.pdf",
        }
        and path.relative_to(ROOT).as_posix() not in {"main.pdf", "SOURCE_MANIFEST.json"}
    ]


def write_source_manifest(files: list[Path]) -> Path:
    """Record exact source and final-artifact bytes without compiling the paper."""
    manifest = {path.relative_to(ROOT).as_posix(): digest(path) for path in files}
    manifest_path = ROOT / "SOURCE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def original_assets(directory: str) -> list[Path]:
    """Check published assets, excluding previews and TeX products from local rebuilds."""
    manifest = json.loads((ORIGINAL / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    paths = []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if not relative.parts or relative.parts[0] != directory:
            continue
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe original source-manifest path")
        path = ORIGINAL / relative
        if not path.is_file() or digest(path) != entry["sha256"]:
            raise ValueError(f"Original published asset changed: {relative}")
        paths.append(path)
    return sorted(paths)


def main() -> None:
    # Fail before copying files or writing a new manifest for an obsolete PDF.
    binding = verify_publication.verify(ROOT)
    included = {path.relative_to(ROOT).as_posix() for path in publication_files()}
    required = set(verify_publication.source_hashes(ROOT)) | set(verify_publication.ARTIFACTS)
    if missing := required - included:
        raise ValueError(f"Publication inventory excludes required build inputs: {sorted(missing)}")
    original_inputs = re.findall(
        r"\\input\{(sections/[^}]+)\}", (ORIGINAL / "main.tex").read_text()
    )
    new_inputs = re.findall(r"\\input\{(sections/[^}]+)\}", (ROOT / "main.tex").read_text())
    if not set(original_inputs) <= set(new_inputs):
        raise ValueError("An original manuscript section is missing")
    unchanged = {}
    for directory in ["figures", "data", "research"]:
        names = original_assets(directory)
        for path in names:
            if digest(path) != digest(ROOT / path.relative_to(ORIGINAL)):
                raise ValueError(f"Original asset changed: {path}")
        unchanged[directory] = len(names)
    retention = {
        "original_section_inputs": original_inputs,
        "expanded_section_inputs": new_inputs,
        "all_original_sections_retained": True,
        "byte_identical_original_assets": unchanged,
        "section_changes": [
            {
                "path": str(path.relative_to(ORIGINAL)),
                "original_sha256": digest(path),
                "expanded_sha256": digest(ROOT / path.relative_to(ORIGINAL)),
                "changed": digest(path) != digest(ROOT / path.relative_to(ORIGINAL)),
            }
            for path in sorted((ORIGINAL / "sections").glob("*.tex"))
        ],
    }
    (ROOT / "updates/review/original_retention.json").write_text(
        json.dumps(retention, indent=2) + "\n", encoding="utf-8"
    )
    validation = json.loads((ROOT / "build/validation.json").read_text())
    (ROOT / "updates/review/expanded_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    output = REPO / "output/pdf"
    output.mkdir(parents=True, exist_ok=True)
    for suffix in [".tex", ".pdf"]:
        shutil.copy2(ROOT / (STEM + suffix), output / (STEM + suffix))
    for directory in ["figures", "updates/figures"]:
        (output / directory).mkdir(parents=True, exist_ok=True)
        for path in (ROOT / directory).iterdir():
            if path.suffix in {".pdf", ".png", ".jpg"}:
                shutil.copy2(path, output / directory / path.name)
    files = publication_files()
    files.append(write_source_manifest(files))
    archive_path = output / f"{STEM}_sources.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT))
    receipt = {
        "pdf": STEM + ".pdf",
        "pdf_sha256": digest(output / (STEM + ".pdf")),
        "source_zip_sha256": digest(archive_path),
        "files": len(files),
        "pages": validation["pages"],
        "words": validation["words"],
        "figures": validation["figures"],
        "tables": validation["tables"],
        "publication_binding": binding,
    }
    (output / f"{STEM}_delivery.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
