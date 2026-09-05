"""Package a terminal Stage66 block with an extracted-file hash and CRC round trip."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def safe_member(name: str) -> None:
    """Reject absolute, alternate-separator and traversal ZIP member names."""
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"unsafe member: {name}")


def file_digest(path: Path) -> str:
    """Hash bytes with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(source: Path, archive: Path, extracted: Path) -> dict[str, object]:
    """Write a new archive, reopen, extract, and verify every member's identity."""
    source = source.resolve(strict=True)
    archive = archive.resolve()
    if archive.is_relative_to(source) or archive.exists() or extracted.exists():
        raise ValueError("archive must be new, outside source; extraction must be new")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    records = []
    seen: set[str] = set()
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError("source symlink/escape")
        name = path.relative_to(source).as_posix()
        safe_member(name)
        if name.casefold() in seen or name == "ESSENTIAL_MANIFEST.json":
            raise ValueError("duplicate or reserved member")
        seen.add(name.casefold())
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": file_digest(path)})
    manifest = json.dumps({"files": records}, indent=2).encode()
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path, record in zip(files, records, strict=True):
            bundle.write(path, str(record["path"]))
        bundle.writestr("ESSENTIAL_MANIFEST.json", manifest)
    temporary.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(name.casefold() for name in names)):
            raise ValueError("duplicate archive members")
        for name in names:
            safe_member(name)
        if bundle.testzip() is not None:
            raise ValueError("ZIP CRC failed")
        bundle.extractall(extracted)
    for record in records:
        path = extracted / str(record["path"])
        if path.stat().st_size != record["bytes"] or file_digest(path) != record["sha256"]:
            raise ValueError(f"extracted identity mismatch: {path}")
    digest = file_digest(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return {
        "archive": str(archive),
        "sha256": digest,
        "crc_verified": True,
        "extracted_files_verified": len(records),
        "members": len(records) + 1,
    }


def main() -> None:
    """Expose packaging separately from all model and data code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extracted", type=Path, required=True)
    args = parser.parse_args()
    result = package(args.source, args.archive, args.extracted)
    args.archive.with_suffix(".zip.verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
