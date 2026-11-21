"""Build a deterministic, portable archive of service replay evidence."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

FIXED_ZIP_TIME = (2026, 11, 21, 0, 0, 0)
METADATA_FILES = (
    "RESULTS.json",
    "POPULATION.json",
    "REPORT.md",
    "PROTOCOL.md",
    "MANIFEST.json",
    "GATE_ASSESSMENT.md",
)


def trace_files(bundle: Path) -> list[Path]:
    """Return only sealed metadata and per-cell JSON traces, in stable order."""
    cells = sorted({*bundle.glob("supplied_*.json"), *bundle.glob("rgb_geo__supplied_*.json")})
    metadata = [bundle / name for name in METADATA_FILES if (bundle / name).is_file()]
    if not cells:
        raise ValueError(f"no per-cell service traces in {bundle}")
    return sorted([*metadata, *cells], key=lambda path: path.name)


def pack(bundle: Path, output: Path) -> str:
    """Write a byte-reproducible ZIP and return its SHA-256 digest."""
    files = trace_files(bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.name, date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    temporary.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii"
    )
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(pack(args.bundle, args.output))


if __name__ == "__main__":
    main()
