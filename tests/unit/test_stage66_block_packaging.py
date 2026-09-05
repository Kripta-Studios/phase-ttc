"""Terminal-block evidence must round-trip and cannot overwrite or traverse."""

from pathlib import Path

import pytest

from scripts.package_stage66_block import package, safe_member


@pytest.mark.parametrize("name", ["../x", "/x", "C:/x", "a\\b", "a/../../x", ""])
def test_unsafe_member_rejected(name):
    with pytest.raises(ValueError):
        safe_member(name)


def test_roundtrip_preserves_block_and_arrays(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "decision.json").write_text('{"status":"INTEGRITY_BLOCKED"}')
    (source / "bytes.bin").write_bytes(bytes(range(256)))
    result = package(source, tmp_path / "bundle.zip", tmp_path / "extracted")
    assert result["crc_verified"] and result["extracted_files_verified"] == 2
    assert (tmp_path / "extracted/bytes.bin").read_bytes() == bytes(range(256))


def test_no_overwrite_or_recursive_self_inclusion(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError):
        package(source, source / "bundle.zip", tmp_path / "extracted")
    archive = tmp_path / "existing.zip"
    archive.write_bytes(b"preserve")
    with pytest.raises(ValueError):
        package(source, archive, tmp_path / "extracted")
    assert archive.read_bytes() == b"preserve"


def test_reserved_manifest_rejected(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "ESSENTIAL_MANIFEST.json").write_text("{}")
    with pytest.raises(ValueError):
        package(source, tmp_path / "bundle.zip", tmp_path / "extracted")
