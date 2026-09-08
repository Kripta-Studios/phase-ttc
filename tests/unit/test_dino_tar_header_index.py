"""Exact source-byte lookup, durable metadata reuse and fail-closed TAR indexing."""

import io
import json
import os
import tarfile

import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, object_digest
from scripts.materialize_dinov3_relational_teacher import TarHeaderIndex


def make_tar(path, *, format=tarfile.PAX_FORMAT, long_name=False):
    name = ("nested/" + "x" * 150 if long_name else "frame") + ".jpg"
    with tarfile.open(path, "w", format=format) as archive:
        for member, data in [(name, b"first"), ("other.jpg", b"other"), (name, b"last")]:
            info = tarfile.TarInfo(member)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return name


@pytest.mark.parametrize(
    "format,long_name",
    [(tarfile.USTAR_FORMAT, False), (tarfile.PAX_FORMAT, True), (tarfile.GNU_FORMAT, True)],
)
def test_index_matches_last_native_member_and_reopens_without_scan(tmp_path, format, long_name):
    path = tmp_path / "rgb.tar"
    name = make_tar(path, format=format, long_name=long_name)
    root = tmp_path / "indices"
    for reader in [TarHeaderIndex(root), TarHeaderIndex(root)]:
        with tarfile.open(path, "r") as archive:
            member = reader.member(archive, path, name)
            assert archive.extractfile(member).read() == b"last"
            assert reader.cached_bytes <= 32 * 1024**2
    assert len(list(root.glob("*.json"))) == 1
    with tarfile.open(path, "r") as archive:
        # A persisted index needs only the requested header, not a full members scan.
        original = archive.next
        calls = []

        def next_header():
            calls.append(True)
            return original()

        archive.next = next_header
        TarHeaderIndex(root).member(archive, path, name)
        assert len(calls) <= 2  # Local PAX/GNU extension plus the actual header.


def test_resigned_wrong_offset_rejected_before_pixel_read(tmp_path):
    path = tmp_path / "rgb.tar"
    name = make_tar(path)
    root = tmp_path / "indices"
    with tarfile.open(path, "r") as archive:
        TarHeaderIndex(root).member(archive, path, name)
    receipt = next(root.glob("*.json"))
    value = json.loads(receipt.read_text())
    value.pop("artifact_sha256")
    value["entries"][name] = value["entries"]["other.jpg"]
    value["artifact_sha256"] = object_digest(value)
    atomic_json(receipt, value)
    with tarfile.open(path, "r") as archive:
        with pytest.raises(ValueError, match="header differs"):
            TarHeaderIndex(root).member(archive, path, name)


def test_changed_source_and_oversized_index_rejected(tmp_path):
    path = tmp_path / "rgb.tar"
    name = make_tar(path)
    root = tmp_path / "indices"
    reader = TarHeaderIndex(root)
    with tarfile.open(path, "r") as archive:
        reader.member(archive, path, name)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with tarfile.open(path, "r") as archive:
        with pytest.raises(ValueError, match="changed"):
            reader.member(archive, path, name)
    receipt = next(root.glob("*.json"))
    receipt.write_bytes(b"x" * (4 * 1024**2 + 1))
    with tarfile.open(path, "r") as archive:
        with pytest.raises(ValueError, match="metadata size/path"):
            reader.member(archive, path, name)


def test_metadata_lru_is_bounded_and_disk_indices_are_reusable(tmp_path):
    root = tmp_path / "indices"
    reader = TarHeaderIndex(root)
    for i in range(10):
        path = tmp_path / f"rgb{i}.tar"
        name = make_tar(path)
        with tarfile.open(path, "r") as archive:
            reader.member(archive, path, name)
        assert len(reader.cached) <= 8 and reader.cached_bytes <= 32 * 1024**2
    path = tmp_path / "rgb0.tar"
    with tarfile.open(path, "r") as archive:
        assert archive.extractfile(reader.member(archive, path, name)).read() == b"last"
