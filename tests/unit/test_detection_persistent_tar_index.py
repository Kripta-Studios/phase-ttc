"""CPU contracts for the isolated persistent TAR-offset candidate."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from e_jepa_ttc.detection.persistent_tar_index import (
    PersistentTarRGBReader,
    build_persistent_tar_index,
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_tar(path: Path, members: list[tuple[str, bytes, str]]) -> str:
    with tarfile.open(path, "w") as archive:
        for name, value, kind in members:
            info = tarfile.TarInfo(name)
            info.size = len(value)
            if kind == "file":
                archive.addfile(info, io.BytesIO(value))
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                info.size = 0
                archive.addfile(info)
            else:
                raise AssertionError(kind)
    return digest(path.read_bytes())


def fixture_index(
    tmp_path: Path, *, shards: int = 1
) -> tuple[dict[str, tuple[str, str, str]], dict[str, str], Path, str]:
    ownership: dict[str, tuple[str, str, str]] = {}
    shard_hashes: dict[str, str] = {}
    for index in range(shards):
        relative = f"s{index}.tar"
        payload = f"payload-{index}".encode()
        shard_hashes[relative] = write_tar(
            tmp_path / relative,
            [("rgb/a.png", payload, "file"), ("unused.bin", b"unused", "file")],
        )
        ownership[f"t{index}"] = (relative, "rgb/a.png", digest(payload))
    sidecar = tmp_path / "index.json"
    sidecar_sha = build_persistent_tar_index(tmp_path, ownership, shard_hashes, sidecar)
    return ownership, shard_hashes, sidecar, sidecar_sha


def test_reuse_does_not_rescan_tar_headers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ownership, shards, sidecar, sidecar_sha = fixture_index(tmp_path)
    monkeypatch.setattr(tarfile, "open", lambda *_args, **_kwargs: pytest.fail("rescanned TAR"))
    with PersistentTarRGBReader(tmp_path, ownership, shards, sidecar, sidecar_sha) as reader:
        assert reader.read("t0") == b"payload-0"


def test_reuse_still_hashes_full_tar_and_catches_same_stat_mutation(tmp_path: Path) -> None:
    ownership, shards, sidecar, sidecar_sha = fixture_index(tmp_path)
    tar_path = tmp_path / "s0.tar"
    before = tar_path.stat()
    raw = bytearray(tar_path.read_bytes())
    raw[700] ^= 1
    tar_path.write_bytes(raw)
    os.utime(tar_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert tar_path.stat().st_size == before.st_size
    with pytest.raises(ValueError, match="source SHA-256"):
        PersistentTarRGBReader(tmp_path, ownership, shards, sidecar, sidecar_sha)


def test_sidecar_sha_is_required_before_json_is_trusted(tmp_path: Path) -> None:
    ownership, shards, sidecar, sidecar_sha = fixture_index(tmp_path)
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    with pytest.raises(ValueError, match="index SHA-256"):
        PersistentTarRGBReader(tmp_path, ownership, shards, sidecar, sidecar_sha)


def test_stale_but_rehashed_offset_fails_on_payload_digest(tmp_path: Path) -> None:
    ownership, shards, sidecar, _ = fixture_index(tmp_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["tokens"][0]["offset_data"] += tarfile.BLOCKSIZE
    sidecar.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    altered_sha = digest(sidecar.read_bytes())
    with PersistentTarRGBReader(tmp_path, ownership, shards, sidecar, altered_sha) as reader:
        with pytest.raises(ValueError, match="member SHA-256"):
            reader.read("t0")


def test_requested_ownership_must_exactly_match_sidecar(tmp_path: Path) -> None:
    ownership, shards, sidecar, sidecar_sha = fixture_index(tmp_path)
    wrong = {"t0": ("s0.tar", "unused.bin", digest(b"unused"))}
    with pytest.raises(ValueError, match="ownership differs"):
        PersistentTarRGBReader(tmp_path, wrong, shards, sidecar, sidecar_sha)
    extra = {**ownership, "other": ("s0.tar", "unused.bin", digest(b"unused"))}
    with pytest.raises(ValueError, match="population"):
        PersistentTarRGBReader(tmp_path, extra, shards, sidecar, sidecar_sha)


@pytest.mark.parametrize(
    ("members", "match"),
    [
        (
            [("rgb/a.png", b"a", "file"), ("dup", b"x", "file"), ("dup", b"y", "file")],
            "duplicate",
        ),
        ([("rgb/a.png", b"a", "file"), ("folder", b"", "dir")], "regular"),
        ([("rgb/a.png", b"a", "file"), ("../escape", b"x", "file")], "Unsafe"),
    ],
)
def test_builder_rejects_hostile_unrequested_members(
    tmp_path: Path, members: list[tuple[str, bytes, str]], match: str
) -> None:
    shard_sha = write_tar(tmp_path / "s.tar", members)
    with pytest.raises(ValueError, match=match):
        build_persistent_tar_index(
            tmp_path,
            {"t": ("s.tar", "rgb/a.png", digest(b"a"))},
            {"s.tar": shard_sha},
            tmp_path / "index.json",
        )


def test_sidecar_is_write_once_and_handles_remain_bounded(tmp_path: Path) -> None:
    ownership, shards, sidecar, sidecar_sha = fixture_index(tmp_path, shards=3)
    with pytest.raises(FileExistsError):
        build_persistent_tar_index(tmp_path, ownership, shards, sidecar)
    with PersistentTarRGBReader(
        tmp_path, ownership, shards, sidecar, sidecar_sha, max_open_handles=2
    ) as reader:
        for index in range(3):
            assert reader.read(f"t{index}") == f"payload-{index}".encode()
        assert reader.open_handle_count == 2
    assert reader.open_handle_count == 0
