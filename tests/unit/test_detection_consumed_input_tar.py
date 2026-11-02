"""CPU-only tests for the narrowed consumed-RGB identity contract."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

import e_jepa_ttc.detection.consumed_input_tar as consumed_module
from e_jepa_ttc.detection.consumed_input_tar import ConsumedInputTarRGBReader
from e_jepa_ttc.detection.persistent_tar_index import build_persistent_tar_index


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def setup_contract(
    tmp_path: Path,
) -> tuple[dict[str, tuple[str, str, str]], dict[str, str], Path, str, Path, str]:
    selected = b"selected-payload"
    with tarfile.open(tmp_path / "s.tar", "w") as archive:
        for name, value in (("rgb/a.png", selected), ("unused.bin", b"unselected-payload")):
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
    shard_hashes = {"s.tar": digest((tmp_path / "s.tar").read_bytes())}
    ownership = {"t": ("s.tar", "rgb/a.png", digest(selected))}
    sidecar = tmp_path / "index.json"
    sidecar_sha = build_persistent_tar_index(tmp_path, ownership, shard_hashes, sidecar)
    dependency = tmp_path / "manifest.json"
    dependency.write_text('{"population":"fixed"}\n', encoding="utf-8")
    dependency_sha = digest(dependency.read_bytes())
    return ownership, shard_hashes, sidecar, sidecar_sha, dependency, dependency_sha


def reader(
    tmp_path: Path,
    *,
    resume: dict[str, object] | None = None,
) -> ConsumedInputTarRGBReader:
    ownership, shards, sidecar, sidecar_sha, dependency, dependency_sha = setup_contract(tmp_path)
    return ConsumedInputTarRGBReader(
        tmp_path,
        ownership,
        shards,
        sidecar,
        sidecar_sha,
        {dependency: dependency_sha},
        resume_ledger=resume,
    )


def test_constructor_does_not_hash_current_whole_tar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ownership, shards, sidecar, sidecar_sha, dependency, dependency_sha = setup_contract(tmp_path)
    original = consumed_module.sha256_file
    observed: list[Path] = []

    def recording_hash(path: Path) -> str:
        observed.append(Path(path))
        return original(path)

    monkeypatch.setattr(consumed_module, "sha256_file", recording_hash)
    with ConsumedInputTarRGBReader(
        tmp_path, ownership, shards, sidecar, sidecar_sha, {dependency: dependency_sha}
    ) as candidate:
        assert candidate.whole_archive_reverified_this_process is False
        evidence = candidate.contract_evidence()
        assert evidence["all_selected_payloads_preflight_verified"] is False
        assert evidence["selected_payloads_verified_per_read"] is True
        assert evidence["fallback_permitted"] is False
        assert candidate.read("t", cursor=0) == b"selected-payload"
    assert observed == [dependency.resolve()]


def test_unselected_mutation_is_outside_claim_but_selected_bytes_stay_exact(tmp_path: Path) -> None:
    ownership, shards, sidecar, sidecar_sha, dependency, dependency_sha = setup_contract(tmp_path)
    tar_path = tmp_path / "s.tar"
    before = tar_path.stat()
    raw = bytearray(tar_path.read_bytes())
    unused_offset = json.loads(sidecar.read_text(encoding="utf-8"))["shards"][0]["identity"]["size"]
    assert unused_offset == len(raw)
    raw[1536] ^= 1  # second member payload; selected payload is at offset 512
    tar_path.write_bytes(raw)
    os.utime(tar_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with ConsumedInputTarRGBReader(
        tmp_path, ownership, shards, sidecar, sidecar_sha, {dependency: dependency_sha}
    ) as candidate:
        assert candidate.unselected_archive_current_identity == "NOT_CLAIMED"
        assert candidate.read("t", cursor=0) == b"selected-payload"


def test_selected_mutation_aborts_before_ledger_advance(tmp_path: Path) -> None:
    ownership, shards, sidecar, sidecar_sha, dependency, dependency_sha = setup_contract(tmp_path)
    tar_path = tmp_path / "s.tar"
    before = tar_path.stat()
    raw = bytearray(tar_path.read_bytes())
    raw[512] ^= 1
    tar_path.write_bytes(raw)
    os.utime(tar_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with ConsumedInputTarRGBReader(
        tmp_path, ownership, shards, sidecar, sidecar_sha, {dependency: dependency_sha}
    ) as candidate:
        with pytest.raises(ValueError, match="payload SHA-256 differs"):
            candidate.read("t", cursor=0)
        assert candidate.ledger_state()["next_cursor"] == 0


def test_source_stat_is_checked_before_and_after_same_handle_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with reader(tmp_path) as candidate:
        original = candidate._assert_source
        calls: list[str] = []

        def recording_assert(relative: str) -> None:
            calls.append(relative)
            original(relative)

        monkeypatch.setattr(candidate, "_assert_source", recording_assert)
        candidate.read("t", cursor=0)
    assert calls == ["s.tar", "s.tar"]


def test_every_repeat_is_ledgered_and_resume_continues_exact_cursor(tmp_path: Path) -> None:
    with reader(tmp_path) as first:
        first.read("t", cursor=0)
        first.read("t", cursor=1)
        state = first.ledger_state()
    assert state["next_cursor"] == 2
    assert len(state["entries"]) == 2  # type: ignore[arg-type]

    # Reuse the same immutable files rather than rebuilding the fixture.
    ownership = {"t": ("s.tar", "rgb/a.png", digest(b"selected-payload"))}
    shards = {"s.tar": digest((tmp_path / "s.tar").read_bytes())}
    sidecar = tmp_path / "index.json"
    dependency = tmp_path / "manifest.json"
    with ConsumedInputTarRGBReader(
        tmp_path,
        ownership,
        shards,
        sidecar,
        digest(sidecar.read_bytes()),
        {dependency: digest(dependency.read_bytes())},
        resume_ledger=state,
    ) as resumed:
        assert resumed.next_cursor == 2
        resumed.read("t", cursor=2)
        final = resumed.ledger_state()
    assert final["next_cursor"] == 3
    assert [entry["cursor"] for entry in final["entries"]] == [0, 1, 2]  # type: ignore[index]


def test_cursor_skip_unknown_token_and_tampered_resume_fail_closed(tmp_path: Path) -> None:
    with reader(tmp_path) as candidate:
        with pytest.raises(ValueError, match="cursor must be exactly 0"):
            candidate.read("t", cursor=1)
        with pytest.raises(KeyError, match="not owned"):
            candidate.read("other", cursor=0)
        candidate.read("t", cursor=0)
        state = candidate.ledger_state()
    state["rolling_sha256"] = "0" * 64

    ownership = {"t": ("s.tar", "rgb/a.png", digest(b"selected-payload"))}
    shards = {"s.tar": digest((tmp_path / "s.tar").read_bytes())}
    sidecar = tmp_path / "index.json"
    dependency = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="rolling SHA"):
        ConsumedInputTarRGBReader(
            tmp_path,
            ownership,
            shards,
            sidecar,
            digest(sidecar.read_bytes()),
            {dependency: digest(dependency.read_bytes())},
            resume_ledger=state,
        )


def test_changed_dependency_and_sidecar_are_rejected(tmp_path: Path) -> None:
    ownership, shards, sidecar, sidecar_sha, dependency, dependency_sha = setup_contract(tmp_path)
    dependency.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Dependency SHA-256"):
        ConsumedInputTarRGBReader(
            tmp_path, ownership, shards, sidecar, sidecar_sha, {dependency: dependency_sha}
        )
    dependency.write_text('{"population":"fixed"}\n', encoding="utf-8")
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    with pytest.raises(ValueError, match="sidecar SHA-256"):
        ConsumedInputTarRGBReader(
            tmp_path, ownership, shards, sidecar, sidecar_sha, {dependency: dependency_sha}
        )


def test_resume_rejects_even_an_admitted_new_dependency_identity(tmp_path: Path) -> None:
    with reader(tmp_path) as candidate:
        candidate.read("t", cursor=0)
        state = candidate.ledger_state()
    dependency = tmp_path / "manifest.json"
    dependency.write_text("new but self-consistent dependency\n", encoding="utf-8")
    ownership = {"t": ("s.tar", "rgb/a.png", digest(b"selected-payload"))}
    shards = {"s.tar": digest((tmp_path / "s.tar").read_bytes())}
    sidecar = tmp_path / "index.json"
    with pytest.raises(ValueError, match="contract differs"):
        ConsumedInputTarRGBReader(
            tmp_path,
            ownership,
            shards,
            sidecar,
            digest(sidecar.read_bytes()),
            {dependency: digest(dependency.read_bytes())},
            resume_ledger=state,
        )
