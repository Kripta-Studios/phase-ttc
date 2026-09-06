"""Synthetic byte publications exercise chronological recovery without any optimizer work."""

import os
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.training.checkpoint_recovery_v11 import (
    existing_recovery,
    latest_valid_publication,
    preserve_corrupt_publications,
)


def publications(root):
    for update in (0, 5, 10):
        path = root / f"checkpoint_{update:06d}.pt"
        path.write_bytes(f"synthetic publication {update}".encode())
        receipt = {
            "identity": "a" * 64,
            "update": update,
            "payload": {**binding(path), "path": str(path.with_suffix(".pt.tmp").absolute())},
        }
        receipt["sha256"] = object_digest(receipt)
        atomic_json(root / f"PUBLISHED_CHECKPOINT_{update:06d}.json", receipt)


def test_latest_physical_state_is_selected_and_corrupt_bytes_are_preserved(tmp_path):
    publications(tmp_path)
    (tmp_path / "checkpoint_000010.pt").write_bytes(b"corruption fixture")
    recovery = latest_valid_publication(tmp_path, "a" * 64, 10)
    assert recovery["selected_update"] == 5
    source = preserve_corrupt_publications(tmp_path, recovery)
    assert verify(source).is_file()
    assert (
        Path(source["path"]).parent / "checkpoint_000010.pt"
    ).read_bytes() == b"corruption fixture"
    assert not (tmp_path / "checkpoint_000010.pt").exists()
    assert existing_recovery(tmp_path, "a" * 64, 10) == recovery
    assert preserve_corrupt_publications(tmp_path, recovery) == source


def test_preservation_interrupted_between_receipt_and_payload_move_is_resumable(
    tmp_path, monkeypatch
):
    publications(tmp_path)
    (tmp_path / "checkpoint_000010.pt").write_bytes(b"corruption fixture")
    recovery = latest_valid_publication(tmp_path, "a" * 64, 10)
    rename = Path.rename

    def interrupt(path, destination):
        if path.name == "checkpoint_000010.pt":
            raise OSError("injected interruption")
        return rename(path, destination)

    monkeypatch.setattr(Path, "rename", interrupt)
    with pytest.raises(OSError, match="injected"):
        preserve_corrupt_publications(tmp_path, recovery)
    monkeypatch.setattr(Path, "rename", rename)
    recovered = existing_recovery(tmp_path, "a" * 64, 10)
    assert recovered == recovery
    source = preserve_corrupt_publications(tmp_path, recovered)
    assert (Path(source["path"]).parent / "PRESERVATION_COMPLETE.json").is_file()


def test_no_valid_publication_blocks_instead_of_reseeding_or_inventing_a_state(tmp_path):
    publications(tmp_path)
    for update in (0, 5, 10):
        (tmp_path / f"checkpoint_{update:06d}.pt").write_bytes(b"bad")
    with pytest.raises(ValueError, match="no physically valid"):
        latest_valid_publication(tmp_path, "a" * 64, 10)


def test_linked_payload_is_rejected_before_hash_read(tmp_path, monkeypatch):
    publications(tmp_path)
    os.link(tmp_path / "checkpoint_000010.pt", tmp_path / "another_link.pt")
    import e_jepa_ttc.training.checkpoint_recovery_v11 as module

    monkeypatch.setattr(module, "verify", lambda record: pytest.fail("linked payload was read"))
    with pytest.raises(ValueError, match="redirected"):
        latest_valid_publication(tmp_path, "a" * 64, 10)
