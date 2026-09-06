"""Metadata inventory cannot follow injected references outside a TRAIN sequence."""

from pathlib import Path

import pytest

from scripts.audit_stage70_train_foreground import train_mask_candidate


@pytest.mark.parametrize(
    "reference",
    [
        "../../protected/mask.npy",
        "C:/outside/mask.npy",
        "C:mask.npy",
        "//server/share/mask.npy",
        "image_blobs/a.npy:stream",
        "image_blobs/a.pkl",
    ],
)
def test_foreground_path_injection_rejected_before_filesystem_lookup(tmp_path, reference):
    with pytest.raises(ValueError, match="permitted TRAIN"):
        train_mask_candidate(tmp_path, "train_seq", reference)


def test_declared_relative_train_path_can_be_inventoried_without_payload_read(tmp_path):
    path = train_mask_candidate(tmp_path, "train_seq", "image_blobs/a.npy")
    assert path == tmp_path / "train_seq" / "image_blobs" / "a.npy"
    assert not path.exists()
    with pytest.raises(ValueError, match="permitted TRAIN"):
        train_mask_candidate(tmp_path, "../outer", "image_blobs/a.npy")
    assert isinstance(path, Path)
