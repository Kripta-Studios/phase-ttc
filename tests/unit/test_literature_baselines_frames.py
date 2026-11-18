"""Synthetic tests for the shared crop and the indexed tar-member reader."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from operational.literature_baselines.common import box_in_crop, common_square, region_mask
from operational.literature_baselines.frames import TarFrameReader, crop, safe_shard


def _scene(width: int = 320, height: int = 180) -> np.ndarray:
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    image[60:120, 100:160] = 220  # bright box, inclusive [100, 60, 159, 119]
    return image


def test_crop_places_the_box_where_the_mapping_says() -> None:
    """Pixels inside the mapped box are bright; the square scales the content uniformly."""
    box = [100, 60, 159, 119]
    square = common_square([box, box])
    patch = crop(_scene(), square, 96)
    mask = region_mask(box_in_crop(box, square, 96), 96, inner=True)
    assert patch.shape == (96, 96, 3)
    assert patch[mask].min() > 200
    assert patch[~region_mask(box_in_crop(box, square, 96), 96)].max() < 150


def test_crop_outside_the_image_is_black() -> None:
    """A square that leaves the frame is zero-padded, not clipped or reflected."""
    box = [0, 0, 39, 39]
    square = common_square([box, box])
    assert square[0] < 0
    patch = crop(_scene(), square, 64)
    assert (patch[:4, :4] == 0).all()
    assert patch[40:, 40:].min() == 40


def test_tar_reader_reads_members_by_offset(tmp_path: Path) -> None:
    """Members are decoded from their recorded offsets and cached."""
    shard = tmp_path / "data/train/SEQ/rgb_shards/rgb-00000.tar"
    shard.parent.mkdir(parents=True)
    with tarfile.open(shard, "w") as archive:
        for name, value in (("rgb/a.png", 10), ("rgb/b.png", 200)):
            buffer = io.BytesIO()
            Image.fromarray(np.full((8, 12, 3), value, dtype=np.uint8)).save(buffer, "PNG")
            info = tarfile.TarInfo(name)
            info.size = buffer.tell()
            buffer.seek(0)
            archive.addfile(info, buffer)
    reader = TarFrameReader(tmp_path, cache_frames=1)
    relative = safe_shard("data/train/SEQ/rgb_shards/rgb-00000.tar", "SEQ")
    assert reader.read(relative, "rgb/b.png").mean() == 200
    assert reader.read(relative, "rgb/a.png").shape == (8, 12, 3)
    with pytest.raises(ValueError):
        safe_shard("data/train/OTHER/rgb_shards/rgb-00000.tar", "SEQ")
    with pytest.raises(ValueError):
        reader.read(relative, "../x.png")
