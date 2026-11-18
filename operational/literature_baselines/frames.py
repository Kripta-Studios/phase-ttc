"""eAP RGB frames: indexed tar-member reads and the shared square crop (numpy + Pillow)."""

from __future__ import annotations

import io
import math
import os
import tarfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .common import CROP_SIZE, Square

ENV_EAP_DATASET = "PHASE_TTC_EAP_DATASET"

UInt8Image = NDArray[np.uint8]


def eap_root() -> Path:
    """eAP public root (holding ``data/train/<sequence>/rgb_shards``) from the environment."""
    value = os.environ.get(ENV_EAP_DATASET, "").strip()
    if not value or not Path(value).is_dir():
        raise RuntimeError(
            f"Set {ENV_EAP_DATASET} to the eAP dataset root holding data/train/<sequence>/"
            "rgb_shards (see docs/sota_20261118/literature_baselines/REPORT.md)."
        )
    return Path(value)


def safe_shard(relative: str, sequence: str) -> str:
    """Accept only ``data/train/<sequence>/rgb_shards/<name>.tar`` without traversal."""
    parts = relative.replace("\\", "/").split("/")
    if (
        len(parts) != 5
        or parts[:4] != ["data", "train", sequence, "rgb_shards"]
        or not parts[4].endswith(".tar")
        or any(p in {"", ".", ".."} for p in parts)
    ):
        raise ValueError(f"Unexpected RGB shard path {relative!r} for {sequence}")
    return "/".join(parts)


class TarFrameReader:
    """Read PNG members of uncompressed tar shards by offset, with a small decoded-frame LRU."""

    def __init__(self, root: Path, cache_frames: int = 8) -> None:
        self.root = root
        self.cache_frames = cache_frames
        self._index: dict[str, dict[str, tuple[int, int]]] = {}
        self._frames: OrderedDict[tuple[str, str], UInt8Image] = OrderedDict()

    def _members(self, shard: str) -> dict[str, tuple[int, int]]:
        if shard not in self._index:
            table: dict[str, tuple[int, int]] = {}
            with tarfile.open(self.root / shard, mode="r:") as archive:
                for member in archive:
                    if member.isfile():
                        table[member.name] = (member.offset_data, member.size)
            self._index[shard] = table
        return self._index[shard]

    def read(self, shard: str, member: str) -> UInt8Image:
        """Decoded RGB frame ``[H, W, 3]`` uint8."""
        key = (shard, member)
        if key in self._frames:
            self._frames.move_to_end(key)
            return self._frames[key]
        if not member.startswith("rgb/") or ".." in member:
            raise ValueError(f"Unexpected RGB member {member!r}")
        offset, size = self._members(shard)[member]
        with (self.root / shard).open("rb") as stream:
            stream.seek(offset)
            payload = stream.read(size)
        with Image.open(io.BytesIO(payload)) as image:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
        self._frames[key] = frame
        while len(self._frames) > self.cache_frames:
            self._frames.popitem(last=False)
        return frame


def crop(image: UInt8Image, square: Square, size: int = CROP_SIZE) -> UInt8Image:
    """Resample ``square`` (source edge coordinates) to ``size``x``size``; outside pixels 0.

    Pillow BILINEAR with an explicit ``box`` (antialiased when reducing).
    """
    height, width = image.shape[:2]
    x0, y0, x1, y1 = square
    ix0, iy0 = math.floor(x0), math.floor(y0)
    ix1, iy1 = math.ceil(x1), math.ceil(y1)
    canvas = np.zeros((iy1 - iy0, ix1 - ix0, 3), dtype=np.uint8)
    sx0, sy0 = max(ix0, 0), max(iy0, 0)
    sx1, sy1 = min(ix1, width), min(iy1, height)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - iy0 : sy1 - iy0, sx0 - ix0 : sx1 - ix0] = image[sy0:sy1, sx0:sx1]
    resized = Image.fromarray(canvas).resize(
        (size, size), Image.Resampling.BILINEAR, box=(x0 - ix0, y0 - iy0, x1 - ix0, y1 - iy0)
    )
    return np.asarray(resized, dtype=np.uint8)
