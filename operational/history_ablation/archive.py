"""In-memory access to the archived rotation inputs of the released context refiners.

Nothing is extracted to disk: each tar member is streamed into memory, hashed, and
checked against a manifest derived from the pod-return ``RESTORE_INDEX.json``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ARCHIVE_ROOT = Path(os.environ.get("HISTORY_ABLATION_ARCHIVE", r"D:\E_JEPA_POD2_RETURN_20261027"))
ROWS_ROOT = Path(os.environ.get("HISTORY_ABLATION_ROWS", r"E:\E_JEPA_TTC_RGB_PORT\metadata"))
ROTATIONS = (0, 1, 2, 3, 4)
MODALITIES = ("event", "rgb")
STAGE = {"event": "E_CTX", "rgb": "R_CTX"}
ITEMS = ("cache", "normalizer", "generation", "checkpoint")

_FINAL_INPUTS = "final_inputs_63e2720817ba6521.tar"
_FINAL_INPUTS_ROOT = (
    "workspace/ejepa_final_inputs/workspace/e-jepa-ttc-v18-joint-measurement/"
    "artifacts/final_refit_20261019_scientific_oof/rotations/{r}/"
)
_EVENT_TARS = {
    2: (
        "event_r2_support_07c558763df2a6d8.tar",
        "event_r2_generation_488311a8d5055ba2.tar",
        "event_r2_fits_7d297b34031f11de.tar",
    ),
    3: (
        "event_r3_support_c91a79b20191bcf6.tar",
        "event_r3_generation_9194983c2d6eba9c.tar",
        "event_r3_fits_65d25f917edd74ea.tar",
    ),
    4: (
        "event_r4_support_2b253f882089a962.tar",
        "event_r4_generation_47fa32fadc171e5d.tar",
        "event_r4_fits_eb7b3010e157f032.tar",
    ),
}
_RGB_CLOSURE = "rgb_fold_closure_06486d50af021f74.tar"
_FINAL_GENERATED = "final_generated_fd84565b0df06d79.tar"
_STITCH_ROOT = "workspace/ejepa_final_runs/final/oof/"


@dataclass(frozen=True)
class Member:
    """One archived file: tar name and member path inside it."""

    archive: str
    member: str


def _relative(item: str, stage: str) -> str:
    return {
        "cache": f"caches/{stage}.npz",
        "normalizer": f"normalizers/{stage}.json",
        "generation": f"generation/{stage}/fragment.npz",
        "checkpoint": f"fits/{stage}/checkpoint_last.pt",
    }[item]


def rotation_member(rotation: int, modality: str, item: str) -> Member:
    """Locate one rotation input of a context refiner inside the pod-return archive."""
    if rotation not in ROTATIONS or modality not in MODALITIES or item not in ITEMS:
        raise ValueError(f"Unknown rotation input: {rotation}, {modality}, {item}")
    stage = STAGE[modality]
    relative = _relative(item, stage)
    if rotation in (0, 1):
        return Member(_FINAL_INPUTS, _FINAL_INPUTS_ROOT.format(r=rotation) + relative)
    if modality == "event":
        support, generation, fits = _EVENT_TARS[rotation]
        archive = {
            "cache": support,
            "normalizer": support,
            "generation": generation,
            "checkpoint": fits,
        }[item]
        return Member(archive, f"workspace/ejepa_event_runs/rotations/{rotation}/{relative}")
    root = f"workspace/ejepa_rgb_handoff/extracted_v10/roots/r{rotation}_downstream/rotations/"
    return Member(_RGB_CLOSURE, f"{root}{rotation}/{relative}")


def stitch_member(interface: str) -> Member:
    """Locate a released stitched OOF file (e.g. ``PHASE20_CONTEXT``)."""
    return Member(_FINAL_GENERATED, f"{_STITCH_ROOT}{interface}.npz")


def all_members() -> dict[str, Member]:
    """Every archived input this ablation reads, keyed by a stable label."""
    result = {
        f"r{r}/{m}/{i}": rotation_member(r, m, i)
        for r in ROTATIONS
        for m in MODALITIES
        for i in ITEMS
    }
    for interface in ("PHASE20_CONTEXT", "PHASE17_RAW_EVENT", "PHASE17_RAW_RGB"):
        result[f"stitch/{interface}"] = stitch_member(interface)
    return result


class ArchiveReader:
    """Read and hash members from a set of tar files, opening each tar once."""

    def __init__(
        self, root: Path = ARCHIVE_ROOT, manifest: Mapping[str, Any] | None = None
    ) -> None:
        self.root = root
        self.manifest = manifest
        self._tars: dict[str, tarfile.TarFile] = {}

    def _tar(self, name: str) -> tarfile.TarFile:
        if name not in self._tars:
            self._tars[name] = tarfile.open(self.root / name, mode="r")
        return self._tars[name]

    def read_bytes(self, member: Member) -> bytes:
        """Return member bytes; verify SHA-256 against the manifest when one is bound."""
        stream = self._tar(member.archive).extractfile(member.member)
        if stream is None:
            raise FileNotFoundError(f"{member.archive}:{member.member}")
        payload = stream.read()
        if self.manifest is not None:
            expected = self.manifest.get(member.member)
            if expected is None:
                raise ValueError(f"Member not bound by the input manifest: {member.member}")
            if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError(f"Archived member changed: {member.member}")
        return payload

    def arrays(self, member: Member) -> dict[str, np.ndarray]:
        """Load a non-pickled NPZ member fully into memory."""
        with np.load(io.BytesIO(self.read_bytes(member)), allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def json(self, member: Member) -> dict[str, Any]:
        """Load a JSON member."""
        value = json.loads(self.read_bytes(member).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {member.member}")
        return value

    def close(self) -> None:
        """Close every opened tar."""
        for handle in self._tars.values():
            handle.close()
        self._tars.clear()


def _iter_index_members(index_path: Path, wanted: set[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    with index_path.open(encoding="utf-8") as stream:
        members = json.load(stream)["members"]
    for name in wanted:
        if name not in members:
            raise KeyError(f"RESTORE_INDEX lacks {name}")
        yield name, members[name]


def build_manifest(root: Path = ARCHIVE_ROOT) -> dict[str, Any]:
    """Bind every needed member to the archive's own restore index, then rehash it."""
    members = all_members()
    wanted = {member.member for member in members.values()}
    expected = dict(_iter_index_members(root / "RESTORE_INDEX.json", wanted))
    reader = ArchiveReader(root)
    bound: dict[str, Any] = {}
    try:
        for label, member in sorted(members.items()):
            entry = expected[member.member]
            if Path(str(entry["archive"])).name != member.archive:
                raise ValueError(f"RESTORE_INDEX places {member.member} in another archive")
            digest = hashlib.sha256(reader.read_bytes(member)).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(f"Archived bytes differ from RESTORE_INDEX: {member.member}")
            bound[member.member] = {
                "label": label,
                "archive": member.archive,
                "bytes": int(entry["bytes"]),
                "sha256": digest,
            }
    finally:
        reader.close()
    return {
        "schema": "history_ablation_input_manifest_v1",
        "archive_root": str(root),
        "restore_index_sha256": _file_sha256(root / "RESTORE_INDEX.json"),
        "members": bound,
    }


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def supervision(rows_root: Path = ROWS_ROOT) -> pd.DataFrame:
    """TRAIN40 targets and group ids by sample token (development labels only)."""
    columns = [
        "source_ordinal",
        "sample_token",
        "group_id",
        "sequence_id",
        "target_ttc",
        "target_phase",
        "actual_span_us",
        "producer_frame_count",
    ]
    frame = pd.concat(
        [pd.read_parquet(rows_root / f"{role}_ROWS.parquet", columns=columns) for role in "PHV"],
        ignore_index=True,
    )
    frame = frame.sort_values("source_ordinal", kind="stable").reset_index(drop=True)
    if (
        len(frame) != 88_744
        or frame.sample_token.duplicated().any()
        or frame.group_id.nunique() != 40
        or not (frame.group_id == frame.sequence_id).all()
    ):
        raise ValueError("TRAIN40 supervision rows are not the admitted 88,744-query union")
    phase = -np.log1p(-0.1 / frame.target_ttc.to_numpy(np.float64))
    if np.max(np.abs(phase - frame.target_phase.to_numpy(np.float64))) > 1e-6:
        raise ValueError("target_phase is inconsistent with target_ttc")
    return frame


def rows_sha256(rows_root: Path = ROWS_ROOT) -> dict[str, str]:
    """Hash the supervision row files for provenance."""
    return {role: _file_sha256(rows_root / f"{role}_ROWS.parquet") for role in "PHV"}


__all__ = [
    "ARCHIVE_ROOT",
    "MODALITIES",
    "ROTATIONS",
    "ArchiveReader",
    "Member",
    "all_members",
    "build_manifest",
    "rotation_member",
    "rows_sha256",
    "stitch_member",
    "supervision",
]
