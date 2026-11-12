"""Exact source reconstruction and bounded causal state for service comparisons."""

from __future__ import annotations

import hashlib
import shutil
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


def relocate_reference_sources(binding: dict[str, Any], original_root: Path) -> dict[str, Any]:
    """Relocate names only; the original inventory still verifies every expected hash."""
    sources = {}
    for name, digest in binding["sources"].items():
        path = Path(name)
        if path.is_absolute():
            if not path.resolve().is_relative_to(original_root.resolve()):
                raise ValueError("Frozen source outside original workspace")
            path = path.resolve().relative_to(original_root.resolve())
        if ".." in path.parts:
            raise ValueError("Frozen source outside original workspace")
        key = path.as_posix()
        if key in sources and sources[key] != digest:
            raise ValueError("Conflicting relocated source hashes")
        sources[key] = digest
    return {**binding, "sources": sources}


def materialize_runtime(
    root: Path, target: Path, archive: Path, binding: dict[str, Any]
) -> dict[str, Any]:
    """Restore hash-bound historical sources in a new isolated execution tree."""
    root, target = root.resolve(), target.resolve()
    if not target.is_relative_to(root / "output"):
        raise ValueError("Runtime target must be inside the workspace output directory")
    if target.exists():
        raise FileExistsError(target)
    restored = {}
    for relative, expected in binding["sources"].items():
        name = relative.replace("\\", "/")
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe archived source path")
        source = archive / (name.replace("/", "__") + ".txt")
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Archived source hash differs: {name}")
        restored[name] = data
    target.mkdir(parents=True)
    if (root / "configs").is_dir():
        shutil.copytree(root / "configs", target / "configs")
    for directory in ("src", "operational", "scripts"):
        for source in (root / directory).rglob("*.py"):
            destination = target / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    for name, data in restored.items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    hashes = {
        str(p.relative_to(target)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(target.rglob("*.py"))
    }
    return {
        "historical_execution_commit": binding.get("git_commit"),
        "restored_sources": len(restored),
        "restored_source_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in restored.items()
        },
        "tree_source_sha256": hashes,
        "configuration_sha256": {
            str(p.relative_to(target)).replace("\\", "/"): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted((target / "configs").rglob("*"))
            if p.is_file()
        },
        "remaining_sources": "Maintained workspace Python dependencies copied without modification",
    }


class EventRing:
    """Retain only causal packets; slices preserve the original event ordering."""

    def __init__(
        self, events: dict[str, np.ndarray], start: int, end: int, *, history_us: int = 600_000
    ) -> None:
        self.history_us = history_us
        self.start, self.end = start, end
        self.packets: deque[tuple[int, int, dict[str, np.ndarray]]] = deque()
        self._validate(events, start, end)
        self.packets.append((start, end, events))

    @staticmethod
    def _validate(events: dict[str, np.ndarray], start: int, end: int) -> None:
        t = events["t"]
        if end <= start or any(len(v) != len(t) for v in events.values()):
            raise ValueError("Invalid event packet")
        if len(t) and (t[0] < start or t[-1] >= end or np.any(t[1:] < t[:-1])):
            raise ValueError("Event packet exceeds causal interval or is not monotonic")

    def append(self, events: dict[str, np.ndarray], start: int, end: int) -> None:
        """Append one contiguous packet and expire past support."""
        if start != self.end:
            raise ValueError("Causal event history gap")
        self._validate(events, start, end)
        self.packets.append((start, end, events))
        self.end = end
        self.start = max(self.start, end - self.history_us)
        while self.packets and self.packets[0][1] <= self.start:
            self.packets.popleft()

    def window(self, start: int, end: int) -> dict[str, np.ndarray]:
        """Select a half-open window with no extrapolation or future access."""
        if end > self.end:
            raise ValueError("Requested future events")
        if start < self.start or end <= start:
            raise ValueError("Requested unavailable event history")
        selected: dict[str, list[np.ndarray]] = {k: [] for k in ("x", "y", "t", "p")}
        for left, right, packet in self.packets:
            if left >= end or right <= start:
                continue
            a, b = np.searchsorted(packet["t"], [start, end])
            for key in selected:
                selected[key].append(packet[key][a:b])
        dtypes = {"x": np.int32, "y": np.int32, "t": np.int64, "p": np.int8}
        return {
            key: np.concatenate(parts) if parts else np.empty(0, dtype=dtypes[key])
            for key, parts in selected.items()
        }
