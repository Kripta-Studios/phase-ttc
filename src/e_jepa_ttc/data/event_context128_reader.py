"""Verified bounded count readers and explicit, ancestor-safe TRAIN views."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.cache_components import verify_components
from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.data.event_context128 import (
    COUNT_SHAPE,
    ArchitectureInput,
    RateMoments,
    RateNormalization,
    four_contexts,
    log_rates,
)
from e_jepa_ttc.data.stage70_roles import TrainRoles


@dataclass(frozen=True)
class CountObservation:
    """Owned arrays: no surviving HDF5, ZIP or memory-map handle."""

    counts: np.ndarray
    durations_s: np.ndarray
    support: np.ndarray
    midpoint_twice_relative_us: np.ndarray


class CountReader:
    """Verify a completed cache once and copy only each requested count row."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        manifest = json.loads((self.root / "CACHE_MANIFEST.json").read_text())
        unsigned = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
        if (
            manifest.get("status") != "COMPLETE_COUNTS_ONLY"
            or object_digest(unsigned) != manifest.get("artifact_sha256")
            or object_digest(manifest["identity"]) != manifest.get("identity_sha256")
            or manifest["identity"].get("targets_present") is not False
        ):
            raise ValueError("count cache identity or completion contract mismatch")
        self.identity = str(manifest["artifact_sha256"])
        self.shards: list[dict[str, Any]] = manifest["shards"]
        self._row_locations: list[tuple[int, int]] = []
        self._stats: dict[str, tuple[int, int]] = {}
        for number, shard in enumerate(self.shards):
            names = {
                f"counts_{number:05d}.npy",
                f"observations_{number:05d}.npz",
                f"diagnostics_{number:05d}.json",
            }
            verify_components(self.root, shard["components"], names)
            rows = shard["rows"]
            if (
                not isinstance(rows, int)
                or not 1 <= rows <= 64
                or shard["first_row"] != len(self._row_locations)
                or shard["components"][f"counts_{number:05d}.npy"]["shape"] != [rows, *COUNT_SHAPE]
                or shard["components"][f"counts_{number:05d}.npy"]["dtype"]
                != manifest["identity"]["dtype"]
            ):
                raise ValueError("count shard schema or order mismatch")
            self._row_locations.extend((number, local) for local in range(rows))
            for name in names:
                stat = (self.root / name).stat()
                self._stats[name] = stat.st_size, stat.st_mtime_ns
        verify_components(self.root, {"metadata.csv": manifest["metadata"]}, {"metadata.csv"})
        self.metadata = pd.read_csv(self.root / "metadata.csv", dtype={"sample_token": str})
        if (
            list(self.metadata.columns) != ["sample_token", "sequence_id", "track_id", "outer_fold"]
            or len(self.metadata) != len(self._row_locations)
            or self.metadata.sample_token.duplicated().any()
            or self.metadata.isna().any().any()
            or manifest["identity"]["shape"] != [len(self.metadata), *COUNT_SHAPE]
            or manifest["identity"]["dtype"] not in ("uint16", "uint32")
        ):
            raise ValueError("count metadata does not bind unique aligned rows")
        self._indices = {str(token): i for i, token in enumerate(self.metadata.sample_token)}

    def _path(self, name: str) -> Path:
        path = self.root / name
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self._stats[name]:
            raise ValueError("verified cache component changed during use")
        return path

    def observation(self, token: str) -> CountObservation:
        """Read a requested row with explicit closure before returning owned arrays."""
        number, local = self._row_locations[self._indices[token]]
        mapped = np.load(self._path(f"counts_{number:05d}.npy"), mmap_mode="r", allow_pickle=False)
        try:
            counts = np.array(mapped[local], copy=True)
        finally:
            mapped._mmap.close()
        with np.load(self._path(f"observations_{number:05d}.npz"), allow_pickle=False) as archive:
            if set(archive.files) != {"durations_s", "support_bits", "midpoint_twice_relative_us"}:
                raise ValueError("unexpected observation fields")
            durations = archive["durations_s"][local].copy()
            times = archive["midpoint_twice_relative_us"][local].copy()
            bits = archive["support_bits"][local].copy()
        if bits.shape != (2048,) or bits.dtype != np.uint8:
            raise ValueError("invalid packed physical support")
        support = np.unpackbits(bits, bitorder="little").reshape(128, 128).astype(bool)
        return CountObservation(counts, durations, support, times)


class TrainingView:
    """A fixed token view whose normalization excludes every held-out ancestor."""

    def __init__(
        self,
        reader: CountReader,
        tokens: tuple[str, ...],
        roles: TrainRoles,
        *,
        excluded_sequences: tuple[str, ...],
        include_expansion: bool,
    ) -> None:
        if not tokens or len(set(tokens)) != len(tokens):
            raise ValueError("nonempty unique TRAIN token list required")
        metadata = reader.metadata.set_index("sample_token")
        if not set(tokens) <= set(metadata.index):
            raise ValueError("TRAIN token absent from verified count cache")
        sequences = tuple(sorted(set(metadata.loc[list(tokens), "sequence_id"].astype(str))))
        roles.check_fit_sequences(
            sequences,
            excluded_sequences=excluded_sequences,
            include_expansion=include_expansion,
        )
        self.reader = reader
        self.tokens = tokens
        self.identity = object_digest(
            {
                "cache": reader.identity,
                "ordered_tokens": tokens,
                "fit_sequences": sequences,
                "excluded_sequences": sorted(excluded_sequences),
                "roles": roles.__dict__,
                "include_expansion": include_expansion,
            }
        )

    def fit_normalization(self) -> RateNormalization:
        """Read only this TRAIN view; zero-activity observations remain included."""
        moments = RateMoments()
        for token in self.tokens:
            observation = self.reader.observation(token)
            moments.add(log_rates(observation.counts, observation.durations_s), observation.support)
        return moments.finish(self.identity)

    def inputs(self, token: str, normalization: RateNormalization) -> ArchitectureInput:
        """Materialize neural inputs only after validating the training-view binding."""
        if token not in self.tokens or normalization.training_view_sha256 != self.identity:
            raise ValueError("token or normalization belongs to another TRAIN view")
        row = self.reader.observation(token)
        return four_contexts(
            row.counts, row.durations_s, row.support, row.midpoint_twice_relative_us, normalization
        )
