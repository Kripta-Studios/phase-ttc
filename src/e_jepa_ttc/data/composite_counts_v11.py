"""Join original and expansion count caches without copying or renormalizing rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.data.event_context128_reader import CountObservation, CountReader
from e_jepa_ttc.data.stage70_roles import TrainRoles


def representation_contract(reader: CountReader) -> dict[str, Any]:
    """Bind exact producer bytes and representation semantics, excluding row/source identity."""
    manifest = json.loads((reader.root / "CACHE_MANIFEST.json").read_text())
    unsigned = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    if object_digest(unsigned) != reader.identity:
        raise ValueError("verified count manifest changed before composition")
    identity = manifest["identity"]
    required = (
        "dtype",
        "sensor_size",
        "channel_order",
        "bin_edges",
        "normalization",
        "targets_present",
        "sensor_support",
    )
    if any(key not in identity for key in required) or not identity.get("producer_code"):
        raise ValueError("composition requires a complete production representation contract")
    result = {key: identity[key] for key in required}
    result["row_shape"] = identity["shape"][1:]
    result["producer_code"] = sorted(
        (Path(record["path"]).name, record["sha256"]) for record in identity["producer_code"]
    )
    return result


class CompositeCountReader:
    """Route each unique token to its verified cache; TRAIN views still enforce fold exclusions."""

    def __init__(self, readers: tuple[CountReader, ...], roles: TrainRoles) -> None:
        if len(readers) < 2 or len({reader.identity for reader in readers}) != len(readers):
            raise ValueError("composition requires distinct verified caches")
        contract = representation_contract(readers[0])
        if any(representation_contract(reader) != contract for reader in readers[1:]):
            raise ValueError("count caches disagree on their immutable representation")
        self.metadata = pd.concat([reader.metadata.copy() for reader in readers], ignore_index=True)
        if self.metadata.sample_token.duplicated().any():
            raise ValueError("duplicate token across original/expansion caches")
        roles.check_fit_sequences(
            tuple(self.metadata.sequence_id.astype(str).unique()),
            excluded_sequences=(),
            include_expansion=True,
        )
        self._readers = {
            str(token): reader for reader in readers for token in reader.metadata.sample_token
        }
        self.identity = object_digest(
            {
                "component_caches": [reader.identity for reader in readers],
                "representation": contract,
                "roles": roles.__dict__,
                "ordered_tokens": self.metadata.sample_token.astype(str).tolist(),
                "operation": "lossless_union_no_normalization",
            }
        )

    def observation(self, token: str) -> CountObservation:
        """Read exactly one original or expansion row through its existing verified reader."""
        if token not in self._readers:
            raise ValueError("token absent from the fixed composite count sources")
        return self._readers[token].observation(token)
