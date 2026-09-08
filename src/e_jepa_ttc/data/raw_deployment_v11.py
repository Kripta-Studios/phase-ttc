"""Real raw-event deployment preprocessing using the proven cache rasterizer and clock."""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest, verify
from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE
from e_jepa_ttc.data.event_context128 import (
    ArchitectureInput,
    RateNormalization,
    Window128,
    checked_count_cast,
    four_contexts,
)
from e_jepa_ttc.data.event_context128_cache import WindowReader
from e_jepa_ttc.data.event_context128_reader import TrainingView
from e_jepa_ttc.data.raw_event_binding import ReadOnlyTrainAccess
from e_jepa_ttc.data.stage70_clock import context_clock


class RawDeploymentInputs:
    """Reprocess authorized TRAIN events without substituting cache lookup for raw latency."""

    def __init__(
        self,
        bindings: dict[str, Any],
        charter_binding: dict[str, Any],
        access: ReadOnlyTrainAccess,
        view: TrainingView,
    ) -> None:
        table = pd.read_csv(verify(bindings), dtype={"sample_token": str, "sequence_id": str})
        charter = json.loads(verify(charter_binding).read_text(encoding="utf-8"))
        if charter.get("status") == "EXPANSION_RAW_BOUND":
            identity = charter.get("identity", {})
            source_hashes = identity.get("sources", {})
            if (
                object_digest({k: v for k, v in charter.items() if k != "artifact_sha256"})
                != charter.get("artifact_sha256")
                or object_digest(identity) != charter.get("identity_sha256")
                or charter.get("binding") != bindings
                or charter.get("samples") != table.sample_token.nunique()
                or charter.get("source_groups") != len(source_hashes)
                or set(table.sequence_id) != set(source_hashes)
                or not set(view.tokens) <= set(table.sample_token)
            ):
                raise ValueError("expanded raw deployment manifest or TRAIN bindings changed")
        elif charter.get("status") == "ORIGINAL_8192_CLOCK_AND_ROI_VERIFIED":
            source_hashes = charter["raw_files"]
        else:
            raise ValueError("raw deployment requires the verified original clock charter")
        if table.duplicated(["sample_token", "window_id"]).any():
            raise ValueError("raw deployment bindings duplicate physical windows")
        self.index = table.set_index(["sample_token", "window_id"])
        self.view = view
        self.sequence_by_token = (
            view.reader.metadata.set_index("sample_token")["sequence_id"].astype(str).to_dict()
        )
        self.reader = WindowReader(access, source_hashes)
        self.sources = {"bindings": bindings, "clock_charter": charter_binding}

    def close(self) -> None:
        """Close all HDF5 datasets and memoized raw buffers before packaging or shutdown."""
        self.reader.close()

    def inputs(self, token: str, normalization: RateNormalization) -> ArchitectureInput:
        """Read both physical windows, rasterize full128² counts, then normalize in float32."""
        if (
            token not in self.view.tokens
            or normalization.training_view_sha256 != self.view.identity
        ):
            raise ValueError(
                "raw deployment benchmark token or normalizer is outside its TRAIN view"
            )
        pair = [cast(pd.Series, self.index.loc[(token, window)]) for window in (0, 1)]
        sequence = self.sequence_by_token[token]
        if any(str(row.sequence_id) != sequence for row in pair):
            raise ValueError("raw and cached TRAIN sequence identities differ")
        if any(
            str(row.h5_file_sha256) != self.reader.source_hashes[sequence]["sha256"] for row in pair
        ):
            raise ValueError("raw deployment window differs from its verified physical source hash")
        bounds = [(int(row.window_start_us), int(row.window_end_us)) for row in pair]
        frames = [int(row.window_end_us) + int(row.frame_to_event_clock_offset_us) for row in pair]
        clock = context_clock(bounds, frames, int(pair[1].target_anchor_us))
        rois = [
            tuple(float(cast(Any, row[f"roi_{axis}"])) for axis in ("x0", "y0", "x1", "y1"))
            for row in pair
        ]
        if rois[0] != rois[1] or any(float(row.event_x_offset_px) != 5.0 for row in pair):
            raise ValueError("raw deployment changed the common ROI or calibration offset")
        # Disable the object-window memo for measured raw preprocessing. OS/HDF5 caches
        # are explicitly reported as warm; they are not equivalent to precomputed inputs.
        self.reader.memo.clear()
        counts, durations, support = [], [], None
        for index, row in enumerate(pair):
            window = Window128(
                *bounds[index],
                cast(tuple[float, float, float, float], rois[index]),
                EAP_IMAGE_SIZE,
                polarity_encoding=cast(Any, str(row.polarity_encoding)),
            )
            observed, _ = self.reader.count(row, window)
            counts.append(checked_count_cast(observed, "uint16"))
            durations.append(np.diff(window.edges()) * 1e-6)
            support = window.support()
        assert support is not None
        return four_contexts(
            np.stack(counts),
            np.stack(durations),
            support,
            np.asarray(clock.midpoint_twice_relative_us, dtype=np.int64),
            normalization,
        )
