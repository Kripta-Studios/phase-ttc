"""Bounded real-media replay through the sealed portable TRAIN40 graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from e_jepa_ttc.rgb_port.data import TarFrameReader
from operational.event_history_match.engine import configure_determinism, load_sealed
from operational.final_refit.test12 import FinalV16Train40Predictor
from operational.rgb_port import transfer
from operational.rgb_port_streaming.contracts import NativeObservation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    candidate: Path,
    inputs_path: Path,
    output: Path,
    device: str,
    sequence_id: str | None = None,
) -> dict[str, Any]:
    configure_determinism()
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    query = next(
        row
        for row in inputs["queries"]
        if row.get("rgb_history") and (sequence_id is None or row.get("sequence_id") == sequence_id)
    )
    models, binding, normalizers = load_sealed(candidate, device)
    native = FinalV16Train40Predictor.__new__(FinalV16Train40Predictor)
    native.device = torch.device(device)
    native.models = models
    native.normalizers = normalizers
    native.inputs = inputs
    native.reader = TarFrameReader(Path(inputs["media_root"]), max_open=2)
    native.event_values = {}
    native.rgb_values = {}
    captured: dict[str, np.ndarray] = {}
    handles = []

    def hook(
        name: str,
    ) -> Callable[[torch.nn.Module, tuple[Any, ...], dict[str, torch.Tensor]], None]:
        def save(
            _module: torch.nn.Module,
            args: tuple[Any, ...],
            result: dict[str, torch.Tensor],
        ) -> None:
            for index, value in enumerate(args):
                if isinstance(value, torch.Tensor):
                    captured[f"{name}.input_{index}"] = value.detach().cpu().numpy()
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    captured[f"{name}.output_{key}"] = value.detach().cpu().numpy()

        return save

    for name in ("E_CTX_MATCHED", "R_CTX", "PHASE_ONLY_EVENT_S7_TRAIN40"):
        handles.append(models[name].register_forward_hook(hook(name)))
    started = time.perf_counter()
    try:
        # Frozen V12/V13 ABI boundary used by the released Test12 runner.  It
        # preserves the aggregate current-query known mask and pair arithmetic.
        with (
            patch.object(transfer, "ProducerObservation", NativeObservation),
            torch.inference_mode(),
        ):
            prediction = native.predict(query, inputs)
        if device == "cuda":
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
        native.close()
    elapsed = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "captured_endpoints.npz", **captured)
    result = {
        "schema": "phase_geometry_real_media_pilot_v1",
        "sample_token": query["sample_token"],
        "sequence_id": query.get("sequence_id"),
        "prediction_ttc_seconds": prediction,
        "elapsed_seconds_including_media_decode": elapsed,
        "event_history_count": len(query["event_history"]),
        "rgb_history_count": len(query["rgb_history"]),
        "captured_tensors": {key: list(value.shape) for key, value in sorted(captured.items())},
        "inputs_path": str(inputs_path.resolve()),
        "inputs_sha256": sha256(inputs_path),
        "candidate_path": str(candidate.resolve()),
        "binding": binding,
        "device": device,
        "media_regenerated": {"rgb_from_tar": True, "event_from_hdf5": False},
        "event_source": "authenticated frozen public event tensor",
        "producer_observation_boundary": "NativeObservation (released Test12 ABI adapter)",
        "private_targets_read": False,
    }
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--sequence-id")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.candidate, args.inputs, args.output, args.device, args.sequence_id), indent=2
        )
    )


if __name__ == "__main__":
    main()
