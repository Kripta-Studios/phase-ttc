"""Sealed event-branch replay under an explicit history plan.

Everything model-related is imported from the existing delivered code paths, never copied:
``encode_union`` (training/inference encoder), ``operational.rgb_port.transfer._expert_block``
with ``NativeObservation`` (exactly as ``FinalTransferPredictor``), the sealed normalizer,
``frozen_context_state`` for E_CTX and the released ``PHASE_ONLY_EVENT_S7_TRAIN40`` head.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from unittest.mock import patch

import numpy as np
import torch
from torch import Tensor, nn

from .history import SLOTS, SlotPlan, plan_timing

HEAD = "PHASE_ONLY_EVENT_S7_TRAIN40"


class WindowReader(Protocol):
    """Reader contract consumed by ``encode_union``."""

    def iter_window_chunks(
        self, start_us: int, end_us: int, *, chunk_events: int = 250_000
    ) -> Iterator[dict[str, np.ndarray]]: ...


@dataclass(frozen=True)
class EventBranchOutput:
    """Frozen-state event block and readouts for one query under one plan."""

    block: np.ndarray  # [170] E_CTX frozen state (fusion coordinates 170:340)
    width_logits: np.ndarray  # [2]
    ctx_ttc: float
    ctx_phase: float
    count: int


def encode_slots(
    reader: WindowReader,
    plan: SlotPlan,
    windows_us: object,
    square_xyxy: Sequence[float],
    *,
    sequence_id: str,
    event_pixel_diff: float,
    retained_bytes_max: int = 1024 * 1024**2,
) -> Tensor:
    """Encode every valid slot of ``plan`` inside ONE crop square (the current query's)."""
    from e_jepa_ttc.efficient_context.mapped_union import encode_union

    windows = np.asarray(windows_us, dtype=np.int64)
    tensor = encode_union(
        reader,  # pyright: ignore[reportArgumentType]
        windows,
        plan.storage_lags(),
        plan.storage_valid(),
        tuple(float(value) for value in square_xyxy),
        sequence_id=sequence_id,
        roi_size=128,
        event_pixel_diff=event_pixel_diff,
        retained_bytes_max=retained_bytes_max,
    )
    return tensor[-SLOTS:]


def observation_rows(
    models: Mapping[str, nn.Module], sensor: Tensor, delta_s: float, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Per-observation PHASE17 rows and expert phases via the delivered producer path."""
    from operational.rgb_port import transfer as engine
    from operational.rgb_port_streaming.contracts import NativeObservation

    if sensor.ndim != 5 or sensor.shape[1:] != (3, 12, 128, 128):
        raise ValueError("event observations must be [N,3,12,128,128]")
    delta = torch.full((len(sensor), 2), float(delta_s), dtype=torch.float32, device=device)
    with torch.inference_mode(), patch.object(engine, "ProducerObservation", NativeObservation):
        raw, experts = engine._expert_block(
            dict(models), "E", sensor.to(device=device, dtype=torch.float32), delta
        )
    return raw, experts


def event_context(
    models: Mapping[str, nn.Module],
    normalizer: tuple[Tensor, Tensor],
    plan: SlotPlan,
    raw: Tensor,
    experts: Tensor,
    device: torch.device,
) -> EventBranchOutput:
    """Place valid rows into the padded 8-slot context and run sealed E_CTX."""
    from e_jepa_ttc.simplex_t.phase import phase_to_ttc
    from operational.final_refit.features import frozen_context_state

    count = plan.count
    if raw.shape != (count, 17) or experts.shape != (count, 3):
        raise ValueError("observation rows must match the plan's valid slots")
    columns = torch.from_numpy(np.flatnonzero(plan.valid)).to(device)
    mean, scale = normalizer
    features = torch.zeros((1, SLOTS, 17), dtype=torch.float32, device=device)
    features[0, columns] = (raw - mean) / scale
    timing = torch.from_numpy(plan_timing(plan))[None].to(device)
    valid = torch.from_numpy(plan.valid.copy())[None].to(device)
    with torch.inference_mode():
        block, width, output = frozen_context_state(
            models["E_CTX_MATCHED"], features, timing, valid, experts[-1:].clone()
        )
        phase = output["point_phase"]
        ttc = phase_to_ttc(phase)
    return EventBranchOutput(
        block=block[0].cpu().numpy().astype(np.float32),
        width_logits=width[0].cpu().numpy().astype(np.float32),
        ctx_ttc=float(ttc[0].cpu()),
        ctx_phase=float(phase[0].cpu()),
        count=count,
    )


def fusion_ttc(
    head: nn.Module, rgb_block: np.ndarray, event_block: np.ndarray, widths: np.ndarray
) -> np.ndarray:
    """Released PHASE_ONLY head on [R170, E170] with frozen R_CTX width logits (CPU)."""
    rgb = np.atleast_2d(np.asarray(rgb_block, np.float32))
    event = np.atleast_2d(np.asarray(event_block, np.float32))
    if rgb.shape[1] != 170 or event.shape != rgb.shape:
        raise ValueError("fusion requires aligned [N,170] RGB and event blocks")
    state = torch.from_numpy(np.concatenate((rgb, event), axis=1))
    width = torch.from_numpy(np.atleast_2d(np.asarray(widths, np.float32)))
    device = next(head.parameters()).device
    with torch.inference_mode():
        output = head(state.to(device), width.to(device))
    return output["ttc"].double().cpu().numpy()


def load_sealed(candidate: Path, device: str) -> tuple[dict[str, nn.Module], dict[str, Any], Any]:
    """Verify all sealed bytes, then strict-load the nine components (FP32, eval)."""
    from scripts.final_train40_transfer import load_models, verify_candidate

    paths, binding = verify_candidate(candidate)
    models, normalizers = load_models(paths, device)
    return models, binding, normalizers


def configure_determinism() -> None:
    """Same numerical settings as the delivered transfer runs (FP32, TF32 disabled)."""
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(7)
