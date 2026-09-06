"""Measured original-TRAIN raw preprocessing, transfer and standalone GPU inference cost."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.data.event_context128 import RateNormalization
from e_jepa_ttc.data.raw_deployment_v11 import RawDeploymentInputs
from e_jepa_ttc.evaluation.pretext_diagnostics_v11 import readonly_diagnostic
from e_jepa_ttc.evaluation.supervised_checkpoint_diagnostics_v11 import diagnostic_tokens
from e_jepa_ttc.models.projective_phase_network import Arm, ProjectivePhaseNetwork
from e_jepa_ttc.training.stage72_augmentation import time_derangement


def benchmark_raw_deployment(
    model: ProjectivePhaseNetwork,
    arm: Arm,
    seed: int,
    raw: RawDeploymentInputs,
    normalization: RateNormalization,
    *,
    check: Callable[[], None],
) -> dict[str, Any]:
    """Measure batch1 full128² raw-to-prediction cost on the same16 original TRAIN tokens.

    Four warmups are excluded, then two complete passes retain every timing. OS
    and HDF5 caches are warm; no precomputed representation replaces raw decoding.
    """
    original = set(raw.index.index.get_level_values("sample_token").astype(str))
    tokens = diagnostic_tokens(tuple(token for token in raw.view.tokens if token in original))
    if next(model.parameters()).device.type != "cuda":
        raise ValueError("deployment acceptance requires measured CUDA execution")
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(720076)
    rows = []
    checked = set()
    with readonly_diagnostic(model, generator):
        for index, token in enumerate((*tokens[:4], *tokens, *tokens)):
            check()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            sample = raw.inputs(token, normalization)
            prepared = time.perf_counter()
            contexts, times = sample.contexts[None].to(device), sample.times[None].to(device)
            permutation = (
                torch.tensor([time_derangement(token, seed)], device=device)
                if arm == "PPS-TIMEPERM"
                else None
            )
            torch.cuda.synchronize(device)
            transferred = time.perf_counter()
            output = model(contexts, times, arm=arm, permutation=permutation)
            torch.cuda.synchronize(device)
            finished = time.perf_counter()
            if not torch.isfinite(output["ttc"]).all():
                raise ValueError("deployment observation emitted a nonfinite endpoint prediction")
            if (
                torch.cuda.max_memory_allocated(device)
                >= 0.9 * torch.cuda.get_device_properties(device).total_memory
            ):
                raise RuntimeError("RESOURCE_BLOCKED: deployment benchmark reached90 percent VRAM")
            if token not in checked:
                expected = raw.view.inputs(token, normalization)
                if not all(
                    torch.equal(getattr(sample, field), getattr(expected, field))
                    for field in ("contexts", "times", "sensor_support")
                ):
                    raise ValueError("raw deployment differs from the proven cached representation")
                checked.add(token)
            rows.append(
                {
                    "token": token,
                    "warmup": index < 4,
                    "raw_preprocessing_ms": (prepared - started) * 1000,
                    "transfer_ms": (transferred - prepared) * 1000,
                    "network_ms": (finished - transferred) * 1000,
                    "end_to_end_ms": (finished - started) * 1000,
                }
            )
    summary = {}
    for field in ("raw_preprocessing_ms", "transfer_ms", "network_ms", "end_to_end_ms"):
        values = np.asarray([row[field] for row in rows if not row["warmup"]], np.float64)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("deployment timings must be positive measured durations")
        summary[field] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "observations": len(values),
        }
    total = sum(parameter.numel() for parameter in model.parameters())
    inactive = (
        sum(parameter.numel() for parameter in model.support.parameters()) if arm == "DIRECT" else 0
    )
    return {
        "kind": "MEASURED_STANDALONE_RAW_DEPLOYMENT_COST",
        "arm": arm,
        "seed": seed,
        "training_view_sha256": raw.view.identity,
        "raw_sources": raw.sources,
        "ordered_tokens": list(tokens),
        "raw_cache_representation_exact_rows": len(checked),
        "observations": rows,
        "summary": summary,
        "total_model_parameters": total,
        "active_model_parameters": total - inactive,
        "inactive_parameters": inactive,
        "parameter_tensor_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        ),
        "batch": 1,
        "roi": [128, 128],
        "precision": "float32_TF32_off",
        "raw_window_memo": "disabled per token; no cached input substitution",
        "filesystem_cache": (
            "warm OS/HDF5 caches; random fixed TRAIN token order, not sequential streaming"
        ),
        "parity_verification_included_in_timing": False,
        "optimizer_updates": 0,
        "deployed_experts": 1,
        "flops": None,
        "flops_note": "not estimated from parameter count",
    }
