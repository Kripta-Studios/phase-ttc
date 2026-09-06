"""Nonselectable TRAIN/held-texture diagnostics for immutable correspondence checkpoints."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.models.projective_phase_network import ContextEncoder
from e_jepa_ttc.training.correspondence_pretext_v11 import PretextTextures
from e_jepa_ttc.training.eap_highres_jepa import _capture_rng_state, _restore_rng_state


@contextmanager
def readonly_diagnostic(encoder: nn.Module, generator: torch.Generator) -> Iterator[None]:
    """Preserve all training RNGs and module modes, and reject mutated model state."""
    state = _capture_rng_state(generator)
    modes = [(module, module.training) for module in encoder.modules()]
    tensors = {key: value.detach().clone() for key, value in encoder.state_dict().items()}
    try:
        encoder.eval()
        with torch.inference_mode():
            yield
        if any(not torch.equal(value, encoder.state_dict()[key]) for key, value in tensors.items()):
            raise ValueError("read-only diagnostic mutated the fitted encoder")
    finally:
        _restore_rng_state(state, generator)
        for module, mode in modes:
            module.training = mode


def summarize_texture_trace(path: Path, *, expected_images: int) -> dict[str, Any]:
    """Aggregate all valid image observations while retaining every rejected draw."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    result = {}
    for held in (False, True):
        selected = [row for row in records if row.get("diagnostic") is held]
        valid = [row for row in selected if row["kind"] == "valid_correspondence"]
        if len(valid) != expected_images:
            raise ValueError("pretext diagnostic omitted prescribed train or held images")
        fields = (
            "cross_entropy",
            "matching_error_pixels",
            "constant_descriptor_ce",
            "shuffled_correspondence_ce",
            "descriptor_variance",
            "masked_fraction",
        )
        means = {}
        for field in fields:
            values = np.array([row[field] for row in valid], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("pretext diagnostic emitted nonfinite measurements")
            means[field] = float(values.mean())
        queries = np.asarray([row["query_counts"] for row in valid])
        if queries.shape != (expected_images, 2) or (queries < 32).any():
            raise ValueError("diagnostic accepted an invalid correspondence query set")
        result["held_texture" if held else "train_texture"] = {
            **means,
            "valid_images": len(valid),
            "unique_tokens": len({row["token"] for row in valid}),
            "invalid_transform_attempts": sum(
                row["kind"] == "invalid_transform" for row in selected
            ),
            "minimum_direction_queries": int(queries.min()),
        }
    return result


def diagnose_pretext_checkpoint(
    checkpoint: dict[str, Any],
    pool: PretextTextures,
    *,
    seed: int,
    output: Path,
    check: Callable[[], None],
    device: str = "cuda",
) -> dict[str, Any]:
    """Observe 16 fixed TRAIN and16 fixed held-texture images, never choose a checkpoint.

    The checkpoint's own source identity must match the texture pool. The caller
    uses update0 and the registered final3000 in scientific jobs; technical QA may
    pass its explicitly marked short checkpoint without creating a candidate.
    """
    payload = torch.load(verify(checkpoint), map_location="cpu", weights_only=False)
    spec = payload["spec"]
    if (
        spec.get("arm") != "CORRESPONDENCE"
        or spec.get("training_view_sha256") != pool.identity
        or spec.get("seed") != seed
        or (spec.get("technical") is False and payload["update"] not in (0, 3000))
    ):
        raise ValueError("diagnostic checkpoint belongs to a different pretext role or seed")
    identity = {
        "checkpoint": checkpoint,
        "texture_pool_sha256": pool.identity,
        "seed": seed,
        "images_per_pool": 16,
        "diagnostic_seed": 730076 + seed,
        "update": payload["update"],
        "technical_checkpoint": spec["technical"],
        "selectable": False,
        "TTC_labels_used": False,
        "optimizer_updates": 0,
    }
    result_path = output / "PRETEXT_DIAGNOSTICS.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        unsigned = {key: value for key, value in result.items() if key != "artifact_sha256"}
        if (
            object_digest(unsigned) != result.get("artifact_sha256")
            or result.get("identity") != identity
        ):
            raise ValueError("pretext diagnostic receipt changed")
        verify(result["trace"])
        return result
    if output.exists() and any(output.iterdir()):
        raise ValueError("unfinished diagnostic trace requires a separate logged recovery attempt")
    output.mkdir(parents=True, exist_ok=True)
    trace = output / "TEXTURE_DIAGNOSTICS.jsonl"
    diagnostic_pool = PretextTextures(pool.view, pool.normalization, trace)
    generator = torch.Generator().manual_seed(730076 + seed)
    global_state = _capture_rng_state(generator)
    encoder = None
    try:
        encoder = ContextEncoder().to(device)
        encoder.load_state_dict(payload["model"], strict=True)
        with readonly_diagnostic(encoder, generator):
            for held in (False, True):
                generator.manual_seed(730076 + seed)
                for index in range(8):
                    check()
                    diagnostic_pool.objective(
                        encoder,
                        update=payload["update"],
                        microbatch_index=index,
                        batch=2,
                        generator=generator,
                        diagnostic=held,
                    )
        result = {
            "kind": "NONSELECTABLE_PRETEXT_CHECKPOINT_DIAGNOSTICS",
            "identity": identity,
            "summary": summarize_texture_trace(trace, expected_images=16),
            "trace": binding(trace),
            "last_training_gradient_norm": (
                payload["history"][-1]["gradient_norm"] if payload["history"] else None
            ),
            "held_normalization_scope": "full permissible TRAIN view; not independent confirmation",
        }
        result["artifact_sha256"] = object_digest(result)
        atomic_json(result_path, result)
        return result
    finally:
        _restore_rng_state(global_state, generator)
        del encoder, payload
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
