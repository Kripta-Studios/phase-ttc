"""Faithful re-fits of the released context refiners and PHASE_ONLY fusion head.

The optimizer loop mirrors ``operational.final_refit.training.fit_cached_stage``
(same seeding order, model factory, loss, learning-rate schedule, AdamW settings,
gradient clipping and ``torch.randint`` sampler) without its receipt and checkpoint
machinery. The released modules are imported, not copied or edited.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from operational.final_refit.features import frozen_context_state
from operational.final_refit.recipe import StageRecipe
from operational.final_refit.scientific_oof_features import (
    normalize_histories,
    physical_phase20_to_state340,
)
from operational.final_refit.training import (
    PhaseOnlyFinal,
    _balanced_mass,
    _loss,
    _model,
    learning_rate,
)

BATCH = 128
WEIGHT_DECAY = 1e-3
BASE_UPDATES = 2_500
BASE_POPULATION = 18_422


def scaled_updates(population: int) -> int:
    """Released upper-stage exposure rule ``ceil(2500 * N / 18422)``."""
    return math.ceil(BASE_UPDATES * population / BASE_POPULATION)


def context_stage(name: str, updates: int) -> StageRecipe:
    """Stage recipe of a released context refiner with a given endpoint."""
    return StageRecipe(
        name,
        "temporal_head",
        updates,
        BATCH,
        WEIGHT_DECAY,
        (),
        True,
        "final_refit_context_cache_v1",
    )


def fusion_stage(updates: int) -> StageRecipe:
    """Stage recipe of the released PHASE_ONLY fusion head with a given endpoint."""
    return StageRecipe(
        "PHASE_ONLY_HISTORY_ABLATION",
        "phase_only_head",
        updates,
        BATCH,
        WEIGHT_DECAY,
        (),
        True,
        "final_refit_phase_cache_v1",
    )


class ArrayCache:
    """Duck-typed stand-in for ``FinalCache`` holding resident tensors."""

    def __init__(self, values: Mapping[str, np.ndarray], device: torch.device) -> None:
        groups = np.asarray(values["group_id"]).astype(str)
        self.groups = int(len(set(groups.tolist())))
        self.mass = _balanced_mass(groups, self.groups)
        self._values: dict[str, Tensor] = {
            name: torch.from_numpy(np.ascontiguousarray(value)).to(device)
            for name, value in values.items()
            if np.asarray(value).dtype.kind in {"b", "f", "i", "u"}
        }
        self._values["mass"] = torch.from_numpy(self.mass).to(device)
        self.population = int(len(groups))

    def tensor(self, name: str, ids: Tensor, device: torch.device) -> Tensor:
        """Select rows exactly as ``FinalCache.tensor`` does for a resident cache."""
        return self._values[name].index_select(0, ids.to(device))


@dataclass
class FitResult:
    """A trained endpoint and its loss trace."""

    model: nn.Module
    losses: list[float]
    updates: int


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def fit_stage(
    stage: StageRecipe, values: Mapping[str, np.ndarray], *, seed: int, device: torch.device
) -> FitResult:
    """Train one cached stage to its fixed endpoint (no selection, no early stopping)."""
    _seed_everything(seed, device)
    cache = ArrayCache(values, device)
    model = _model(stage).float().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=stage.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    for update in range(1, stage.updates + 1):
        ids = torch.randint(cache.population, (stage.batch_size,), generator=generator)
        optimizer.param_groups[0]["lr"] = learning_rate(stage, update)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, stage, cache, ids, device)  # type: ignore[arg-type]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{stage.stage_id} loss became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval().requires_grad_(False)
    return FitResult(model, losses, stage.updates)


def context_outputs(
    model: nn.Module, values: Mapping[str, np.ndarray], *, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Released 10 physical coordinates ``block[:,160:170]`` and 2 width logits."""
    states, widths = [], []
    for start in range(0, len(values["valid"]), BATCH):
        part = slice(start, start + BATCH)
        block, width, _ = frozen_context_state(
            model,
            *[
                torch.from_numpy(np.ascontiguousarray(values[key][part])).to(device)
                for key in ("features", "timing", "valid", "expert_phase")
            ],
        )
        states.append(block[:, 160:170].cpu().numpy())
        widths.append(width.cpu().numpy())
    return np.concatenate(states).astype(np.float32), np.concatenate(widths).astype(np.float32)


def normalize_fragment(
    fragment: Mapping[str, np.ndarray], normalizer: Mapping[str, object]
) -> dict[str, np.ndarray]:
    """Normalize a raw generation fragment with its rotation's label-free normalizer."""
    mean = np.asarray(normalizer["mean"], np.float32)
    scale = np.asarray(normalizer["scale"], np.float32)
    return {
        "features": normalize_histories(fragment, mean, scale),
        "timing": np.asarray(fragment["timing"], np.float32),
        "valid": np.asarray(fragment["valid"], np.bool_),
        "expert_phase": np.asarray(fragment["expert_phase"], np.float32),
    }


def fusion_inputs(rgb10: np.ndarray, event10: np.ndarray) -> np.ndarray:
    """Released PHASE20 order (RGB 10 then event 10) embedded into the 340-D state."""
    return physical_phase20_to_state340(
        np.ascontiguousarray(np.concatenate((rgb10, event10), axis=1), dtype=np.float32)
    )


def fusion_predict(
    model: nn.Module, features340: np.ndarray, widths: np.ndarray, *, device: torch.device
) -> np.ndarray:
    """Raw fused phase location of the PHASE_ONLY head."""
    if not isinstance(model, PhaseOnlyFinal):
        raise TypeError("Expected the released PHASE_ONLY architecture")
    out = []
    with torch.inference_mode():
        for start in range(0, len(features340), 4096):
            part = slice(start, start + 4096)
            result = model(
                torch.from_numpy(features340[part]).to(device),
                torch.from_numpy(widths[part]).to(device),
            )
            out.append(result["raw_location"].cpu().numpy())
    return np.concatenate(out).astype(np.float32)


__all__ = [
    "ArrayCache",
    "FitResult",
    "context_outputs",
    "context_stage",
    "fit_stage",
    "fusion_inputs",
    "fusion_predict",
    "fusion_stage",
    "normalize_fragment",
    "scaled_updates",
]
