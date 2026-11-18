"""PHASE_ONLY-family heads and the released cached-head optimizer, generalized to any input set.

The released recipe is reused verbatim where possible: ``fusion_loss`` (point + 0.1 quantile,
global mass), ``learning_rate`` (warmup + cosine), AdamW(3e-4, wd 1e-3), clip 1.0, batch 128
drawn uniformly with replacement from a seeded CPU generator, ``torch.manual_seed(seed)``
immediately before model construction.  Module names mirror ``PhaseOnlyFinal`` so the released
checkpoint loads strictly into ``ResidualPhaseHead(20)``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional

from e_jepa_ttc.rgb_port.fusion import fusion_loss
from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, emitted_phase, phase_to_ttc, pinball
from operational.final_refit.recipe import StageRecipe
from operational.final_refit.training import learning_rate

BATCH_SIZE = 128
WEIGHT_DECAY = 1e-3
ORIGINAL_H = 18_422
BASE_UPDATES = 2_500


def scaled_updates(population: int) -> int:
    """Released exposure scaling ``ceil(2500 * N / 18422)`` (12,044 for 88,744 rows)."""
    return math.ceil(BASE_UPDATES * population / ORIGINAL_H)


class ResidualPhaseHead(nn.Module):
    """``Linear(d,32)-LN-SiLU-Linear(32,32)-LN-SiLU-Linear(32,3)``, zero-init output."""

    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inputs, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        self.output = nn.Linear(32, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: Tensor, anchor: Tensor, width_base: Tensor) -> dict[str, Tensor]:
        correction = self.output(self.mlp(x))
        location = anchor + 0.03 * correction[:, 0]
        widths = 0.03 * functional.softplus(width_base + correction[:, 1:])
        return {
            "raw_location": location,
            "point_phase": emitted_phase(location),
            "ttc": phase_to_ttc(location),
            "q10": (location - widths[:, 0]).clamp(MIN_PHASE, MAX_PHASE),
            "q90": (location + widths[:, 1]).clamp(MIN_PHASE, MAX_PHASE),
            "correction": correction,
        }


def parameter_count(inputs: int) -> int:
    """Trainable parameters of ``ResidualPhaseHead(inputs)``."""
    return sum(p.numel() for p in ResidualPhaseHead(inputs).parameters())


@dataclass(frozen=True)
class FitResult:
    """Endpoint of one deterministic fit (no checkpoint selection)."""

    state_dict: dict[str, Tensor]
    updates: int
    final_loss_mean_last100: float
    seconds: float


def fit_head(
    x: NDArray[Any],
    anchor: NDArray[Any],
    width_base: NDArray[Any],
    target_phase: NDArray[Any],
    mass: NDArray[Any],
    *,
    seed: int,
    updates: int | None = None,
) -> FitResult:
    """Train one head to its fixed endpoint with the released optimizer and schedule."""
    population = len(target_phase)
    endpoint = scaled_updates(population) if updates is None else updates
    stage = StageRecipe(
        "PHASE_GEOMETRY", "phase_only_head", endpoint, BATCH_SIZE, WEIGHT_DECAY, (), True
    )
    torch.use_deterministic_algorithms(True)
    values = {
        "x": torch.as_tensor(np.asarray(x, dtype=np.float32)),
        "anchor": torch.as_tensor(np.asarray(anchor, dtype=np.float32)),
        "width": torch.as_tensor(np.asarray(width_base, dtype=np.float32)),
        "truth": torch.as_tensor(np.asarray(target_phase, dtype=np.float32)),
        "mass": torch.as_tensor(np.asarray(mass, dtype=np.float32)),
    }
    if not all(bool(torch.isfinite(v).all()) for v in values.values()):
        raise ValueError("Training tensors must be finite")
    torch.manual_seed(seed)
    model = ResidualPhaseHead(values["x"].shape[1]).float()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    started = time.perf_counter()
    for update in range(1, endpoint + 1):
        ids = torch.randint(population, (BATCH_SIZE,), generator=generator)
        optimizer.param_groups[0]["lr"] = learning_rate(stage, update)
        optimizer.zero_grad(set_to_none=True)
        output = model(values["x"][ids], values["anchor"][ids], values["width"][ids])
        loss = fusion_loss(output, values["truth"][ids], values["mass"][ids], population)
        if not torch.isfinite(loss):
            raise FloatingPointError("Head loss became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    return FitResult(
        state_dict={k: v.detach().clone() for k, v in model.state_dict().items()},
        updates=endpoint,
        final_loss_mean_last100=float(np.mean(losses[-100:])),
        seconds=time.perf_counter() - started,
    )


def predict_head(
    state_dict: dict[str, Tensor], x: NDArray[Any], anchor: NDArray[Any], width_base: NDArray[Any]
) -> dict[str, NDArray[np.float64]]:
    """Return raw location, emitted phase and TTC (float64 copies of float32 outputs)."""
    model = ResidualPhaseHead(int(np.asarray(x).shape[1])).float()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    with torch.inference_mode():
        out = model(
            torch.as_tensor(np.asarray(x, dtype=np.float32)),
            torch.as_tensor(np.asarray(anchor, dtype=np.float32)),
            torch.as_tensor(np.asarray(width_base, dtype=np.float32)),
        )
    return {
        "location": out["raw_location"].double().numpy(),
        "phase": out["point_phase"].double().numpy(),
        "ttc": out["ttc"].double().numpy(),
    }


@dataclass(frozen=True)
class StackMember:
    """One model of a stacked fit: its input columns, anchor column, width flag, mass, seed."""

    columns: NDArray[np.int64]
    anchor_column: int
    use_width: bool
    mass: NDArray[np.float32]
    seed: int


def _layer_norm(h: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    centered = h - h.mean(-1, keepdim=True)
    var = (centered * centered).mean(-1, keepdim=True)
    return centered / torch.sqrt(var + 1e-5) * weight[:, None, :] + bias[:, None, :]


def fit_heads_stacked(
    x_full: NDArray[Any],
    anchors: NDArray[Any],
    width_base: NDArray[Any],
    target_phase: NDArray[Any],
    members: list[StackMember],
    *,
    updates: int | None = None,
) -> list[FitResult]:
    """Train K independent heads at once; numerically equivalent to K ``fit_head`` calls.

    Each member keeps its own initialization (``torch.manual_seed(seed)`` then
    ``ResidualPhaseHead(d)``), its own seeded batch generator and its own gradient-norm clip;
    AdamW and the shared schedule act elementwise, so stacking only removes Python overhead.
    Unused (padded) input columns read a constant zero column and carry zero weights.
    """
    population = len(target_phase)
    endpoint = scaled_updates(population) if updates is None else updates
    stage = StageRecipe(
        "PHASE_GEOMETRY", "phase_only_head", endpoint, BATCH_SIZE, WEIGHT_DECAY, (), True
    )
    torch.use_deterministic_algorithms(True)
    k = len(members)
    width = max(len(m.columns) for m in members)
    x = torch.as_tensor(
        np.concatenate((np.asarray(x_full, np.float32), np.zeros((population, 1), np.float32)), 1)
    )
    zero_column = x.shape[1] - 1
    columns = torch.full((k, width), zero_column, dtype=torch.long)
    weights1 = torch.zeros((k, 32, width))
    tensors: dict[str, list[Tensor]] = {
        name: [] for name in ("b1", "g1", "e1", "w2", "b2", "g2", "e2", "w3", "b3")
    }
    for index, member in enumerate(members):
        d = len(member.columns)
        columns[index, :d] = torch.as_tensor(member.columns)
        torch.manual_seed(member.seed)
        model = ResidualPhaseHead(d)
        weights1[index, :, :d] = model.mlp[0].weight.detach()
        for name, value in (
            ("b1", model.mlp[0].bias),
            ("g1", model.mlp[1].weight),
            ("e1", model.mlp[1].bias),
            ("w2", model.mlp[3].weight),
            ("b2", model.mlp[3].bias),
            ("g2", model.mlp[4].weight),
            ("e2", model.mlp[4].bias),
            ("w3", model.output.weight),
            ("b3", model.output.bias),
        ):
            tensors[name].append(value.detach().clone())
    params = {"w1": weights1, **{n: torch.stack(v) for n, v in tensors.items()}}
    params = {n: nn.Parameter(v.float().contiguous()) for n, v in params.items()}
    optimizer = torch.optim.AdamW(
        list(params.values()),
        lr=3e-4,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )
    anchor_all = torch.as_tensor(np.asarray(anchors, np.float32))
    anchor_cols = torch.as_tensor([m.anchor_column for m in members], dtype=torch.long)
    width_all = torch.as_tensor(np.asarray(width_base, np.float32))
    width_flag = torch.as_tensor([1.0 if m.use_width else 0.0 for m in members])[:, None, None]
    truth_all = torch.as_tensor(np.asarray(target_phase, np.float32))
    mass_all = torch.as_tensor(np.stack([m.mass for m in members]).astype(np.float32))
    generators = [torch.Generator(device="cpu").manual_seed(m.seed) for m in members]
    model_index = torch.arange(k)[:, None]
    losses: list[Tensor] = []
    started = time.perf_counter()
    for update in range(1, endpoint + 1):
        ids = torch.stack(
            [torch.randint(population, (BATCH_SIZE,), generator=g) for g in generators]
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(stage, update)
        optimizer.zero_grad(set_to_none=True)
        xb = x[ids].gather(2, columns[:, None, :].expand(k, BATCH_SIZE, width))
        h = torch.einsum("kbi,koi->kbo", xb, params["w1"]) + params["b1"][:, None, :]
        h = functional.silu(_layer_norm(h, params["g1"], params["e1"]))
        h = torch.einsum("kbi,koi->kbo", h, params["w2"]) + params["b2"][:, None, :]
        h = functional.silu(_layer_norm(h, params["g2"], params["e2"]))
        correction = torch.einsum("kbi,koi->kbo", h, params["w3"]) + params["b3"][:, None, :]
        anchor = anchor_all[ids, anchor_cols[:, None]]
        location = anchor + 0.03 * correction[..., 0]
        widths = 0.03 * functional.softplus(width_all[ids] * width_flag + correction[..., 1:])
        truth = truth_all[ids]
        point = (emitted_phase(location) - truth).abs() / 0.03
        q10 = (location - widths[..., 0]).clamp(MIN_PHASE, MAX_PHASE)
        q90 = (location + widths[..., 1]).clamp(MIN_PHASE, MAX_PHASE)
        quantile = (pinball(q10, truth, 0.1) + pinball(q90, truth, 0.9)) / 0.03
        mass = mass_all[model_index, ids]
        per_model = (population * mass * (point + 0.1 * quantile)).mean(-1)
        if not torch.isfinite(per_model).all():
            raise FloatingPointError("Stacked head loss became nonfinite")
        per_model.sum().backward()
        norms = torch.sqrt(sum((p.grad.reshape(k, -1) ** 2).sum(1) for p in params.values()))
        if not torch.isfinite(norms).all():
            raise FloatingPointError("Stacked gradient norm became nonfinite")
        coefficient = (1.0 / (norms + 1e-6)).clamp(max=1.0)
        for p in params.values():
            p.grad.mul_(coefficient.reshape(k, *([1] * (p.ndim - 1))))
        optimizer.step()
        losses.append(per_model.detach())
    seconds = time.perf_counter() - started
    history = torch.stack(losses)
    results = []
    for index, member in enumerate(members):
        d = len(member.columns)
        state = {
            "mlp.0.weight": params["w1"][index, :, :d],
            "mlp.0.bias": params["b1"][index],
            "mlp.1.weight": params["g1"][index],
            "mlp.1.bias": params["e1"][index],
            "mlp.3.weight": params["w2"][index],
            "mlp.3.bias": params["b2"][index],
            "mlp.4.weight": params["g2"][index],
            "mlp.4.bias": params["e2"][index],
            "output.weight": params["w3"][index],
            "output.bias": params["b3"][index],
        }
        results.append(
            FitResult(
                state_dict={n: v.detach().clone().contiguous() for n, v in state.items()},
                updates=endpoint,
                final_loss_mean_last100=float(history[-100:, index].mean()),
                seconds=seconds / k,
            )
        )
    return results


__all__ = [
    "FitResult",
    "ResidualPhaseHead",
    "StackMember",
    "fit_head",
    "fit_heads_stacked",
    "parameter_count",
    "predict_head",
    "scaled_updates",
]
