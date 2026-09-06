"""Executable standalone reference architecture; no frozen A5 inputs.

Production integration, data loading, memory profiling and GPU QA remain Codex
work. A small-image forward is tested here; the scientific representation is 128².
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .phase_distribution_v11 import phase_edges, phase_to_finite_ttc, phase_to_inverse_ttc, quantile
from .projective_geometry_v11 import scale_between
from .projective_phase_sweep import marginalize_translation, sweep_scores, translations

Arm = Literal["DIRECT", "PPS-PAIR", "PPS-TRAJ", "PPS-TIMEPERM"]
Capacity = Literal["2M", "4M", "8M"]

# Prospective width scaling, rounded to GroupNorm-compatible multiples of eight.
# Geometry, representation, evidence refiner and the108-bin output stay fixed.
CAPACITY_WIDTHS: dict[Capacity, tuple[int, int, int, int, int, int]] = {
    "2M": (32, 64, 96, 128, 160, 256),
    "4M": (48, 88, 136, 184, 224, 360),
    "8M": (64, 128, 192, 256, 320, 512),
}


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return functional.silu(x + self.block(x))


class ContextEncoder(nn.Module):
    def __init__(self, capacity: Capacity = "2M") -> None:
        super().__init__()
        if capacity not in CAPACITY_WIDTHS:
            raise ValueError("unregistered model capacity")
        c0, c1, c2, _, _, _ = CAPACITY_WIDTHS[capacity]
        self.capacity = capacity
        self.net = nn.Sequential(
            nn.Conv2d(8, c0, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, c0),
            nn.SiLU(),
            ResidualBlock(c0),
            ResidualBlock(c0),
            nn.Conv2d(c0, c1, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            ResidualBlock(c1),
            ResidualBlock(c1),
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.GroupNorm(8, c2),
            nn.SiLU(),
            ResidualBlock(c2),
            ResidualBlock(c2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class QuadrantAveragePool(nn.Module):
    """Adaptive 2x2 averages with deterministic CUDA reduction backward.

    Floor starts and ceiling ends preserve the original adaptive-pool regions,
    including overlapping regions when a spatial dimension is odd.
    """

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        rows = []
        for iy in range(2):
            columns = []
            y0, y1 = iy * height // 2, ((iy + 1) * height + 1) // 2
            for ix in range(2):
                x0, x1 = ix * width // 2, ((ix + 1) * width + 1) // 2
                columns.append(x[..., y0:y1, x0:x1].mean(dim=(-2, -1)))
            rows.append(torch.stack(columns, -1))
        return torch.stack(rows, -2)


class ProjectivePhaseNetwork(nn.Module):
    """Shared architecture with matched interventions; inference returns phase distribution.

    Inputs:
      contexts: [B,4,8,H,W] = 4 chronological half-windows, each 4 bins × 2 polarities.
      times: [B,4] midpoint times in seconds relative to LABEL anchor, NOT window end.
    Inputs must be already normalized using TRAIN-only statistics. Common ROI is fixed.
    """

    edges: Tensor
    offsets: Tensor

    def __init__(
        self,
        *,
        radius: int = 6,
        displacement_step: float = 0.0625,
        sweep_chunk: int = 8,
        checkpoint_sweep: bool = False,
        edges: Tensor | None = None,
        capacity: Capacity = "2M",
    ) -> None:
        super().__init__()
        self.register_buffer("edges", phase_edges() if edges is None else edges.detach().clone())
        self.register_buffer("offsets", translations(radius=radius, step=displacement_step))
        self.sweep_chunk = sweep_chunk
        self.checkpoint_sweep = checkpoint_sweep
        if capacity not in CAPACITY_WIDTHS:
            raise ValueError("unregistered model capacity")
        _, _, features, head0, head1, hidden = CAPACITY_WIDTHS[capacity]
        self.capacity = capacity
        self.feature_channels = features
        self.encoder = ContextEncoder(capacity)
        bins = len(self.edges) - 1
        self.direct = nn.Sequential(
            nn.Conv2d(4 * features, head0, 1, bias=False),
            nn.GroupNorm(8, head0),
            nn.SiLU(),
            ResidualBlock(head0),
            ResidualBlock(head0),
            nn.Conv2d(head0, head1, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, head1),
            nn.SiLU(),
            ResidualBlock(head1),
            QuadrantAveragePool(),
            nn.Flatten(),
            nn.Linear(head1 * 4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, bins),
        )
        self.support = nn.Sequential(nn.Conv2d(features, 16, 1), nn.SiLU(), nn.Conv2d(16, 1, 1))
        # 4 evidence + 4 validity + 4 gap embeddings + phase coordinate + direct logit.
        self.score_refiner = nn.Sequential(
            nn.Conv1d(14, 48, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(48, 48, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(48, 1, 1),
        )
        final_layer = cast(nn.Conv1d, self.score_refiner[-1])
        assert final_layer.bias is not None
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def encode(self, contexts: Tensor) -> Tensor:
        if contexts.ndim != 5 or contexts.shape[1:3] != (4, 8):
            raise ValueError("contexts must be [B,4,8,H,W]")
        if contexts.shape[-1] != contexts.shape[-2] or contexts.shape[-1] % 4:
            raise ValueError("common ROI must be square and divisible by four")
        b, _, _, h, w = contexts.shape
        f = self.encoder(contexts.reshape(b * 4, 8, h, w))
        return f.reshape(b, 4, self.feature_channels, f.shape[-2], f.shape[-1])

    def forward(
        self,
        contexts: Tensor,
        times: Tensor,
        *,
        arm: Arm = "PPS-TRAJ",
        permutation: Tensor | None = None,
        diagnostic_zero_geometry: bool = False,
        diagnostic_time_permutation: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if arm not in ("DIRECT", "PPS-PAIR", "PPS-TRAJ", "PPS-TIMEPERM"):
            raise ValueError("unknown arm")
        b = contexts.shape[0]
        if times.shape != (b, 4) or not bool(torch.isfinite(times).all()):
            raise ValueError("finite Bx4 relative times required")
        if not bool((times[:, 1:] > times[:, :-1]).all()):
            raise ValueError("context midpoints must be strictly chronological")
        if self.training and (diagnostic_zero_geometry or diagnostic_time_permutation is not None):
            raise ValueError("mechanism interventions are evaluation-only, never training recipes")
        geometry_times = times
        if diagnostic_time_permutation is not None:
            reference = torch.arange(4, device=times.device)[None].expand(b, -1)
            if (
                diagnostic_time_permutation.shape != (b, 4)
                or not torch.equal(diagnostic_time_permutation.sort(-1).values, reference)
                or bool((diagnostic_time_permutation == reference).any())
            ):
                raise ValueError("diagnostic time assignment requires a fixed-point-free bijection")
            geometry_times = times.gather(1, diagnostic_time_permutation)
        if arm == "PPS-TIMEPERM":
            if permutation is None or permutation.shape != (b, 4):
                raise ValueError("a per-row context permutation is required")
            reference = torch.arange(4, device=contexts.device)[None].expand(b, -1)
            if not torch.equal(permutation.sort(-1).values, reference) or bool(
                (permutation == reference).any()
            ):
                raise ValueError("TIMEPERM must be a fixed-point-free bijection")
            contexts = contexts.gather(1, permutation[:, :, None, None, None].expand_as(contexts))
        features = self.encode(contexts)
        prior = self.direct(features.flatten(1, 2)).float()
        with torch.autocast(device_type=contexts.device.type, enabled=False):
            k = prior.shape[-1]
            evidence = prior.new_zeros((b, 4, k))
            observable = torch.zeros((b, 4, k), device=prior.device, dtype=torch.bool)
            gaps = prior.new_zeros((b, 4, k))
            pairs = [(1, 3)] if arm == "PPS-PAIR" else [(0, 1), (1, 2), (2, 3), (0, 3)]
            centers = (self.edges[:-1] + self.edges[1:]) / 2
            q = phase_to_inverse_ttc(centers)[None].expand(b, -1)
            for j, (a, c) in enumerate(pairs):
                gaps[:, j] = (geometry_times[:, c, None] - geometry_times[:, a, None]) / 0.1
            if arm != "DIRECT" and not diagnostic_zero_geometry:
                for j, (a, c) in enumerate(pairs):
                    f0, f1 = features[:, a].float(), features[:, c].float()
                    h, w = f0.shape[-2:]
                    learned = self.support(f0).flatten(1).softmax(-1).reshape(b, h, w)
                    # Fixed uniform floor prevents one-pixel focus from nullifying scale evidence.
                    weights = 0.25 / (h * w) + 0.75 * learned
                    scale, physical = scale_between(
                        q.float(),
                        geometry_times[:, a, None].float(),
                        geometry_times[:, c, None].float(),
                    )
                    scores, overlap = sweep_scores(
                        f0,
                        f1,
                        scale,
                        self.offsets.float(),
                        weights,
                        chunk=self.sweep_chunk,
                        checkpoint_chunks=self.checkpoint_sweep,
                    )
                    ev, ok = marginalize_translation(scores, overlap)
                    ok = ok & physical
                    evidence[:, j] = torch.where(ok, ev, torch.zeros_like(ev))
                    observable[:, j] = ok
                    gaps[:, j] = (geometry_times[:, c, None] - geometry_times[:, a, None]) / 0.1
            coordinate = torch.asinh(centers / 0.03)[None, None].expand(b, 1, -1)
            joined = torch.cat(
                [
                    evidence,
                    observable.to(prior.dtype),
                    gaps,
                    coordinate.to(prior.dtype),
                    prior[:, None],
                ],
                1,
            )
            correction = self.score_refiner(joined).squeeze(1)
            logits = prior + correction
            med = quantile(logits, self.edges)
            return {
                "logits": logits,
                "phase": med,
                "ttc": phase_to_finite_ttc(med),
                "direct_logits": prior,
                "score_correction": correction,
                "evidence": evidence,
                "observability": observable,
            }
