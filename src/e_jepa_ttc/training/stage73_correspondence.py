"""Known-transform TRAIN texture correspondence, with no TTC or teacher inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional


@dataclass(frozen=True)
class TexturePair:
    """Two independently perturbed textures and their exact forward coordinate map."""

    source: Tensor
    target: Tensor
    source_valid: Tensor
    target_valid: Tensor
    source_to_target: Tensor


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand((), generator=generator))


def _warp(texture: Tensor, validity: Tensor, forward: Tensor) -> tuple[Tensor, Tensor]:
    inverse = torch.linalg.inv(forward)[:2].to(texture)
    grid = functional.affine_grid(inverse[None], [1, *texture.shape], align_corners=False)
    image = functional.grid_sample(texture[None], grid, align_corners=False)[0]
    support = functional.grid_sample(validity[None, None].float(), grid, align_corners=False)[0, 0]
    return image, support >= 1 - 1e-6


@torch.no_grad()
def texture_pair(
    texture: Tensor, sensor_support: Tensor, generator: torch.Generator
) -> TexturePair:
    """Use chartered spatial ranges, fixed ±10% gain and 0.01 Gaussian noise.

    The texture has already received its ancestor-safe TRAIN normalization.
    Noise/gain are independent per view. Padding is masked; optional cutout is off.
    Random draws use an explicit CPU generator whose state belongs in checkpoints.
    """
    if texture.shape != (8, 128, 128) or sensor_support.shape != (128, 128):
        raise ValueError("pretext requires one full-resolution 8-channel half-window")
    if texture.requires_grad or not bool(torch.isfinite(texture).all()):
        raise ValueError("pretext textures must be finite detached observations")
    common = torch.eye(3, dtype=torch.float32)
    common[0, 0] = common[1, 1] = _uniform(generator, 0.8, 1.2)
    scale = math.exp(_uniform(generator, -0.15, 0.15))
    angle = math.radians(_uniform(generator, -3, 3))
    relative = torch.tensor(
        [
            [scale * math.cos(angle), -scale * math.sin(angle), _uniform(generator, -0.25, 0.25)],
            [scale * math.sin(angle), scale * math.cos(angle), _uniform(generator, -0.25, 0.25)],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    source, source_valid = _warp(texture, sensor_support, common)
    target, target_valid = _warp(texture, sensor_support, relative @ common)
    images = []
    for image, mask in ((source, source_valid), (target, target_valid)):
        gain = _uniform(generator, 0.9, 1.1)
        noise = torch.randn(image.shape, generator=generator, dtype=torch.float32).to(image.device)
        images.append((image * gain + 0.01 * noise) * mask[None])
    return TexturePair(images[0], images[1], source_valid, target_valid, relative.to(texture))


def grid_centers(height: int, width: int, device: torch.device) -> Tensor:
    """Return row-major homogeneous align_corners=False feature centers."""
    y, x = torch.meshgrid(
        (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * 2 / height - 1,
        (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * 2 / width - 1,
        indexing="ij",
    )
    return torch.stack((x.flatten(), y.flatten(), torch.ones(height * width, device=device)), -1)


def bilinear_labels(coordinates: Tensor, height: int, width: int) -> tuple[Tensor, Tensor, Tensor]:
    """Four target class indices, probability masses and full-grid-domain validity."""
    xy = (coordinates[:, :2] + 1) * coordinates.new_tensor([width, height]) / 2 - 0.5
    lower = xy.floor()
    residual = xy - lower
    indices, masses = [], []
    valid = (xy[:, 0] >= 0) & (xy[:, 0] <= width - 1)
    valid &= (xy[:, 1] >= 0) & (xy[:, 1] <= height - 1)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x = (lower[:, 0] + dx).long().clamp(0, width - 1)
        y = (lower[:, 1] + dy).long().clamp(0, height - 1)
        indices.append(y * width + x)
        masses.append(
            (residual[:, 0] if dx else 1 - residual[:, 0])
            * (residual[:, 1] if dy else 1 - residual[:, 1])
        )
    return torch.stack(indices, -1), torch.stack(masses, -1), valid


@dataclass(frozen=True)
class CorrespondenceResult:
    """Differentiable symmetric loss and detached diagnostic controls."""

    loss: Tensor
    matching_error_pixels: float
    constant_descriptor_ce: float
    shuffled_correspondence_ce: float
    descriptor_variance: float
    masked_fraction: float
    query_counts: tuple[int, int]


def correspondence_loss(
    encoder: nn.Module, pair: TexturePair, generator: torch.Generator
) -> CorrespondenceResult | None:
    """Compute symmetric temperature-.07 soft CE; fewer than32 valid queries rejects.

    Query centers are sampled without replacement, up to128 per direction.
    A rejected pair must be redrawn at most three times, then logged/replenished
    by the training owner. It never constitutes an optimizer update.
    """
    with torch.autocast(device_type=pair.source.device.type, enabled=False):
        features = encoder(torch.stack((pair.source, pair.target)).float()).float()
        _, channels, height, width = features.shape
        descriptors = functional.normalize(features.flatten(2).transpose(1, 2), dim=-1)
        centers = grid_centers(height, width, features.device)
        masks = (
            functional.interpolate(
                torch.stack((pair.source_valid, pair.target_valid))[:, None].float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[:, 0].flatten(1)
            >= 1 - 1e-6
        )
        mappings = (pair.source_to_target, torch.linalg.inv(pair.source_to_target))
        selections = []
        for source, target in ((0, 1), (1, 0)):
            destination = centers @ mappings[source].T
            indices, mass, valid = bilinear_labels(destination, height, width)
            valid &= masks[source] & ((masks[target][indices] | (mass == 0)).all(-1))
            eligible = valid.nonzero().flatten()
            if len(eligible) < 32:
                return None
            order = torch.randperm(len(eligible), generator=generator)[:128].to(eligible.device)
            chosen = eligible[order]
            selections.append((source, target, destination, indices, mass, valid, chosen))
        losses, errors, shuffled, counts = [], [], [], []
        masked = 0.0
        for source, target, destination, indices, mass, valid, chosen in selections:
            logits = descriptors[source, chosen] @ descriptors[target].T / 0.07
            log_probs = logits.log_softmax(-1)
            losses.append(-(log_probs.gather(1, indices[chosen]) * mass[chosen]).sum(-1).mean())
            with torch.no_grad():
                predicted = centers[logits.argmax(-1), :2]
                pixel_delta = (predicted - destination[chosen, :2]) * pair.source.shape[-1] / 2
                errors.append(float(pixel_delta.norm(dim=-1).mean()))
                shuffled_rows = chosen.roll(1)
                shuffled.append(
                    float(
                        -(log_probs.gather(1, indices[shuffled_rows]) * mass[shuffled_rows])
                        .sum(-1)
                        .mean()
                    )
                )
                masked += 1 - float(valid.float().mean())
                counts.append(len(chosen))
        return CorrespondenceResult(
            torch.stack(losses).mean(),
            sum(errors) / 2,
            math.log(height * width),
            sum(shuffled) / 2,
            float(descriptors.detach().reshape(-1, channels).var(0, unbiased=False).mean()),
            masked / 2,
            (counts[0], counts[1]),
        )
