"""Registered supervised count augmentations with unchanged physical time and TTC."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from e_jepa_ttc.data.event_context128 import ArchitectureInput, RateNormalization, four_contexts
from e_jepa_ttc.data.event_context128_reader import CountObservation

DERANGEMENTS = tuple(
    perm
    for perm in itertools.permutations(range(4))
    if all(index != value for index, value in enumerate(perm))
)


def time_derangement(token: str, seed: int) -> tuple[int, ...]:
    """Fix one of all nine derangements per token/seed without consuming sampler RNG."""
    if not token:
        raise ValueError("TIMEPERM needs a nonempty token")
    digest = hashlib.sha256(f"EJEPA_STAGE72_TIMEPERM_20260906|{seed}|{token}".encode()).digest()
    return DERANGEMENTS[int.from_bytes(digest, "big") % len(DERANGEMENTS)]


def augmentation_generator(seed: int, update: int, position: int, token: str) -> torch.Generator:
    """Make row augmentations invariant to physical microbatch and independent of arm."""
    if update < 1 or not 0 <= position < 16 or not token:
        raise ValueError("invalid effective-batch augmentation position")
    digest = hashlib.sha256(
        f"EJEPA_STAGE72_AUG_20260906|{seed}|{update}|{position}|{token}".encode()
    ).digest()
    return torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "big") % (2**63))


@dataclass(frozen=True)
class AugmentationParameters:
    """Common spatial transform, independent binomial events, two polarity-wide gains."""

    horizontal_flip: bool
    dx: int
    dy: int
    retention_probability: float
    polarity_gain: tuple[float, float]

    def __post_init__(self) -> None:
        if type(self.horizontal_flip) is not bool or any(
            type(v) is not int for v in (self.dx, self.dy)
        ):
            raise ValueError("flip and integer translation types are invalid")
        if not -4 <= self.dx <= 4 or not -4 <= self.dy <= 4:
            raise ValueError("crop translation exceeds four ROI pixels")
        if not 0.8 <= self.retention_probability <= 1 or len(self.polarity_gain) != 2:
            raise ValueError("event thinning probability or polarity gain shape invalid")
        if any(not 0.9 <= value <= 1.1 for value in self.polarity_gain):
            raise ValueError("polarity-wide count gain outside registered range")


def sample_parameters(generator: torch.Generator) -> AugmentationParameters:
    """Sample the fixed recipe on CPU; no labels, buckets or sequence categories are accepted."""
    flip = bool(torch.rand((), generator=generator) < 0.5)
    shift = torch.randint(-4, 5, (2,), generator=generator).tolist()
    values = torch.rand(3, generator=generator, dtype=torch.float64).tolist()
    return AugmentationParameters(
        flip,
        int(shift[0]),
        int(shift[1]),
        0.8 + 0.2 * values[0],
        (0.9 + 0.2 * values[1], 0.9 + 0.2 * values[2]),
    )


def spatial_transform(value: Tensor, parameters: AugmentationParameters) -> Tensor:
    """Apply flip then translation identically to all leading channels, with zero padding."""
    result = value.flip(-1) if parameters.horizontal_flip else value
    result = torch.roll(result, (parameters.dy, parameters.dx), (-2, -1))
    if parameters.dy > 0:
        result[..., : parameters.dy, :] = 0
    elif parameters.dy < 0:
        result[..., parameters.dy :, :] = 0
    if parameters.dx > 0:
        result[..., : parameters.dx] = 0
    elif parameters.dx < 0:
        result[..., parameters.dx :] = 0
    return result


def augment_observation(
    observation: CountObservation,
    normalization: RateNormalization,
    generator: torch.Generator,
    *,
    parameters: AugmentationParameters | None = None,
) -> tuple[ArchitectureInput, AugmentationParameters]:
    """Thin integer event counts before rate/gain/log normalization; keep all four clocks."""
    original = four_contexts(
        observation.counts,
        observation.durations_s,
        observation.support,
        observation.midpoint_twice_relative_us,
        normalization,
    )
    parameters = sample_parameters(generator) if parameters is None else parameters
    # Float64 represents every uint32 count exactly for independent binomial draws.
    counts = torch.from_numpy(observation.counts.astype(np.float64))
    thinned = (
        torch.binomial(
            counts, torch.full_like(counts, parameters.retention_probability), generator=generator
        )
        .numpy()
        .astype(np.float32)
    )
    gains = np.asarray(parameters.polarity_gain, dtype=np.float32)[None, None, :, None, None]
    duration = observation.durations_s.astype(np.float32)[:, :, None, None, None]
    values = np.log1p(thinned * gains / duration)
    mean = np.asarray(normalization.mean, dtype=np.float32)[None, None, :, None, None]
    std = np.asarray(normalization.std, dtype=np.float32)[None, None, :, None, None]
    normalized = (values - mean) / std
    normalized *= observation.support[None, None, None]
    contexts = spatial_transform(torch.from_numpy(normalized.reshape(4, 8, 128, 128)), parameters)
    support = spatial_transform(original.sensor_support, parameters)
    return ArchitectureInput(contexts, original.times, support), parameters
