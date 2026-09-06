"""Role-bound TRAIN textures, diagnostic holdout and bounded correspondence redraws."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor, nn

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.data.event_context128 import RateNormalization
from e_jepa_ttc.data.event_context128_reader import TrainingView
from e_jepa_ttc.training.stage73_correspondence import correspondence_loss, texture_pair


def texture_split(tokens: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Reserve the first floor10% hashed TRAIN tokens, including all four of their textures."""
    if len(tokens) < 10 or len(set(tokens)) != len(tokens):
        raise ValueError("pretext requires at least10 unique TRAIN tokens")
    ordered = sorted(
        tokens,
        key=lambda token: hashlib.sha256(
            f"EJEPA_STAGE73_HELD_TEXTURE_20260906|{token}".encode()
        ).digest(),
    )
    held = frozenset(ordered[: len(tokens) // 10])
    return tuple(token for token in tokens if token not in held), tuple(
        token for token in tokens if token in held
    )


class PretextTextures:
    """No labels or task heads; every draw comes from the ancestor-safe TRAIN count view.

    The diagnostic10% is held out of encoder fitting, within the same permitted
    TRAIN universe. The shared normalization was fitted on that full TRAIN view;
    this is not independent confirmation or an outer validation split.
    """

    def __init__(self, view: TrainingView, normalization: RateNormalization, journal: Path) -> None:
        if normalization.training_view_sha256 != view.identity:
            raise ValueError("pretext normalization belongs to another ancestor view")
        self.view, self.normalization, self.journal = view, normalization, journal
        self.train_tokens, self.held_tokens = texture_split(view.tokens)
        self.identity = object_digest(
            {
                "training_view": view.identity,
                "normalization": asdict(normalization),
                "encoder_fit_tokens": self.train_tokens,
                "diagnostic_tokens": self.held_tokens,
                "maximum_redraws": 3,
                "maximum_replenishment_draws_per_microbatch": 1024,
                "diagnostic_normalization": (
                    "full permissible TRAIN view; not an independent holdout"
                ),
            }
        )
        journal.parent.mkdir(parents=True, exist_ok=True)

    def _record(self, record: dict[str, object]) -> None:
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def objective(
        self,
        encoder: nn.Module,
        *,
        update: int,
        microbatch_index: int,
        batch: int,
        generator: torch.Generator,
        diagnostic: bool = False,
    ) -> Tensor:
        """Return an unscaled valid-image mean, replenishing rejected draws without updates."""
        if batch not in (1, 2):
            raise ValueError("pretext must preserve the registered physical microbatch")
        tokens = self.held_tokens if diagnostic else self.train_tokens
        device = next(encoder.parameters()).device
        losses = []
        for draw in range(1024):
            token = tokens[int(torch.randint(len(tokens), (), generator=generator))]
            context = int(torch.randint(4, (), generator=generator))
            sample = self.view.inputs(token, self.normalization)
            texture, mask = sample.contexts[context].to(device), sample.sensor_support.to(device)
            self._record(
                {
                    "kind": "texture_read",
                    "update": update,
                    "microbatch_index": microbatch_index,
                    "draw": draw,
                    "token": token,
                    "context": context,
                    "diagnostic": diagnostic,
                }
            )
            for redraw in range(4):
                pair = texture_pair(texture, mask, generator)
                result = correspondence_loss(encoder, pair, generator)
                if result is None:
                    self._record(
                        {
                            "kind": "invalid_transform",
                            "update": update,
                            "microbatch_index": microbatch_index,
                            "draw": draw,
                            "redraw": redraw,
                            "token": token,
                            "context": context,
                            "diagnostic": diagnostic,
                            "optimizer_update_consumed": False,
                        }
                    )
                    continue
                self._record(
                    {
                        "kind": "valid_correspondence",
                        "update": update,
                        "microbatch_index": microbatch_index,
                        "draw": draw,
                        "redraw": redraw,
                        "token": token,
                        "context": context,
                        "diagnostic": diagnostic,
                        "cross_entropy": float(result.loss.detach()),
                        "matching_error_pixels": result.matching_error_pixels,
                        "constant_descriptor_ce": result.constant_descriptor_ce,
                        "shuffled_correspondence_ce": result.shuffled_correspondence_ce,
                        "descriptor_variance": result.descriptor_variance,
                        "masked_fraction": result.masked_fraction,
                        "query_counts": result.query_counts,
                    }
                )
                losses.append(result.loss)
                break
            if len(losses) == batch:
                return torch.stack(losses).mean()
        raise RuntimeError(
            "PRETEXT_DATA_BLOCKED:1024 replenishment draws without a valid microbatch"
        )
