"""Production standalone supervised data flow for the fixed full-state trainer."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from e_jepa_ttc.data.event_context128 import RateNormalization
from e_jepa_ttc.data.event_context128_reader import TrainingView
from e_jepa_ttc.models.phase_distribution_v11 import weighted_training_loss
from e_jepa_ttc.models.projective_phase_network import Arm, ProjectivePhaseNetwork
from e_jepa_ttc.training.architecture_loss_diagnostics_v11 import supervised_loss_diagnostics
from e_jepa_ttc.training.stage70_fixed_fit import deterministic_runtime
from e_jepa_ttc.training.stage72_augmentation import (
    augment_observation,
    augmentation_generator,
    time_derangement,
)
from e_jepa_ttc.training.stage72_supervision import Supervision, initialize_train_prior


def initialize_architecture(
    seed: int,
    supervision: Supervision,
    *,
    encoder_state: Mapping[str, Tensor] | None = None,
) -> ProjectivePhaseNetwork:
    """Recreate every task head from the registered seed/prior; transfer only an encoder."""
    deterministic_runtime(seed)
    model = ProjectivePhaseNetwork(checkpoint_sweep=True)
    initialize_train_prior(model, supervision)
    if encoder_state is not None:
        expected = model.encoder.state_dict()
        if set(encoder_state) != set(expected) or any(
            value.shape != expected[key].shape
            or value.dtype != expected[key].dtype
            or not bool(torch.isfinite(value).all())
            for key, value in encoder_state.items()
        ):
            raise ValueError("pretext transfer requires exactly the finite context encoder state")
        model.encoder.load_state_dict(dict(encoder_state), strict=True)
    return model


class SupervisedBatches:
    """Uniform TRAIN sampling and fixed augmentation; identities never enter network channels."""

    def __init__(
        self,
        model: ProjectivePhaseNetwork,
        view: TrainingView,
        normalization: RateNormalization,
        supervision: Supervision,
        arm: Arm,
        seed: int,
        journal: Path,
    ) -> None:
        if view.tokens != supervision.tokens or normalization.training_view_sha256 != view.identity:
            raise ValueError("sampler, supervision and normalization TRAIN identities differ")
        if arm not in ("DIRECT", "PPS-PAIR", "PPS-TRAJ", "PPS-TIMEPERM"):
            raise ValueError("unregistered standalone arm")
        self.model, self.view, self.normalization = model, view, normalization
        self.supervision, self.arm, self.seed, self.journal = supervision, arm, seed, journal
        self.device = next(model.parameters()).device

    def __call__(self, update: int, index: int, batch: int, generator: torch.Generator) -> Tensor:
        """Return the globally weighted microbatch mean; FixedFit applies batch/16 once."""
        if batch not in (1, 2) or not 0 <= index < 16 // batch:
            raise ValueError("invalid accumulated supervised batch position")
        indices = torch.randint(len(self.view.tokens), (batch,), generator=generator)
        tokens = [self.view.tokens[int(i)] for i in indices]
        samples, trace = [], []
        for local, token in enumerate(tokens):
            position = index * batch + local
            sample, parameters = augment_observation(
                self.view.reader.observation(token),
                self.normalization,
                augmentation_generator(self.seed, update, position, token),
            )
            samples.append(sample)
            trace.append({"position": position, "token": token, "augmentation": asdict(parameters)})
        permutation = None
        if self.arm == "PPS-TIMEPERM":
            permutation = torch.tensor(
                [time_derangement(token, self.seed) for token in tokens],
                device=self.device,
                dtype=torch.int64,
            )
        output = self.model(
            torch.stack([sample.contexts for sample in samples]).to(self.device),
            torch.stack([sample.times for sample in samples]).to(self.device),
            arm=self.arm,
            permutation=permutation,
        )
        if not bool(torch.isfinite(output["ttc"]).all()):
            raise ValueError("standalone model emitted nonfinite TTC")
        targets = self.supervision.target_phase[indices].to(self.device)
        masses = self.supervision.mass[indices].to(self.device)
        loss = weighted_training_loss(
            output["logits"],
            targets,
            self.model.edges,
            masses,
            len(self.view.tokens),
        )
        components = supervised_loss_diagnostics(
            output["logits"],
            output["ttc"],
            targets,
            self.model.edges,
            masses,
            len(self.view.tokens),
            loss,
        )
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "update": update,
                        "rows": trace,
                        "time_permutation": None if permutation is None else permutation.tolist(),
                        "loss_components": components,
                    }
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        return loss
