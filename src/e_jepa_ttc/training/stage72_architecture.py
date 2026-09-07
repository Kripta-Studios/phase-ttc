"""Production standalone supervised data flow for the fixed full-state trainer."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from e_jepa_ttc.data.event_context128 import RateNormalization
from e_jepa_ttc.data.event_context128_reader import TrainingView
from e_jepa_ttc.models.phase_distribution_v11 import weighted_training_loss
from e_jepa_ttc.models.projective_phase_network import Arm, Capacity, ProjectivePhaseNetwork
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
    capacity: Capacity = "2M",
) -> ProjectivePhaseNetwork:
    """Recreate every task head from the registered seed/prior; transfer only an encoder."""
    deterministic_runtime(seed)
    model = ProjectivePhaseNetwork(checkpoint_sweep=True, capacity=capacity)
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


@dataclass
class PreparedMicrobatch:
    """One CPU-only batch; speculative sampling state is committed on consumption."""

    indices: Tensor
    contexts: Tensor
    times: Tensor
    tokens: list[str]
    trace: list[dict[str, object]]
    sampler_state: Tensor


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
        *,
        prefetch: bool = False,
    ) -> None:
        if view.tokens != supervision.tokens or normalization.training_view_sha256 != view.identity:
            raise ValueError("sampler, supervision and normalization TRAIN identities differ")
        if arm not in ("DIRECT", "PPS-PAIR", "PPS-TRAJ", "PPS-TIMEPERM"):
            raise ValueError("unregistered standalone arm")
        self.model, self.view, self.normalization = model, view, normalization
        self.supervision, self.arm, self.seed, self.journal = supervision, arm, seed, journal
        self.device = next(model.parameters()).device
        self.executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self.pending: Future[PreparedMicrobatch] | None = None
        self.pending_key: tuple[int, int, int] | None = None
        self.pending_rng: Tensor | None = None

    def close(self) -> None:
        """Join the sole CPU preparer before readers, checkpoints or producers change."""
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        self.pending = None
        self.pending_key = None
        self.pending_rng = None

    def _prepare(
        self, update: int, index: int, batch: int, generator: torch.Generator
    ) -> PreparedMicrobatch:
        if batch not in (1, 2) or not 0 <= index < 16 // batch:
            raise ValueError("invalid accumulated supervised batch position")
        indices = torch.randint(len(self.view.tokens), (batch,), generator=generator)
        tokens = [self.view.tokens[int(i)] for i in indices]
        samples = []
        trace: list[dict[str, object]] = []
        for local, token in enumerate(tokens):
            position = index * batch + local
            sample, parameters = augment_observation(
                self.view.reader.observation(token),
                self.normalization,
                augmentation_generator(self.seed, update, position, token),
            )
            samples.append(sample)
            trace.append({"position": position, "token": token, "augmentation": asdict(parameters)})
        return PreparedMicrobatch(
            indices,
            torch.stack([sample.contexts for sample in samples]),
            torch.stack([sample.times for sample in samples]),
            tokens,
            trace,
            generator.get_state(),
        )

    def __call__(self, update: int, index: int, batch: int, generator: torch.Generator) -> Tensor:
        """Return the globally weighted microbatch mean; FixedFit applies batch/16 once."""
        if batch not in (1, 2) or not 0 <= index < 16 // batch:
            raise ValueError("invalid accumulated supervised batch position")
        if self.pending is None:
            prepared = self._prepare(update, index, batch, generator)
        else:
            if (
                self.pending_key != (update, index, batch)
                or self.pending_rng is None
                or not torch.equal(generator.get_state(), self.pending_rng)
            ):
                raise ValueError("prefetched batch differs from the consumed sampler position")
            prepared = self.pending.result()
            generator.set_state(prepared.sampler_state)
            self.pending = None
        # Do not prepare across an optimizer/checkpoint boundary. Only this clone
        # advances speculatively; the captured training RNG stays at consumption.
        if self.executor is not None and index + 1 < 16 // batch:
            self.pending_key = (update, index + 1, batch)
            self.pending_rng = generator.get_state()
            clone = torch.Generator(device="cpu").set_state(self.pending_rng)
            self.pending = self.executor.submit(self._prepare, update, index + 1, batch, clone)
        indices, tokens, trace = prepared.indices, prepared.tokens, prepared.trace
        permutation = None
        if self.arm == "PPS-TIMEPERM":
            permutation = torch.tensor(
                [time_derangement(token, self.seed) for token in tokens],
                device=self.device,
                dtype=torch.int64,
            )
        output = self.model(
            prepared.contexts.to(self.device),
            prepared.times.to(self.device),
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
