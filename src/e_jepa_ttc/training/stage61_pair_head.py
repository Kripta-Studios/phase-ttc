"""Fixed-budget cached PAIR-head training for Stage 61."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch, PairSupervisionBatch
from e_jepa_ttc.models.collision_clock_math import (
    benchmark_phase_to_inverse_ttc,
    neutral_raw_phase,
    phase_lower_bound,
)
from e_jepa_ttc.training.incremental_residual import deterministic_sequence_grouped_schedule
from e_jepa_ttc.training.stage70_fixed_fit import finish_checkpoint_publication
from e_jepa_ttc.training.stage70_update_ledger import FitSpec, UpdateLedger


@dataclass(frozen=True)
class PairHeadTrainingConfig:
    """The exact X0-PAIR-U recipe, applied to a sealed 133-D cache."""

    seed: int
    batch_size: int = 32
    update_budget: int = 6840
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    dropout: float = 0.05
    metric_delta_t_s: float = 0.1
    minimum_abs_prediction_ttc_s: float = 0.1
    checkpoint_policy: str = "last_update_fixed_budget"

    def __post_init__(self) -> None:
        if (self.batch_size, self.update_budget) != (32, 6840):
            raise ValueError("PAIR batch size/update budget are frozen to 32/6840")
        if (self.learning_rate, self.weight_decay, self.dropout) != (3e-4, 1e-4, 0.05):
            raise ValueError("PAIR optimizer/dropout recipe drifted")
        if self.checkpoint_policy != "last_update_fixed_budget":
            raise ValueError("PAIR requires the fixed last-update checkpoint")


class CachedPairDirectPhase(nn.Module):
    """Public cache-only equivalent of the X0 direct phase head."""

    def __init__(self, *, dropout: float = 0.05) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(133),
            nn.Linear(133, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.constant_(
            final.bias,
            neutral_raw_phase(metric_delta_t_s=0.1, minimum_abs_prediction_ttc_s=0.1),
        )

    def raw_phase(self, batch: PairFeatureBatch) -> torch.Tensor:
        """Return the unconstrained scalar emitted by the historical head."""

        return self.network(batch.features).squeeze(-1)

    def forward(self, batch: PairFeatureBatch) -> torch.Tensor:
        lower = phase_lower_bound(metric_delta_t_s=0.1, minimum_abs_prediction_ttc_s=0.1)
        return lower + functional.softplus(self.raw_phase(batch))

    def predict_ttc(self, batch: PairFeatureBatch) -> torch.Tensor:
        """Map the safe benchmark phase to unclipped scientific TTC."""

        return torch.reciprocal(benchmark_phase_to_inverse_ttc(self(batch), metric_delta_t_s=0.1))


def train_pair_head(
    *,
    features: np.ndarray,
    target_phase: np.ndarray,
    sample_weight: np.ndarray,
    sequence_ids: list[str],
    sample_tokens: list[str],
    config: PairHeadTrainingConfig,
    output_dir: Path,
    device: torch.device,
    identity: dict[str, Any],
    resume: bool = False,
    accounting: tuple[UpdateLedger, FitSpec] | None = None,
    technical_update_limit: int | None = None,
    check: Callable[[], None] = lambda: None,
) -> tuple[CachedPairDirectPhase, dict[str, Any]]:
    """Train only the PAIR head; no dev arrays are accepted by this API."""

    check()
    endpoint = config.update_budget
    if technical_update_limit is not None:
        if (
            accounting is None
            or not accounting[1].technical
            or not 1 <= technical_update_limit <= endpoint
        ):
            raise ValueError(
                "partial PAIR work is restricted to explicitly nonselectable technical fits"
            )
        endpoint = technical_update_limit
    if accounting is not None:
        ledger, spec = accounting
        if spec.endpoint != endpoint or spec.seed != config.seed:
            raise ValueError("PAIR reservation differs from the prescribed endpoint")
        ledger.reserve(spec)
    feature = np.asarray(features, dtype=np.float32)
    target = np.asarray(target_phase, dtype=np.float32).reshape(-1)
    weights = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    PairFeatureBatch(torch.from_numpy(feature))
    PairSupervisionBatch(torch.from_numpy(target), torch.from_numpy(weights))
    if len(feature) != len(sequence_ids) or len(feature) != len(sample_tokens):
        raise ValueError("PAIR cache identities are row-misaligned")
    schedule, schedule_sha = deterministic_sequence_grouped_schedule(
        sequence_ids,
        sample_tokens,
        seed=config.seed,
        batch_size=config.batch_size,
        updates=config.update_budget,
    )
    run_identity = {**identity, "config": asdict(config), "batch_schedule_sha256": schedule_sha}
    if accounting is not None:
        run_identity["fit_spec"] = asdict(accounting[1])
    publication_identity = object_digest(run_identity)
    output_dir.mkdir(parents=True, exist_ok=resume)
    checkpoint = output_dir / "model_last.pt"
    progress = output_dir / "progress.jsonl"
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = CachedPairDirectPhase(dropout=config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    start, losses = 0, []
    if resume:
        if accounting is not None:
            finish_checkpoint_publication(
                output_dir, publication_identity, accounting[0], accounting[1].job_id
            )
            recovered = accounting[0].recover(accounting[1].job_id)
            if recovered is None:
                raise ValueError("PAIR has no valid checkpoint to resume")
            checkpoint = verify(recovered)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload["identity"] != run_identity:
            raise ValueError("PAIR resume identity mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start = int(payload["completed_updates"])
        losses = [float(value) for value in payload["losses"]]
        torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
        if torch.cuda.is_available() and payload["torch_cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state"])
    elif checkpoint.exists() or progress.exists():
        raise FileExistsError("PAIR output exists; pass resume explicitly")
    tensors = (
        torch.from_numpy(feature).to(device),
        torch.from_numpy(target).to(device),
        torch.from_numpy(weights).to(device),
    )
    milestones = {0, 250, 500, 1000, 2000, 4000, 6840}

    def save(update: int, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_type": "stage61_pair_head_checkpoint_v1",
            "identity": run_identity,
            "completed_updates": update,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "losses": losses,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def publish(update: int) -> Path:
        check()
        path = output_dir / f"checkpoint_{update:06d}.pt"
        if path.exists():
            raise ValueError("PAIR checkpoint exists; recover its publication")
        temporary = path.with_suffix(".pt.tmp")
        save(update, temporary)
        intent = {"identity": publication_identity, "update": update, "payload": binding(temporary)}
        atomic_json(
            output_dir / "PENDING_CHECKPOINT.json", {**intent, "sha256": object_digest(intent)}
        )
        assert accounting is not None
        finish_checkpoint_publication(
            output_dir, publication_identity, accounting[0], accounting[1].job_id
        )
        return path

    if start == 0 and not resume:
        if accounting is not None:
            publish(0)
        else:
            save(0, output_dir / "milestones" / "update-000000.pt")
    began = time.perf_counter()
    for update in range(start, endpoint):
        check()
        indices = torch.as_tensor(schedule[update], dtype=torch.long, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(PairFeatureBatch(tensors[0][indices]))
        error = functional.smooth_l1_loss(prediction, tensors[1][indices], reduction="none")
        loss = error.mean()  # Exact X0 PAIR reduction; weights are intentionally unused.
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("PAIR loss became non-finite")
        loss.backward()
        if accounting is not None:
            accounting[0].begin_step(accounting[1].job_id)
        optimizer.step()
        if accounting is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            accounting[0].confirm_step(accounting[1].job_id)
        losses.append(float(loss.detach().cpu()))
        completed = update + 1
        with progress.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"update": completed, "loss": losses[-1]}) + "\n")
        if accounting is not None:
            if completed % 100 == 0 or completed in milestones or completed == endpoint:
                checkpoint = publish(completed)
        else:
            if completed % 100 == 0 or completed in milestones:
                save(completed, checkpoint)
            if completed in milestones:
                save(completed, output_dir / "milestones" / f"update-{completed:06d}.pt")
    elapsed = time.perf_counter() - began
    if accounting is not None:
        accounting[0].complete(accounting[1].job_id)
    manifest = sign_artifact(
        {
            "artifact_type": "stage61_pair_head_training_v1",
            "status": "technical_nonselectable"
            if technical_update_limit is not None
            else "completed",
            "identity": run_identity,
            "completed_updates": endpoint,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": compute_file_hash(str(checkpoint)),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "elapsed_seconds": elapsed,
            "outer_dev_available_to_trainer": False,
            "outer_dev_used_for_selection": False,
        }
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return model.eval(), manifest


def load_pair_head(path: Path, *, device: torch.device) -> CachedPairDirectPhase:
    """Load a completed Stage 61 cached PAIR checkpoint."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("completed_updates") != 6840:
        raise ValueError("PAIR checkpoint is not the fixed-budget endpoint")
    model = CachedPairDirectPhase().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


__all__ = [
    "CachedPairDirectPhase",
    "PairHeadTrainingConfig",
    "load_pair_head",
    "train_pair_head",
]
