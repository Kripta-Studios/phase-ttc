"""Fixed-endpoint FP32 training with accumulated batches and exact local resume."""

from __future__ import annotations

import json
import math
import os
import platform
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor, nn

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.reproducibility import seed_everything
from e_jepa_ttc.training.eap_highres_jepa import _capture_rng_state, _restore_rng_state
from e_jepa_ttc.training.stage70_update_ledger import FitSpec, UpdateLedger


def finish_checkpoint_publication(
    output: Path, identity: str, ledger: UpdateLedger, job_id: str
) -> None:
    """Finish the existing flushed-payload publication transaction for either trainer."""
    intent_path = output / "PENDING_CHECKPOINT.json"
    if not intent_path.exists():
        return
    intent = json.loads(intent_path.read_text())
    signature = intent.pop("sha256")
    if signature != object_digest(intent) or intent["identity"] != identity:
        raise ValueError("checkpoint publication intent identity mismatch")
    update = intent["update"]
    target = output / f"checkpoint_{update:06d}.pt"
    temporary = target.with_suffix(".pt.tmp")
    if intent["payload"]["path"] != str(temporary.resolve()):
        raise ValueError("checkpoint publication path escaped its owned output")
    if target.exists():
        verify({**intent["payload"], "path": str(target.resolve())})
    else:
        verify(intent["payload"])
        os.replace(temporary, target)
    ledger.checkpoint(job_id, target, update)
    intent_path.rename(output / f"PUBLISHED_CHECKPOINT_{update:06d}.json")


def prescribed_lr(update: int, endpoint: int, warmup: int) -> float:
    """One-based update schedule: warmup3e-5→3e-4, cosine→3e-5 at endpoint."""
    if not 1 <= update <= endpoint or not 2 <= warmup < endpoint:
        raise ValueError("invalid fixed optimizer schedule")
    start, peak, end = 3e-5, 3e-4, 3e-5
    if update <= warmup:
        return start + (peak - start) * (update - 1) / (warmup - 1)
    fraction = (update - warmup) / (endpoint - warmup)
    return end + (peak - end) * (1 + math.cos(math.pi * fraction)) / 2


def deterministic_runtime(seed: int) -> None:
    """Configure the registered FP32 runtime before constructing model or CUDA tensors."""
    if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS determinism must be configured before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(seed)


class FixedFit:
    """Use the existing RNG helpers and resource owner, with one atomic checkpoint owner.

    The callback must return an unscaled microbatch mean and consume all sampling
    randomness through the supplied generator or captured Python/NumPy/Torch RNGs.
    It must not maintain an uncaptured data-loader cursor. Logical update and
    microbatch index are provided for deterministic row/permutation reconstruction.
    """

    def __init__(
        self,
        model: nn.Module,
        spec: FitSpec,
        ledger: UpdateLedger,
        owner: ResourceOwner,
        output: Path,
        *,
        warmup: int,
        microbatch: int = 2,
        checkpoint_interval: int = 250,
    ) -> None:
        if microbatch not in (1, 2) or checkpoint_interval < 1:
            raise ValueError("physical microbatch must preserve effective batch16")
        prescribed_lr(1, spec.endpoint, warmup)
        self.model, self.spec, self.ledger, self.owner = model, spec, ledger, owner
        self.output, self.warmup = output, warmup
        self.microbatch, self.checkpoint_interval = microbatch, checkpoint_interval
        self.device = next(model.parameters()).device
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("this registered execution implementation requires FP32 parameters")
        self.generator = torch.Generator(device="cpu").manual_seed(spec.seed)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-5,
            weight_decay=1e-4,
            foreach=False,
        )
        self.update = 0
        self.history: list[dict[str, float | int]] = []
        self.runtime = {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "warmup": warmup,
            "microbatch": microbatch,
            "effective_batch": 16,
            "precision": "float32_TF32_off",
            "checkpoint_interval": checkpoint_interval,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
        if not self.runtime["deterministic"]:
            raise ValueError("strict deterministic runtime required")
        self.identity = object_digest({"spec": asdict(spec), "runtime": self.runtime})
        self.output.mkdir(parents=True, exist_ok=True)
        ledger.reserve(spec)

    def resume(self) -> None:
        """Restore every state from the latest ledger-bound checkpoint, never a best score."""
        self._finish_publication()
        job = self.ledger.state["jobs"][self.spec.job_id]
        fallback = None
        if job["status"] != "COMPLETE" and job["checkpoint"] is not None:
            checkpoint_path = Path(job["checkpoint"]["path"])
            expected_path = self.output.absolute() / f"checkpoint_{job['checkpoint_update']:06d}.pt"
            if (
                checkpoint_path.resolve() != expected_path
                or checkpoint_path.is_symlink()
                or (checkpoint_path.is_file() and checkpoint_path.stat().st_nlink != 1)
            ):
                raise PermissionError(
                    "checkpoint recovery cannot follow a redirected or linked source"
                )
            try:
                verify(job["checkpoint"])
            except (OSError, ValueError):
                from e_jepa_ttc.training.checkpoint_recovery_v11 import (
                    existing_recovery,
                    latest_valid_publication,
                    preserve_corrupt_publications,
                )

                recovery = existing_recovery(self.output, self.identity, job["checkpoint_update"])
                if recovery is None:
                    recovery = latest_valid_publication(
                        self.output, self.identity, job["checkpoint_update"]
                    )
                fallback = preserve_corrupt_publications(self.output, recovery)
        record = self.ledger.recover(self.spec.job_id, fallback=fallback)
        if record is None:
            if self.update != 0:
                raise ValueError("initial-state resume requires a freshly seeded model")
            return
        payload = torch.load(record["path"], map_location=self.device, weights_only=False)
        if payload["identity"] != self.identity or payload["spec"] != asdict(self.spec):
            raise ValueError("resume code/config/runtime/view identity differs")
        expected_update = self.ledger.state["jobs"][self.spec.job_id]["checkpoint_update"]
        if payload["update"] != expected_update:
            raise ValueError("payload and ledger checkpoint update disagree")
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.update, self.history = payload["update"], payload["history"]
        _restore_rng_state(payload["rng"], self.generator)

    def _finish_publication(self) -> None:
        finish_checkpoint_publication(self.output, self.identity, self.ledger, self.spec.job_id)

    def save(self) -> Path:
        """Synchronously flush the complete state before atomically publishing its binding."""
        self.owner.check()
        path = self.output / f"checkpoint_{self.update:06d}.pt"
        if path.exists():
            raise ValueError("immutable checkpoint already exists; resume its ledger binding")
        temporary = path.with_suffix(".pt.tmp")
        payload = {
            "identity": self.identity,
            "spec": asdict(self.spec),
            "runtime": self.runtime,
            "update": self.update,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": _capture_rng_state(self.generator),
            "history": self.history,
        }
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        intent = {"identity": self.identity, "update": self.update, "payload": binding(temporary)}
        atomic_json(
            self.output / "PENDING_CHECKPOINT.json", {**intent, "sha256": object_digest(intent)}
        )
        self._finish_publication()
        return path

    def run(
        self,
        microbatch_loss: Callable[[int, int, int, torch.Generator], Tensor],
        *,
        stop_after: int | None = None,
    ) -> None:
        """Run prescribed updates; an explicit technical interruption never completes a fit."""
        end = self.spec.endpoint if stop_after is None else stop_after
        if not self.update < end <= self.spec.endpoint:
            raise ValueError("invalid fixed-fit execution boundary")
        self.model.train()
        for update in range(self.update + 1, end + 1):
            self.owner.check()
            started = time.perf_counter()
            lr = prescribed_lr(update, self.spec.endpoint, self.warmup)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for micro_index in range(16 // self.microbatch):
                self.owner.check()
                loss = microbatch_loss(update, micro_index, self.microbatch, self.generator)
                if loss.ndim or not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite/non-scalar microbatch objective")
                scaled = loss * (self.microbatch / 16)
                scaled.backward()
                loss_value += float(scaled.detach())
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0, error_if_nonfinite=True
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                allocated = torch.cuda.max_memory_allocated(self.device)
                if allocated >= 0.9 * torch.cuda.get_device_properties(self.device).total_memory:
                    raise RuntimeError("RESOURCE_BLOCKED: GPU allocation exceeds90%")
            self.ledger.begin_step(self.spec.job_id)
            self.optimizer.step()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.ledger.confirm_step(self.spec.job_id)
            self.update = update
            self.history.append(
                {
                    "update": update,
                    "loss": loss_value,
                    "gradient_norm": float(norm),
                    "learning_rate": lr,
                    "seconds": time.perf_counter() - started,
                }
            )
            if update % self.checkpoint_interval == 0 or update == end:
                self.save()
                print(
                    f"{self.spec.job_id} update{update}/{self.spec.endpoint}; checkpoint published",
                    flush=True,
                )
        if self.update == self.spec.endpoint:
            self.ledger.complete(self.spec.job_id)
