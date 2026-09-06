"""Nonselectable TRAIN observations at the fixed supervised diagnostic checkpoints."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.artifacts.stage_endpoint_barrier_v11 import _owned
from e_jepa_ttc.data.event_context128 import RateNormalization
from e_jepa_ttc.data.event_context128_reader import TrainingView
from e_jepa_ttc.evaluation.architecture_diagnostics_v11 import posterior_diagnostics
from e_jepa_ttc.evaluation.architecture_mechanisms_v11 import (
    MECHANISM_NAMES,
    MECHANISM_RECIPE,
    mechanism_input,
)
from e_jepa_ttc.evaluation.initial_treatment_export_v11 import publish_export_payloads
from e_jepa_ttc.evaluation.pretext_diagnostics_v11 import readonly_diagnostic
from e_jepa_ttc.models.phase_distribution_v11 import weighted_training_loss
from e_jepa_ttc.models.projective_phase_network import ProjectivePhaseNetwork
from e_jepa_ttc.training.eap_highres_jepa import _capture_rng_state, _restore_rng_state
from e_jepa_ttc.training.stage70_update_ledger import FitSpec
from e_jepa_ttc.training.stage72_augmentation import time_derangement
from e_jepa_ttc.training.stage72_supervision import Supervision


def diagnose_supervised_series(
    root: Path,
    view: TrainingView,
    normalization: RateNormalization,
    supervision: Supervision,
    *,
    recipe_sha256: str,
    endpoint: int,
    check: Callable[[], None],
) -> list[dict[str, Any]]:
    """Observe every prescribed checkpoint before publishing a scientific stage barrier."""
    if endpoint not in (8000, 11000):
        raise ValueError("unsupported scientific supervised diagnostic endpoint")
    updates = (0, 1000, 2000, 4000, 8000) + ((11000,) if endpoint == 11000 else ())
    observations = [
        diagnose_supervised_checkpoint(
            binding(root / "supervised" / f"checkpoint_{update:06d}.pt"),
            view,
            normalization,
            supervision,
            recipe_sha256=recipe_sha256,
            output=root / f"supervised_diagnostics_{update:06d}",
            check=check,
        )
        for update in updates
    ]
    for mechanism in MECHANISM_NAMES:
        observations.append(
            diagnose_supervised_checkpoint(
                binding(root / "supervised" / f"checkpoint_{endpoint:06d}.pt"),
                view,
                normalization,
                supervision,
                recipe_sha256=recipe_sha256,
                output=root / "mechanism_diagnostics" / mechanism,
                check=check,
                mechanism=mechanism,
            )
        )
    return observations


def diagnostic_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Choose16 fixed target-free TRAIN rows, independent of checkpoint, score and seed."""
    if len(tokens) < 16 or len(set(tokens)) != len(tokens):
        raise ValueError("TRAIN checkpoint diagnostics require at least16 unique permitted rows")
    return tuple(
        sorted(
            tokens,
            key=lambda token: hashlib.sha256(("stage72_diag_v1/" + token).encode()).hexdigest(),
        )[:16]
    )


def embedding_health(values: np.ndarray) -> dict[str, float | None]:
    """Summarize global encoder descriptors without inventing a new model abort criterion."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("embedding diagnostics require a descriptor matrix")
    if not np.isfinite(x).all():
        return {"mean_std": None, "fraction_std_below_0p001": None, "effective_rank": None}
    std = x.std(axis=0)
    centered = x - x.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(x)).clip(0)
    probability = eigenvalues / eigenvalues.sum() if eigenvalues.sum() else eigenvalues
    nonzero = probability > 0
    rank = (
        float(np.exp(-(probability[nonzero] * np.log(probability[nonzero])).sum()))
        if nonzero.any()
        else 0.0
    )
    return {
        "mean_std": float(std.mean()),
        "fraction_std_below_0p001": float((std < 1e-3).mean()),
        "effective_rank": rank,
    }


def diagnose_supervised_checkpoint(
    checkpoint: dict[str, Any],
    view: TrainingView,
    normalization: RateNormalization,
    supervision: Supervision,
    *,
    recipe_sha256: str,
    output: Path,
    check: Callable[[], None],
    device: str = "cuda",
    mechanism: str | None = None,
) -> dict[str, Any]:
    """Observe frozen weights, full posterior and descriptor health; never choose an epoch."""
    if mechanism is not None and mechanism not in MECHANISM_NAMES:
        raise ValueError("unknown fixed mechanism observation")
    scored = mechanism in (None, "zero_geometry", "pair_time_permutation")
    payload = torch.load(verify(checkpoint), map_location="cpu", weights_only=False)
    spec = FitSpec(**payload["spec"])
    update = payload["update"]
    if (
        spec.recipe_sha256 != recipe_sha256
        or spec.training_view_sha256 != view.identity
        or normalization.training_view_sha256 != view.identity
        or supervision.tokens != view.tokens
        or spec.arm not in ("DIRECT", "PPS-PAIR", "PPS-TRAJ", "PPS-TIMEPERM")
        or not 0 <= update <= spec.endpoint
        or (not spec.technical and update not in (0, 1000, 2000, 4000, 8000, 11000))
    ):
        raise ValueError("supervised diagnostic checkpoint/source identity differs")
    tokens = diagnostic_tokens(view.tokens)
    identity = {
        "checkpoint": checkpoint,
        "recipe_sha256": recipe_sha256,
        "training_view_sha256": view.identity,
        "supervision_sha256": supervision.identity_sha256,
        "update": update,
        "seed": spec.seed,
        "arm": spec.arm,
        "ordered_tokens": list(tokens),
        "selectable": False,
        "augmentation": False,
        "optimizer_updates": 0,
        "target_precision_note": "float32 supervised phase target; diagnostic scope only",
        "mechanism": mechanism,
        "mechanism_recipe_sha256": object_digest(MECHANISM_RECIPE) if mechanism else None,
        "target_scoring": scored,
    }
    if output.resolve() != output.absolute():
        raise ValueError("supervised diagnostics output was redirected")
    output.mkdir(parents=True, exist_ok=True)
    intent = output / "DIAGNOSTIC_INTENT.json"
    if intent.exists():
        if json.loads(intent.read_text(encoding="utf-8")) != identity:
            raise ValueError("supervised diagnostic recovery changed its fixed observations")
    elif any(output.iterdir()):
        raise ValueError("supervised diagnostics lack a prospective identity")
    else:
        atomic_json(intent, identity)
    final = output / "SUPERVISED_DIAGNOSTICS.json"
    candidates = (
        [final] if final.exists() else list(output.glob("attempt_*/SUPERVISED_DIAGNOSTICS.json"))
    )
    if len(candidates) > 1:
        raise ValueError("multiple completed diagnostic attempts cannot be selected")
    if candidates:
        if candidates[0].resolve() != candidates[0].absolute():
            raise ValueError("completed supervised diagnostic was redirected")
        record = json.loads(candidates[0].read_text(encoding="utf-8"))
        unsigned = {key: value for key, value in record.items() if key != "artifact_sha256"}
        if (
            object_digest(unsigned) != record.get("artifact_sha256")
            or record.get("identity") != identity
        ):
            raise ValueError("completed supervised diagnostic changed")
        directory = Path(record["rows"]["path"]).parent
        if directory.parent != output or not directory.name.startswith("attempt_"):
            raise ValueError("supervised diagnostic payload escaped its owned attempt")
        _owned(record["rows"], directory, "ROWS.csv")
        _owned(record["arrays"], directory, "DIAGNOSTIC_ARRAYS.npz")
        if not final.exists():
            atomic_json(final, record)
        return record
    attempt = output / f"attempt_{len(list(output.glob('attempt_*'))):04d}"
    attempt.mkdir()
    atomic_json(attempt / "STARTED.json", identity)
    generator = torch.Generator().manual_seed(720076)
    rng = _capture_rng_state(generator)
    model = None
    hook = None
    collected: dict[str, list[np.ndarray]] = {}
    embeddings = []
    losses = []
    started = time.monotonic()
    try:
        model = ProjectivePhaseNetwork(checkpoint_sweep=False).to(device)
        model.load_state_dict(payload["model"], strict=True)

        def observe_encoder(
            module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], result: torch.Tensor
        ) -> None:
            del module, inputs
            embeddings.append(result.float().mean(dim=(-2, -1)).detach().cpu().numpy())

        hook = model.encoder.register_forward_hook(observe_encoder)
        index_by_token = {token: index for index, token in enumerate(view.tokens)}
        indices = [index_by_token[token] for token in tokens]
        with readonly_diagnostic(model, generator):
            for token, index in zip(tokens, indices, strict=True):
                check()
                sample = view.inputs(token, normalization)
                contexts, times, mechanism_kwargs = (
                    (sample.contexts, sample.times, {})
                    if mechanism is None
                    else mechanism_input(
                        sample, normalization, mechanism, token=token, seed=spec.seed
                    )
                )
                mechanism_kwargs = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in mechanism_kwargs.items()
                }
                permutation = (
                    torch.tensor([time_derangement(token, spec.seed)], device=device)
                    if spec.arm == "PPS-TIMEPERM"
                    else None
                )
                result = model(
                    contexts[None].to(device),
                    times[None].to(device),
                    arm=spec.arm,
                    permutation=permutation,
                    **mechanism_kwargs,
                )
                if (
                    device.startswith("cuda")
                    and torch.cuda.max_memory_allocated()
                    >= 0.9 * torch.cuda.get_device_properties(0).total_memory
                ):
                    raise RuntimeError(
                        "RESOURCE_BLOCKED: supervised diagnostic reached90 percent VRAM"
                    )
                if scored:
                    loss = weighted_training_loss(
                        result["logits"],
                        supervision.target_phase[index : index + 1].to(device),
                        model.edges,
                        supervision.mass[index : index + 1].to(device),
                        len(view.tokens),
                    )
                    losses.append(float(loss))
                for key, value in result.items():
                    collected.setdefault(key, []).append(value.cpu().numpy().copy())
        arrays = {key: np.concatenate(values) for key, values in collected.items()}
        diagnostic = posterior_diagnostics(arrays, model.edges.detach().cpu().numpy())
        diagnostic.rows.insert(0, "sample_token", tokens)
        if scored:
            target = supervision.target_phase[indices].cpu().numpy()
            diagnostic.rows["supervised_target_phase_float32"] = target
            diagnostic.rows["phase_absolute_error_vs_training_target"] = abs(
                diagnostic.rows.prediction_phase - target
            )
        descriptor = np.concatenate(embeddings)
        diagnostic.arrays["encoder_global_descriptors"] = descriptor
        row_source, array_source = publish_export_payloads(
            attempt, diagnostic.rows, diagnostic.arrays
        )
        finite_losses = np.asarray(losses, dtype=np.float64)
        record = {
            "kind": "NONSELECTABLE_SUPERVISED_TRAIN_CHECKPOINT_DIAGNOSTICS",
            "identity": identity,
            "rows": row_source,
            "arrays": array_source,
            "finite_fraction": float(diagnostic.rows.finite.mean()),
            "mean_training_objective_on_fixed_rows": float(finite_losses.mean())
            if len(finite_losses) and np.isfinite(finite_losses).all()
            else None,
            "embedding_health": embedding_health(descriptor),
            "last_training_gradient_norm": payload["history"][-1]["gradient_norm"]
            if payload["history"]
            else None,
            "incomplete_prior_attempts": len(list(output.glob("attempt_*"))) - 1,
            "elapsed_this_attempt_seconds": time.monotonic() - started,
        }
        record["artifact_sha256"] = object_digest(record)
        atomic_json(attempt / "SUPERVISED_DIAGNOSTICS.json", record)
        atomic_json(final, record)
        return record
    finally:
        if hook is not None:
            hook.remove()
        _restore_rng_state(rng, generator)
        del model, payload
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
