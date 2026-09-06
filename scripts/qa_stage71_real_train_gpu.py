"""Nonselectable real TRAIN GPU integration: DIRECT/PPS-TRAJ,10 versus5+resume5."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch
from qa_stage71_fixed_fit import equal_state

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.data.event_context128 import RateNormalization
from e_jepa_ttc.data.event_context128_reader import CountReader, TrainingView
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.models.phase_distribution_v11 import weighted_training_loss
from e_jepa_ttc.models.projective_phase_network import Arm, ProjectivePhaseNetwork
from e_jepa_ttc.training.stage70_fixed_fit import FixedFit, deterministic_runtime
from e_jepa_ttc.training.stage70_update_ledger import FitSpec, UpdateLedger
from e_jepa_ttc.training.stage72_supervision import (
    Supervision,
    initialize_train_prior,
    train_supervision,
)


def loss_callback(
    model: ProjectivePhaseNetwork,
    view: TrainingView,
    normalization: RateNormalization,
    supervision: Supervision,
    arm: Arm,
    journal: Path,
) -> Callable[[int, int, int, torch.Generator], torch.Tensor]:
    """Uniform full fold-TRAIN sampling; only contexts and clocks reach the network."""

    def loss(update: int, index: int, batch: int, generator: torch.Generator) -> torch.Tensor:
        indices = torch.randint(len(view.tokens), (batch,), generator=generator)
        tokens = [view.tokens[int(i)] for i in indices]
        samples = [view.inputs(token, normalization) for token in tokens]
        contexts = torch.stack([sample.contexts for sample in samples]).cuda()
        times = torch.stack([sample.times for sample in samples]).cuda()
        output = model(contexts, times, arm=arm)
        if not bool(torch.isfinite(output["ttc"]).all()):
            raise ValueError("real TRAIN smoke emitted nonfinite TTC")
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"update": update, "microbatch_index": index, "tokens": tokens}) + "\n"
            )
        return weighted_training_loss(
            output["logits"],
            supervision.target_phase[indices].cuda(),
            model.edges,
            supervision.mass[indices].cuda(),
            len(view.tokens),
        )

    return loss


def main() -> None:
    """Run40 counted technical updates; no outer prediction or selection is performed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if not campaign.is_relative_to(ROOT / "artifacts"):
        raise ValueError("GPU QA output must remain in this worktree")
    output = campaign / "technical_real_train_gpu"
    receipt_path = output / "REAL_TRAIN_GPU_EQUIVALENCE.json"
    if receipt_path.exists():
        raise ValueError("completed GPU QA receipt exists; do not rerun completed fits")
    parity = json.loads((campaign / "histogram_reference64/HISTOGRAM_PARITY.json").read_text())
    if parity["status"] != "PASS" or parity["samples"] != 64:
        raise ValueError("independent real-count parity prerequisite failed")
    torch.set_num_threads(2)
    deterministic_runtime(7)
    if not torch.cuda.is_available():
        raise RuntimeError("real TRAIN GPU smoke requires CUDA")
    reader = CountReader(campaign / "counts_original8192")
    role_record = json.loads((campaign / "data_roles/DATA_ROLES.json").read_text())
    signature = role_record.pop("artifact_sha256")
    if object_digest(role_record) != signature:
        raise ValueError("fixed TRAIN roles signature changed")
    role_values = role_record["roles"]
    roles = TrainRoles(
        tuple((str(pair[0]), str(pair[1])) for pair in role_values["sequence_to_group"]),
        tuple(role_values["original"]),
        tuple(role_values["historical_exposed"]),
        tuple(role_values["protected"]),
        tuple(role_values["confirmation"]),
        tuple(role_values["expansion"]),
        str(role_values["role_prefix"]),
    )
    excluded = tuple(
        sorted(set(reader.metadata.loc[reader.metadata.outer_fold == 0, "sequence_id"]))
    )
    tokens = tuple(sorted(reader.metadata.loc[reader.metadata.outer_fold != 0, "sample_token"]))
    view = TrainingView(reader, tokens, roles, excluded_sequences=excluded, include_expansion=False)
    labels_path = campaign / "preflight/HISTORICAL_AUDIT_ROWS.csv"
    labels = pd.read_csv(
        labels_path, usecols=["sample_token", "sequence_id", "target_ttc", "bucket"]
    )
    supervision = train_supervision(view, labels, expanded=False)
    with ResourceOwner(
        campaign / "training_writer", remaining_bytes=2 * 1024**3, planned_ram_bytes=3 * 1024**3
    ) as owner:
        output.mkdir(parents=True, exist_ok=True)
        normalization_path = output / "NORMALIZATION.json"
        if normalization_path.exists():
            normalization = RateNormalization(**json.loads(normalization_path.read_text()))
            if normalization.training_view_sha256 != view.identity:
                raise ValueError("stored normalization belongs to another TRAIN view")
        else:
            print(f"fitting TRAIN-only normalization over{len(tokens)} rows", flush=True)
            normalization = view.fit_normalization()
            atomic_json(normalization_path, asdict(normalization))
        atomic_json(
            output / "INPUTS.json",
            {
                "training_view_sha256": view.identity,
                "tokens": tokens,
                "excluded_sequences": excluded,
                "normalization": binding(normalization_path),
                "labels_source": binding(labels_path),
                "label_read_scope": (
                    "historical CSV projected to token/sequence/target/bucket; "
                    "fit uses only fold0-TRAIN rows"
                ),
                "supervision_sha256": supervision.identity_sha256,
                "nonselectable": True,
                "outside_phase_support_TRAIN_mass": supervision.outside_support_mass,
            },
        )
        code_files = [Path(__file__), ROOT / "scripts/qa_stage71_fixed_fit.py"] + [
            ROOT / "src/e_jepa_ttc" / name
            for name in (
                "training/stage70_fixed_fit.py",
                "training/stage70_update_ledger.py",
                "training/stage72_supervision.py",
                "training/eap_highres_jepa.py",
                "reproducibility.py",
                "data/event_context128_reader.py",
                "data/event_context128.py",
                "data/stage70_roles.py",
                "models/projective_phase_network.py",
                "models/projective_phase_sweep.py",
                "models/projective_geometry_v11.py",
                "models/phase_distribution_v11.py",
            )
        ]
        recipe = object_digest(
            {
                "code": [binding(path) for path in code_files],
                "inputs": binding(output / "INPUTS.json"),
                "normalization": asdict(normalization),
                "endpoint": 10,
                "warmup": 2,
                "checkpoint_sweep": True,
                "precision": "float32_TF32_off",
                "microbatch": 2,
                "effective_batch": 16,
                "technical_only": True,
            }
        )
        ledger = UpdateLedger(campaign / "UPDATE_LEDGER.json", owner.root)
        results = []
        for arm in ("DIRECT", "PPS-TRAJ"):
            endpoints, profiles = [], []
            for mode in ("continuous", "resumed"):
                deterministic_runtime(7)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                model = ProjectivePhaseNetwork(checkpoint_sweep=True)
                initialize_train_prior(model, supervision)
                model.cuda()
                spec = FitSpec(
                    f"technical_gpu_{arm}_{mode}",
                    "S71_TECHNICAL",
                    arm,
                    0,
                    7,
                    10,
                    recipe,
                    view.identity,
                    True,
                )
                fit = FixedFit(model, spec, ledger, owner, output / arm / mode, warmup=2)
                if ledger.state["jobs"][spec.job_id]["status"] == "COMPLETE":
                    endpoints.append(ledger.state["jobs"][spec.job_id]["endpoint"])
                    del fit, model
                    continue
                fit.resume()
                journal = fit.output / "CONSUMED_TRAIN_TOKENS.jsonl"
                if mode == "resumed" and fit.update < 5:
                    fit.run(
                        loss_callback(model, view, normalization, supervision, arm, journal),
                        stop_after=5,
                    )
                    del fit, model
                    deterministic_runtime(123)
                    model = ProjectivePhaseNetwork(checkpoint_sweep=True).cuda()
                    fit = FixedFit(model, spec, ledger, owner, output / arm / mode, warmup=2)
                    fit.resume()
                fit.run(loss_callback(model, view, normalization, supervision, arm, journal))
                profiles.append(
                    {
                        "mode": mode,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "parameters_with_gradient": sum(
                            p.numel() for p in model.parameters() if p.grad is not None
                        ),
                        "seconds_per_update": [row["seconds"] for row in fit.history],
                    }
                )
                endpoints.append(ledger.state["jobs"][spec.job_id]["endpoint"])
                del fit, model
            states = [
                torch.load(verify(record), map_location="cpu", weights_only=False)
                for record in endpoints
            ]
            equality = {
                key: equal_state(states[0][key], states[1][key])
                for key in ("model", "optimizer", "rng", "update")
            }
            equality["loss_gradient_lr_history"] = equal_state(
                [{k: v for k, v in row.items() if k != "seconds"} for row in states[0]["history"]],
                [{k: v for k, v in row.items() if k != "seconds"} for row in states[1]["history"]],
            )
            results.append(
                {"arm": arm, "equality": equality, "profiles": profiles, "endpoints": endpoints}
            )
            del states
            atomic_json(
                output / "PROGRESS.json", {"results": results, "campaign_totals": ledger.totals()}
            )
        passed = all(all(result["equality"].values()) for result in results)
        receipt = {
            "status": "PASS" if passed else "FAIL",
            "nonselectable": True,
            "scientific_candidate": False,
            "real_TRAIN_smoke": True,
            "recipe_sha256": recipe,
            "results": results,
            "campaign_totals": ledger.totals(),
        }
        atomic_json(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)
        if not passed:
            raise RuntimeError("real TRAIN GPU exact-resume acceptance failed")


if __name__ == "__main__":
    main()
