"""Counted nonselectable real-TRAIN correspondence10 versus5+resume5."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from qa_stage71_fixed_fit import equal_state

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.data.event_context128 import RateNormalization
from e_jepa_ttc.data.event_context128_reader import CountReader, TrainingView
from e_jepa_ttc.data.stage70_roles import TrainRoles
from e_jepa_ttc.models.projective_phase_network import ContextEncoder
from e_jepa_ttc.training.correspondence_pretext_v11 import PretextTextures
from e_jepa_ttc.training.stage70_fixed_fit import FixedFit, deterministic_runtime
from e_jepa_ttc.training.stage70_update_ledger import FitSpec, UpdateLedger


def main() -> None:
    """Reuse the verified fold0 TRAIN view and its label-free normalization only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if not campaign.is_relative_to(ROOT / "artifacts"):
        raise ValueError("pretext QA output must remain inside this worktree")
    output = campaign / "technical_pretext_resume"
    receipt_path = output / "PRETEXT_RESUME_EQUIVALENCE.json"
    if receipt_path.exists():
        raise ValueError("pretext QA endpoint already recorded; no repeat selection")
    learning = json.loads(
        (campaign / "technical_correspondence_learning/LEARNING_ACCEPTANCE.json").read_text()
    )
    if learning["status"] != "PASS":
        raise ValueError("fixed generated correspondence acceptance failed")
    inputs_path = campaign / "technical_real_train_gpu/INPUTS.json"
    inputs = json.loads(inputs_path.read_text())
    normalization = RateNormalization(**json.loads(verify(inputs["normalization"]).read_text()))
    record = json.loads((campaign / "data_roles/DATA_ROLES.json").read_text())
    signature = record.pop("artifact_sha256")
    if object_digest(record) != signature:
        raise ValueError("fixed role signature mismatch")
    role = record["roles"]
    roles = TrainRoles(
        tuple((str(pair[0]), str(pair[1])) for pair in role["sequence_to_group"]),
        tuple(role["original"]),
        tuple(role["historical_exposed"]),
        tuple(role["protected"]),
        tuple(role["confirmation"]),
        tuple(role["expansion"]),
        str(role["role_prefix"]),
    )
    reader = CountReader(campaign / "counts_original8192")
    view = TrainingView(
        reader,
        tuple(inputs["tokens"]),
        roles,
        excluded_sequences=tuple(inputs["excluded_sequences"]),
        include_expansion=False,
    )
    if view.identity != inputs["training_view_sha256"]:
        raise ValueError("reconstructed TRAIN view differs from verified smoke input")
    sources = [Path(__file__)] + [
        ROOT / "src/e_jepa_ttc" / name
        for name in (
            "training/correspondence_pretext_v11.py",
            "training/stage73_correspondence.py",
            "training/stage70_fixed_fit.py",
            "training/stage70_update_ledger.py",
            "training/eap_highres_jepa.py",
            "models/projective_phase_network.py",
            "reproducibility.py",
            "data/event_context128.py",
            "data/event_context128_reader.py",
            "data/stage70_roles.py",
        )
    ]
    recipe = object_digest(
        {
            "sources": [binding(path) for path in sources],
            "inputs": binding(inputs_path),
            "updates": 10,
            "warmup": 2,
            "technical_only": True,
            "encoder_only": True,
        }
    )
    torch.set_num_threads(2)
    endpoints, traces = [], []
    with ResourceOwner(
        campaign / "training_writer", remaining_bytes=1024**3, planned_ram_bytes=2 * 1024**3
    ) as owner:
        ledger = UpdateLedger(campaign / "UPDATE_LEDGER.json", owner.root)
        for mode in ("continuous", "resumed"):
            deterministic_runtime(7)
            model = ContextEncoder().cuda()
            pool = PretextTextures(view, normalization, output / mode / "TEXTURE_TRACE.jsonl")
            spec = FitSpec(
                f"technical_pretext_{mode}",
                "S71_TECHNICAL",
                "CORRESPONDENCE",
                0,
                7,
                10,
                recipe,
                pool.identity,
                True,
            )
            fit = FixedFit(model, spec, ledger, owner, output / mode, warmup=2)
            if ledger.state["jobs"][spec.job_id]["status"] == "COMPLETE":
                endpoints.append(ledger.state["jobs"][spec.job_id]["endpoint"])
                traces.append(binding(pool.journal))
                del fit, model
                continue
            fit.resume()

            def objective(
                update: int,
                index: int,
                batch: int,
                generator: torch.Generator,
                *,
                bound_pool: PretextTextures = pool,
                bound_model: ContextEncoder = model,
            ) -> torch.Tensor:
                return bound_pool.objective(
                    bound_model,
                    update=update,
                    microbatch_index=index,
                    batch=batch,
                    generator=generator,
                )

            if mode == "resumed" and fit.update < 5:
                fit.run(objective, stop_after=5)
                # Reconstruct model and optimizer; closure is rebound after resume below.
                del objective, fit, model
                deterministic_runtime(123)
                model = ContextEncoder().cuda()
                fit = FixedFit(model, spec, ledger, owner, output / mode, warmup=2)
                fit.resume()

                def objective(
                    update: int,
                    index: int,
                    batch: int,
                    generator: torch.Generator,
                    *,
                    bound_pool: PretextTextures = pool,
                    bound_model: ContextEncoder = model,
                ) -> torch.Tensor:
                    return bound_pool.objective(
                        bound_model,
                        update=update,
                        microbatch_index=index,
                        batch=batch,
                        generator=generator,
                    )

            fit.run(objective)
            endpoints.append(ledger.state["jobs"][spec.job_id]["endpoint"])
            traces.append(binding(pool.journal))
            del objective, fit, model
        states = [
            torch.load(verify(endpoint), map_location="cpu", weights_only=False)
            for endpoint in endpoints
        ]
        equality = {
            key: equal_state(states[0][key], states[1][key])
            for key in ("model", "optimizer", "rng", "update")
        }
        equality["loss_gradient_lr_history"] = equal_state(
            [{k: v for k, v in row.items() if k != "seconds"} for row in states[0]["history"]],
            [{k: v for k, v in row.items() if k != "seconds"} for row in states[1]["history"]],
        )
        equality["texture_augmentation_diagnostics_trace"] = (
            traces[0]["sha256"] == traces[1]["sha256"]
        )
        receipt = {
            "status": "PASS" if all(equality.values()) else "FAIL",
            "nonselectable": True,
            "encoder_only": True,
            "TTC_labels_read_by_this_script": False,
            "train_tokens": len(pool.train_tokens),
            "diagnostic_tokens": len(pool.held_tokens),
            "pretext_pool_sha256": pool.identity,
            "equality": equality,
            "endpoints": endpoints,
            "traces": traces,
            "campaign_totals": ledger.totals(),
        }
        atomic_json(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)
        if not all(equality.values()):
            raise RuntimeError("real TRAIN correspondence exact resume failed")


if __name__ == "__main__":
    main()
