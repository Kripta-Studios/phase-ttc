"""Counted synthetic CPU fit recovery: preserve a corrupt update8 and replay from valid update4."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from qa_stage71_fixed_fit import equal_state, make_loss
from torch import nn

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.training.stage70_fixed_fit import FixedFit, deterministic_runtime
from e_jepa_ttc.training.stage70_update_ledger import FitSpec, UpdateLedger


def main() -> None:
    """Run24 technical optimizer steps; do not read real data or choose a scientific model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if campaign != ROOT / "artifacts/stage70_76_architecture":
        raise ValueError("recovery QA must remain in its owned campaign")
    output = campaign / "technical_cpu_corruption_recovery"
    if output.exists():
        raise ValueError("corruption recovery QA attempt already exists; preserve its evidence")
    output.mkdir()
    code = [
        binding(path)
        for path in (
            Path(__file__),
            ROOT / "src/e_jepa_ttc/training/checkpoint_recovery_v11.py",
            ROOT / "src/e_jepa_ttc/training/stage70_fixed_fit.py",
            ROOT / "src/e_jepa_ttc/training/stage70_update_ledger.py",
        )
    ]
    recipe = object_digest(
        {
            "code": code,
            "technical": True,
            "seed": 7,
            "endpoint": 10,
            "corrupt_update": 8,
            "recover_update": 4,
            "warmup": 2,
            "checkpoint_interval": 4,
        }
    )
    torch.set_num_threads(2)
    endpoints = []
    with ResourceOwner(
        campaign / "training_writer", remaining_bytes=1024**3, planned_ram_bytes=2 * 1024**3
    ) as owner:
        ledger = UpdateLedger(campaign / "UPDATE_LEDGER.json", owner.root)
        before = ledger.totals()
        for suffix in ("continuous", "corrupt_recovery"):
            deterministic_runtime(7)
            model = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Dropout(0.2), nn.Linear(8, 1))
            spec = FitSpec(
                f"technical_cpu_corruption_{suffix}",
                "S71_TECHNICAL",
                "SYNTHETIC_ENGINE",
                0,
                7,
                10,
                recipe,
                object_digest({"synthetic": True, "seed": 7}),
                True,
            )
            fit = FixedFit(
                model, spec, ledger, owner, output / suffix, warmup=2, checkpoint_interval=4
            )
            fit.resume()
            if suffix == "corrupt_recovery":
                fit.run(make_loss(fit.model), stop_after=8)
                path = output / suffix / "checkpoint_000008.pt"
                original_binding = binding(path)
                atomic_json(
                    output / "INJECTED_TECHNICAL_CORRUPTION.json",
                    {
                        "original": original_binding,
                        "purpose": "synthetic checkpoint corruption QA only",
                        "expected_recovery_update": 4,
                    },
                )
                if path.resolve() != path.absolute() or not path.is_relative_to(output):
                    raise ValueError(
                        "injected corruption escaped the explicitly owned technical fixture"
                    )
                path.write_bytes(b"intentional synthetic QA checkpoint corruption")
                deterministic_runtime(123)
                model = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Dropout(0.2), nn.Linear(8, 1))
                fit = FixedFit(
                    model, spec, ledger, owner, output / suffix, warmup=2, checkpoint_interval=4
                )
                fit.resume()
                if fit.update != 4 or ledger.state["jobs"][spec.job_id]["remaining"] != 6:
                    raise ValueError(
                        "corrupt checkpoint did not reserve and restore the latest valid state"
                    )
            fit.run(make_loss(fit.model))
            endpoints.append(ledger.state["jobs"][spec.job_id]["endpoint"])
        states = [
            torch.load(verify(source), map_location="cpu", weights_only=False)
            for source in endpoints
        ]
        equality = {
            key: equal_state(states[0][key], states[1][key])
            for key in ("model", "optimizer", "rng", "update")
        }
        equality["loss_gradient_lr_history"] = equal_state(
            [
                {key: value for key, value in row.items() if key != "seconds"}
                for row in states[0]["history"]
            ],
            [
                {key: value for key, value in row.items() if key != "seconds"}
                for row in states[1]["history"]
            ],
        )
        after = ledger.totals()
        actual = after["confirmed_optimizer_updates"] - before["confirmed_optimizer_updates"]
        if (
            actual != 24
            or after["uncertain_optimizer_attempts"] != before["uncertain_optimizer_attempts"]
        ):
            raise ValueError("corruption replay work was not exactly charged")
        result = {
            "status": "PASS" if all(equality.values()) else "FAIL",
            "nonselectable": True,
            "real_TRAIN_smoke": False,
            "code": code,
            "recipe_sha256": recipe,
            "equality": equality,
            "optimizer_updates": actual,
            "prescribed_model_endpoints": 2,
            "optimizer_instances_constructed": 3,
            "known_replayed_updates": 4,
            "endpoints": endpoints,
            "ledger_before": before,
            "ledger_after": after,
        }
        atomic_json(output / "CORRUPT_CHECKPOINT_RESUME_EQUIVALENCE.json", result)
        print(
            json.dumps(
                {"status": result["status"], "equality": equality, "optimizer_updates": actual}
            ),
            flush=True,
        )
        if not all(equality.values()):
            raise RuntimeError("corruption recovery changed exact continuous-training state")


if __name__ == "__main__":
    main()
