"""Counted nonselectable synthetic CPU training:10 continuous versus5+resume5."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch import nn

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest, verify
from e_jepa_ttc.training.stage70_fixed_fit import FixedFit, deterministic_runtime
from e_jepa_ttc.training.stage70_update_ledger import FitSpec, UpdateLedger


def equal_state(a: object, b: object) -> bool:
    """Compare complete nested numerical/RNG payloads without serialization-byte assumptions."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict) and a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
        )
    if isinstance(a, (tuple, list)):
        if not isinstance(b, (tuple, list)):
            return False
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(equal_state(x, y) for x, y in zip(a, b, strict=True))
        )
    return bool(a == b)


def make_loss(model: nn.Module) -> Callable[[int, int, int, torch.Generator], torch.Tensor]:
    """Bind the currently constructed model without retaining an uncaptured sampler cursor."""

    def loss(update: int, index: int, batch: int, generator: torch.Generator) -> torch.Tensor:
        del update, index
        inputs = torch.randn(batch, 4, generator=generator)
        target = (
            inputs.sum(-1, keepdim=True) * float(np.random.uniform(0.9, 1.1))
            + random.random() * 0.01
        )
        return (model(inputs) - target).square().mean()

    return loss


def main() -> None:
    """Execute two budgeted technical endpoints; no real data or scientific scores are read."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if not campaign.is_relative_to(ROOT / "artifacts"):
        raise ValueError("QA output must remain in the current worktree artifacts")
    output = campaign / "technical_cpu_resume"
    receipt_path = output / "CPU_RESUME_EQUIVALENCE.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        for endpoint in receipt["endpoints"]:
            verify(endpoint)
        print(json.dumps(receipt), flush=True)
        return
    recipe = object_digest(
        {
            "code": [
                binding(path)
                for path in (
                    Path(__file__),
                    ROOT / "src/e_jepa_ttc/training/stage70_fixed_fit.py",
                    ROOT / "src/e_jepa_ttc/training/stage70_update_ledger.py",
                    ROOT / "src/e_jepa_ttc/training/eap_highres_jepa.py",
                    ROOT / "src/e_jepa_ttc/reproducibility.py",
                )
            ],
            "technical": True,
            "architecture": "synthetic4→8→1_dropout0.2",
            "endpoint": 10,
            "warmup": 2,
            "effective_batch": 16,
            "microbatch": 2,
        }
    )
    torch.set_num_threads(2)
    endpoints = []
    with ResourceOwner(
        campaign / "training_writer", remaining_bytes=1024**3, planned_ram_bytes=2 * 1024**3
    ) as owner:
        ledger = UpdateLedger(campaign / "UPDATE_LEDGER.json", owner.root)
        for suffix in ("continuous", "resumed"):
            deterministic_runtime(7)
            model = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Dropout(0.2), nn.Linear(8, 1))
            spec = FitSpec(
                f"technical_cpu_{suffix}",
                "S71_TECHNICAL",
                "SYNTHETIC_ENGINE",
                0,
                7,
                10,
                recipe,
                object_digest({"synthetic": True, "seed": 7}),
                True,
            )
            fit = FixedFit(model, spec, ledger, owner, output / suffix, warmup=2)
            if ledger.state["jobs"][spec.job_id]["status"] == "COMPLETE":
                endpoints.append(ledger.state["jobs"][spec.job_id]["endpoint"])
                continue
            fit.resume()

            if suffix == "resumed" and fit.update < 5:
                fit.run(make_loss(fit.model), stop_after=5)
                deterministic_runtime(123)  # Deliberately disturb state before reconstructing.
                model = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Dropout(0.2), nn.Linear(8, 1))
                fit = FixedFit(model, spec, ledger, owner, output / suffix, warmup=2)
                fit.resume()
            fit.run(make_loss(fit.model))
            endpoints.append(ledger.state["jobs"][spec.job_id]["endpoint"])
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
        receipt = {
            "status": "PASS" if all(equality.values()) else "FAIL",
            "nonselectable": True,
            "real_TRAIN_smoke": False,
            "recipe_sha256": recipe,
            "equality": equality,
            "endpoints": endpoints,
            "campaign_totals": ledger.totals(),
        }
        atomic_json(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)
        if not all(equality.values()):
            raise RuntimeError("synthetic exact-resume acceptance failed")


if __name__ == "__main__":
    main()
