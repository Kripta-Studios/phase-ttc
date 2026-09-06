"""Fixed generated translation/scale learnability acceptance:100 counted updates."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.nn import functional

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, object_digest
from e_jepa_ttc.models.projective_phase_network import ContextEncoder
from e_jepa_ttc.training.stage70_fixed_fit import FixedFit, deterministic_runtime
from e_jepa_ttc.training.stage70_update_ledger import FitSpec, UpdateLedger
from e_jepa_ttc.training.stage73_correspondence import TexturePair, _warp, correspondence_loss


def fixture() -> list[TexturePair]:
    """Generate32 fixed band-limited8-channel textures and simple known affine pairs."""
    generator = torch.Generator().manual_seed(20260906)
    mask = torch.ones(128, 128, dtype=torch.bool)
    pairs = []
    for _ in range(32):
        texture = functional.interpolate(
            torch.randn(1, 8, 32, 32, generator=generator),
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )[0]
        transform = torch.eye(3)
        scale = math.exp(float(torch.rand((), generator=generator)) * 0.2 - 0.1)
        transform[0, 0] = transform[1, 1] = scale
        transform[:2, 2] = torch.rand(2, generator=generator) * 0.25 - 0.125
        target, valid = _warp(texture, mask, transform)
        pairs.append(
            TexturePair(texture.cuda(), target.cuda(), mask.cuda(), valid.cuda(), transform.cuda())
        )
    return pairs


@torch.no_grad()
def measure(model: ContextEncoder, pairs: list[TexturePair]) -> dict[str, float]:
    """Evaluate all fixed fixtures with identical query draws at the two fixed endpoints."""
    model.eval()
    generator = torch.Generator().manual_seed(20260907)
    results = [correspondence_loss(model, pair, generator) for pair in pairs]
    if any(result is None for result in results):
        raise ValueError("fixed simple fixture has insufficient known correspondences")
    valid = [result for result in results if result is not None]
    return {
        "cross_entropy": sum(float(result.loss) for result in valid) / len(valid),
        "matching_error_pixels": sum(result.matching_error_pixels for result in valid) / len(valid),
        "constant_descriptor_ce": sum(result.constant_descriptor_ce for result in valid)
        / len(valid),
        "shuffled_correspondence_ce": sum(result.shuffled_correspondence_ce for result in valid)
        / len(valid),
        "descriptor_variance": sum(result.descriptor_variance for result in valid) / len(valid),
    }


def main() -> None:
    """Freeze the fixture before fitting; keep a failure as evidence without a new sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if not campaign.is_relative_to(ROOT / "artifacts"):
        raise ValueError("technical output must remain inside this worktree")
    output = campaign / "technical_correspondence_learning"
    receipt_path = output / "LEARNING_ACCEPTANCE.json"
    if receipt_path.exists():
        raise ValueError("fixed learning acceptance already recorded; no repeat selection")
    torch.set_num_threads(2)
    deterministic_runtime(7)
    with ResourceOwner(
        campaign / "training_writer", remaining_bytes=1024**3, planned_ram_bytes=2 * 1024**3
    ) as owner:
        output.mkdir(parents=True, exist_ok=True)
        pairs = fixture()
        fixture_recipe = {
            "seed": 20260906,
            "textures": 32,
            "base_shape": [8, 32, 32],
            "resize": "bilinear align_corners=False128x128",
            "log_scale": [-0.1, 0.1],
            "translation_normalized": [-0.125, 0.125],
            "rotation": 0,
            "photometric_noise": "none in simple generated acceptance fixture",
            "query_measurement_seed": 20260907,
        }
        source_files = [Path(__file__)] + [
            ROOT / "src/e_jepa_ttc" / name
            for name in (
                "training/stage73_correspondence.py",
                "training/stage70_fixed_fit.py",
                "training/stage70_update_ledger.py",
                "training/eap_highres_jepa.py",
                "models/projective_phase_network.py",
                "reproducibility.py",
            )
        ]
        recipe = {
            "fixture": fixture_recipe,
            "code": [binding(path) for path in source_files],
            "encoder_seed": 7,
            "updates": 100,
            "warmup": 5,
            "microbatch": 2,
            "effective_batch": 16,
            "acceptance_relative_loss_reduction": 0.2,
            "nonselectable": True,
            "uses_TTC": False,
        }
        recipe_hash = object_digest(recipe)
        lock_path = output / "FIXTURE_LOCK.json"
        if lock_path.exists():
            if json.loads(lock_path.read_text()) != recipe:
                raise ValueError("prospective fixture recipe changed")
        else:
            atomic_json(lock_path, recipe)
        # Fixture generation is separate from the registered encoder initialization.
        deterministic_runtime(7)
        model = ContextEncoder().cuda()
        initial_path = output / "INITIAL_DIAGNOSTICS.json"
        if not initial_path.exists():
            atomic_json(initial_path, measure(model, pairs))
        initial = json.loads(initial_path.read_text())
        ledger = UpdateLedger(campaign / "UPDATE_LEDGER.json", owner.root)
        spec = FitSpec(
            "technical_correspondence_learning",
            "S71_TECHNICAL",
            "GENERATED_CORRESPONDENCE",
            0,
            7,
            100,
            recipe_hash,
            object_digest(fixture_recipe),
            True,
        )
        fit = FixedFit(model, spec, ledger, owner, output / "fit", warmup=5, checkpoint_interval=25)
        fit.resume()

        def loss(update: int, index: int, batch: int, generator: torch.Generator) -> torch.Tensor:
            del update, index
            indices = torch.randint(len(pairs), (batch,), generator=generator)
            results = [correspondence_loss(model, pairs[int(i)], generator) for i in indices]
            if any(result is None for result in results):
                raise ValueError("fixed fixture became invalid; do not consume a fake update")
            return torch.stack([result.loss for result in results if result is not None]).mean()

        fit.run(loss)
        final = measure(model, pairs)
        reduction = 1 - final["cross_entropy"] / initial["cross_entropy"]
        passed = reduction >= 0.2
        receipt = {
            "status": "PASS" if passed else "FAIL",
            "nonselectable": True,
            "fixture_lock": binding(lock_path),
            "initial": initial,
            "final": final,
            "relative_loss_reduction": reduction,
            "fixed_threshold": 0.2,
            "endpoint": ledger.state["jobs"][spec.job_id]["endpoint"],
            "campaign_totals": ledger.totals(),
            "does_not_by_itself_certify_A": (
                "real TRAIN texture lifecycle and pretext resume remain required"
            ),
        }
        atomic_json(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
