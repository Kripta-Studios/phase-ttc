"""Bounded execution-only chunk8/32/64 parity/timing probe; zero optimizer updates."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from e_jepa_ttc.artifacts.campaign_v11 import ResourceOwner
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, verify
from e_jepa_ttc.data.event_context128 import RateNormalization, four_contexts
from e_jepa_ttc.data.event_context128_reader import CountReader
from e_jepa_ttc.models.projective_phase_network import ProjectivePhaseNetwork
from e_jepa_ttc.training.stage70_fixed_fit import deterministic_runtime


def main() -> None:
    """Use only a nonselectable technical checkpoint and two previously permitted TRAIN rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if not campaign.is_relative_to(ROOT / "artifacts"):
        raise ValueError("execution probe must remain in this worktree")
    output = campaign / "sweep_chunk_profile"
    if (output / "SWEEP_CHUNK_PARITY.json").exists():
        raise ValueError("bounded execution probe already recorded")
    qa = json.loads(
        (campaign / "technical_real_train_gpu/REAL_TRAIN_GPU_EQUIVALENCE.json").read_text()
    )
    endpoint = next(row for row in qa["results"] if row["arm"] == "PPS-TRAJ")["endpoints"][0]
    inputs = json.loads((campaign / "technical_real_train_gpu/INPUTS.json").read_text())
    normalization = RateNormalization(**json.loads(verify(inputs["normalization"]).read_text()))
    torch.set_num_threads(2)
    deterministic_runtime(7)
    with ResourceOwner(
        campaign / "training_writer", remaining_bytes=128 * 1024**2, planned_ram_bytes=3 * 1024**3
    ):
        output.mkdir(parents=True, exist_ok=True)
        reader = CountReader(campaign / "counts_original8192")
        tokens = inputs["tokens"][:2]
        if not set(tokens) <= set(inputs["tokens"]):
            raise ValueError("profile rows outside the permitted TRAIN view")
        samples = []
        for token in tokens:
            row = reader.observation(token)
            samples.append(
                four_contexts(
                    row.counts,
                    row.durations_s,
                    row.support,
                    row.midpoint_twice_relative_us,
                    normalization,
                )
            )
        contexts = torch.stack([sample.contexts for sample in samples]).cuda()
        times = torch.stack([sample.times for sample in samples]).cuda()
        payload = torch.load(verify(endpoint), map_location="cpu", weights_only=False)
        records, reference_output, reference_gradients = [], None, None
        for chunk in (8, 32, 64):
            torch.cuda.empty_cache()
            model = ProjectivePhaseNetwork(sweep_chunk=chunk, checkpoint_sweep=True).cuda()
            model.load_state_dict(payload["model"], strict=True)
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for repeat in range(4):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                prediction = model(contexts, times, arm="PPS-TRAJ")
                prediction["logits"].square().mean().backward()
                torch.cuda.synchronize()
                if repeat:
                    durations.append(time.perf_counter() - started)
            actual_output = {key: value.detach().cpu() for key, value in prediction.items()}
            gradients = {
                name: parameter.grad.detach().cpu()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            if len(gradients) != len(list(model.parameters())) or not all(
                bool(value.isfinite().all()) for value in gradients.values()
            ):
                raise ValueError("execution probe lost or invalidated model gradients")
            passed = True
            maximum_gradient_difference = 0.0
            if reference_output is None:
                reference_output, reference_gradients = actual_output, gradients
            else:
                assert reference_gradients is not None
                try:
                    for key in reference_output:
                        torch.testing.assert_close(
                            actual_output[key], reference_output[key], rtol=1e-5, atol=1e-6
                        )
                    for key in reference_gradients:
                        maximum_gradient_difference = max(
                            maximum_gradient_difference,
                            float((gradients[key] - reference_gradients[key]).abs().max()),
                        )
                        torch.testing.assert_close(
                            gradients[key], reference_gradients[key], rtol=1e-4, atol=1e-6
                        )
                except AssertionError:
                    passed = False
            peak = torch.cuda.max_memory_allocated()
            memory_ok = peak < 0.9 * torch.cuda.get_device_properties(0).total_memory
            record = {
                "chunk": chunk,
                "value_gradient_parity": passed,
                "memory_margin_passes": memory_ok,
                "seconds_per_microbatch": durations,
                "median_seconds": statistics.median(durations),
                "peak_allocated_bytes": peak,
                "maximum_gradient_difference": maximum_gradient_difference,
            }
            records.append(record)
            atomic_json(output / "PROGRESS.json", records)
            print(json.dumps(record), flush=True)
            del model, prediction, gradients, actual_output
        eligible = [
            record
            for record in records
            if record["value_gradient_parity"] and record["memory_margin_passes"]
        ]
        recommended = min(eligible, key=lambda record: record["median_seconds"])["chunk"]
        atomic_json(
            output / "SWEEP_CHUNK_PARITY.json",
            {
                "status": "EXECUTION_PROBE_COMPLETE",
                "nonselectable": True,
                "optimizer_updates": 0,
                "technical_checkpoint": endpoint,
                "source_code": binding(Path(__file__)),
                "TRAIN_tokens": tokens,
                "physical_microbatch": 2,
                "effective_batch_unchanged": 16,
                "checkpoint_sweep": True,
                "output_tolerance": {"rtol": 1e-5, "atol": 1e-6},
                "gradient_tolerance": {"rtol": 1e-4, "atol": 1e-6},
                "records": records,
                "recommended_chunk": recommended,
                "requires_before_scientific_use": (
                    "same-environment exact resume at the chosen chunk if changed"
                ),
            },
        )


if __name__ == "__main__":
    main()
