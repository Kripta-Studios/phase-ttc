"""Zero-update real-TRAIN GPU QA of nonselectable pretext checkpoint diagnostics."""

# ruff: noqa: E402 -- repository bootstrap
from __future__ import annotations

import argparse
import json
import sys
import time
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
from e_jepa_ttc.evaluation.pretext_diagnostics_v11 import diagnose_pretext_checkpoint
from e_jepa_ttc.training.correspondence_pretext_v11 import PretextTextures
from e_jepa_ttc.training.eap_highres_jepa import _capture_rng_state
from e_jepa_ttc.training.stage70_fixed_fit import deterministic_runtime


def main() -> None:
    """Observe existing technical checkpoints without fitting or selecting a new model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if campaign != ROOT / "artifacts/stage70_76_architecture":
        raise ValueError("technical diagnostics must stay in the current resolved campaign")
    output = campaign / "technical_pretext_checkpoint_diagnostics_recovery"
    if output.exists():
        raise ValueError("technical checkpoint diagnostic attempt already exists")
    ledger_path = campaign / "UPDATE_LEDGER.json"
    ledger_binding = binding(ledger_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    signature = ledger.pop("sha256")
    if signature != object_digest(ledger):
        raise ValueError("optimizer accounting signature differs")
    job = ledger["jobs"]["technical_pretext_continuous"]
    if job["status"] != "COMPLETE" or job["spec"]["technical"] is not True:
        raise ValueError("only the completed nonselectable technical pretext producer is allowed")
    checkpoint = job["endpoint"]
    input_path = campaign / "technical_real_train_gpu/INPUTS.json"
    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    normalization = RateNormalization(**json.loads(verify(inputs["normalization"]).read_text()))
    role_path = campaign / "data_roles/DATA_ROLES.json"
    roles_record = json.loads(role_path.read_text(encoding="utf-8"))
    signature = roles_record.pop("artifact_sha256")
    if object_digest(roles_record) != signature:
        raise ValueError("TRAIN role signature differs")
    values = roles_record["roles"]
    roles = TrainRoles(
        tuple(tuple(pair) for pair in values["sequence_to_group"]),
        tuple(values["original"]),
        tuple(values["historical_exposed"]),
        tuple(values["protected"]),
        tuple(values["confirmation"]),
        tuple(values["expansion"]),
        values["role_prefix"],
    )
    with ResourceOwner(
        campaign / "training_writer", remaining_bytes=1024**3, planned_ram_bytes=2 * 1024**3
    ) as owner:
        reader = CountReader(campaign / "counts_original8192")
        view = TrainingView(
            reader,
            tuple(inputs["tokens"]),
            roles,
            excluded_sequences=tuple(inputs["excluded_sequences"]),
            include_expansion=False,
        )
        if view.identity != inputs["training_view_sha256"]:
            raise ValueError("technical diagnostic view differs from the proven real TRAIN smoke")
        pool = PretextTextures(view, normalization, output / "POOL_UNUSED_TRACE.jsonl")
        deterministic_runtime(7)
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(19)
        before = _capture_rng_state(generator)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        def interrupt_observer() -> None:
            raise RuntimeError("injected pretext observer interruption")

        try:
            diagnose_pretext_checkpoint(
                checkpoint,
                pool,
                seed=7,
                output=output / "endpoint10",
                check=interrupt_observer,
            )
        except RuntimeError as error:
            if "injected pretext observer interruption" not in str(error):
                raise
        else:
            raise AssertionError("observer interruption was not reached")
        if not equal_state(before, _capture_rng_state(generator)):
            raise AssertionError("interrupted pretext observer changed RNG")
        result = diagnose_pretext_checkpoint(
            checkpoint, pool, seed=7, output=output / "endpoint10", check=owner.check
        )
        recovered = diagnose_pretext_checkpoint(
            checkpoint,
            pool,
            seed=7,
            output=output / "endpoint10",
            check=owner.check,
        )
        if recovered != result or result["incomplete_prior_attempts"] != 1:
            raise AssertionError("pretext observer recovery differs or omitted partial attempt")
        unchanged = equal_state(before, _capture_rng_state(generator))
        elapsed = time.perf_counter() - started
        allocated = torch.cuda.max_memory_allocated()
        if allocated >= 0.9 * torch.cuda.get_device_properties(0).total_memory:
            raise RuntimeError("RESOURCE_BLOCKED: diagnostic reached90 percent VRAM")
        if not unchanged or binding(ledger_path) != ledger_binding:
            raise AssertionError("diagnostics changed RNG or optimizer accounting")
        receipt = {
            "kind": "TECHNICAL_PRETEXT_DIAGNOSTIC_GPU_ACCEPTANCE",
            "status": "PASS",
            "selectable": False,
            "optimizer_updates": 0,
            "new_model_fits": 0,
            "rng_unchanged": unchanged,
            "partial_recovery_pass": True,
            "completed_recovery_pass": True,
            "elapsed_seconds": elapsed,
            "peak_allocated_vram_bytes": allocated,
            "gpu_name": torch.cuda.get_device_name(0),
            "ledger_unchanged": ledger_binding,
            "checkpoint": checkpoint,
            "inputs": binding(input_path),
            "roles": binding(role_path),
            "observer_code": binding(ROOT / "src/e_jepa_ttc/evaluation/pretext_diagnostics_v11.py"),
            "runner_code": binding(Path(__file__)),
            "diagnostic_receipt": binding(output / "endpoint10/PRETEXT_DIAGNOSTICS.json"),
            "summary": result["summary"],
        }
        receipt["artifact_sha256"] = object_digest(receipt)
        atomic_json(output / "GPU_ACCEPTANCE.json", receipt)
        print(
            json.dumps({"status": "PASS", "optimizer_updates": 0, "summary": result["summary"]}),
            flush=True,
        )


if __name__ == "__main__":
    main()
