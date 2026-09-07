"""Apply a verified CPU execution amendment to the unchanged original campaign."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import runpy
import sys
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Literal


def load_source(name: str, path: Path) -> ModuleType:
    """Load an explicitly verified execution implementation under a distinct name."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("execution source cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_overlap(sources: dict[str, Path]) -> None:
    """Replace CPU batching and resource polling only; retain original fit state methods."""
    context = importlib.import_module("e_jepa_ttc.data.event_context128")
    metadata = load_source("_stage70_overlap_context", sources["metadata"])
    context.__dict__["four_context_metadata"] = metadata.four_context_metadata
    augmentation = importlib.import_module("e_jepa_ttc.training.stage72_augmentation")
    optimized_augmentation = load_source("_stage70_overlap_augmentation", sources["augmentation"])
    augmentation.__dict__["augment_observation"] = optimized_augmentation.augment_observation
    architecture = importlib.import_module("e_jepa_ttc.training.stage72_architecture")
    network = importlib.import_module("e_jepa_ttc.models.projective_phase_network")
    # The external module imports this annotation, but its capacity constructor is
    # never installed or called. Preserve the original network class and factory.
    had_capacity = hasattr(network, "Capacity")
    previous_capacity = getattr(network, "Capacity", None)
    network.__dict__["Capacity"] = Literal["2M"]
    try:
        batches = load_source("_stage70_overlap_batches", sources["batches"])
    finally:
        if had_capacity:
            network.__dict__["Capacity"] = previous_capacity
        else:
            delattr(network, "Capacity")
    architecture.__dict__["SupervisedBatches"] = partial(batches.SupervisedBatches, prefetch=True)
    fixed = importlib.import_module("e_jepa_ttc.training.stage70_fixed_fit")
    execution = load_source("_stage70_overlap_fit", sources["fit"])

    def run(self: object, microbatch_loss: object, *, stop_after: int | None = None) -> None:
        try:
            execution.FixedFit.run(self, microbatch_loss, stop_after=stop_after)
        finally:
            if isinstance(microbatch_loss, batches.SupervisedBatches):
                batches.SupervisedBatches.close(microbatch_loss)

    fixed.FixedFit.run = run
    resources = importlib.import_module("e_jepa_ttc.artifacts.campaign_v11")
    owner_init = resources.ResourceOwner.__init__

    def owner(self: object, root: Path, *, remaining_bytes: int, planned_ram_bytes: int) -> None:
        owner_init(
            self,
            root,
            remaining_bytes=remaining_bytes,
            planned_ram_bytes=planned_ram_bytes + 128 * 1024**2,
        )

    resources.ResourceOwner.__init__ = owner


def main() -> None:
    """Verify signed sources/equivalence, then resume with the original scientific lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-amendment", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args, forwarded = parser.parse_known_args()
    amendment = json.loads(args.execution_amendment.read_text(encoding="utf-8"))
    original = Path(amendment["original_worktree"]).resolve(strict=True)
    sys.path.insert(0, str(original / "src"))
    publication = importlib.import_module("e_jepa_ttc.artifacts.risk_geometry_v10")
    signature = amendment.pop("artifact_sha256", None)
    if (
        publication.object_digest(amendment) != signature
        or amendment["scope"] != "CPU_OVERLAP_EXACT_EXECUTION"
        or amendment["scientific_recipe_changed"] is not False
        or amendment["adapter"] != publication.binding(Path(__file__))
    ):
        raise ValueError("execution amendment or adapter identity mismatch")
    root = original / "artifacts/stage70_76_architecture"
    if publication.verify(amendment["training_lock"]) != root / "TRAINING_LOCK.json":
        raise ValueError("execution amendment refers to another original campaign")
    sources = {key: publication.verify(value) for key, value in amendment["sources"].items()}
    for source in amendment["evidence"]:
        publication.verify(source)
    gpu = json.loads(publication.verify(amendment["gpu_equivalence"]).read_text())
    if (
        gpu["status"] != "PASS"
        or gpu["capacity"] != "2M"
        or gpu["prefetch_parity"] is not True
        or len(gpu["results"]) != 4
        or not all(all(row["equality"].values()) for row in gpu["results"])
    ):
        raise ValueError("all original architecture GPU equivalence checks must pass")
    windows_path = publication.verify(amendment["windows_adapter"])
    windows = load_source("_stage70_windows_publication", windows_path)
    legacy = json.loads(publication.verify(amendment["windows_amendment"]).read_text())
    legacy_signature = legacy.pop("artifact_sha256", None)
    if (
        publication.object_digest(legacy) != legacy_signature
        or legacy["adapter"] != publication.binding(windows_path)
        or legacy["scope"] != windows.SCOPE
        or legacy["retry_policy"] != {"maximum_retries": 100, "delay_seconds": 0.1}
    ):
        raise ValueError("original Windows publication amendment changed")
    for source in [legacy["training_lock"], *legacy["tests"], *legacy["incidents"]]:
        publication.verify(source)
    if args.verify_only:
        print(f"Verified execution amendment {signature}; no fit or source data read")
        return
    if "--resume" not in forwarded or "--execute-authorized" not in forwarded:
        parser.error("only the authorized original full-graph resume is supported")
    journal_path = root / "execution_amendments" / f"PUBLICATION_RETRIES_{os.getpid()}.jsonl"

    def journal(value: dict[str, object]) -> None:
        with journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    publication.__dict__["atomic_json"] = windows.guarded_publication(
        publication.atomic_json, root / "UPDATE_LEDGER.json", journal
    )
    install_overlap(sources)
    launcher = original / "scripts/run_scientific_recovery_v10_stage70_76.py"
    os.chdir(original)
    print(
        f"Execution amendment {signature}: one CPU microbatch ahead; original FP32 recipe",
        flush=True,
    )
    sys.argv = [str(launcher), *forwarded]
    runpy.run_path(str(launcher), run_name="__main__")


if __name__ == "__main__":
    main()

