"""Recover three sensor-only states excluded by the obsolete target prefilter."""

from pathlib import Path

import numpy as np
import torch

from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_robust.bundle import verified_upstream_paths
from scripts.final_train40_transfer import HEAD, FinalTransferPredictor, sha
from scripts.run_scientific_revision import read


def main() -> None:
    """Use unchanged frozen weights; never open TTC labels during inference."""
    root = Path(".")
    base = root / "output/phase_adapt_final/data_v3"
    destination = root / "output/scientific_revision_20261111/recovered_features"
    wanted = {
        "EXTENDED:CPNAO-low:25550860",
        "EXTENDED:CCRm-medium-0-overlap-0:18000463",
        "EXTENDED:CCRs-1-medium-100-overlap-100:13350546",
    }
    bundle = root / "output/phase_submission/FINAL_MODEL/upstream"
    paths = verified_upstream_paths(bundle)
    binding = read(bundle / "BUNDLE.json")["parent"]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    predictor = FinalTransferPredictor(paths, "cuda")
    captured = {}

    def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["features"] = inputs[0].detach().cpu().numpy().copy()
        captured["rgb_width_logits"] = inputs[1].detach().cpu().numpy().copy()

    hook = predictor.models[HEAD].register_forward_pre_hook(capture)
    rows = [r for r in read(base / "POPULATION.json")["rows"] if r["query_id"] in wanted]
    if len(rows) != 3:
        raise ValueError("Missing sensor metadata")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for i, row in enumerate(rows):
            captured.clear()
            sensor = {k: v for k, v in row.items() if k != "ttc_csv_path"}
            prediction = predictor.predict(sensor)
            if not captured or not np.isfinite(captured["features"]).all():
                raise ValueError("Missing or nonfinite captured state")
            path = destination / f"query_{i}.npz"
            save_npz(path, captured)
            atomic_receipt(
                path.with_suffix(".json"),
                {
                    "query_id": row["query_id"],
                    "feature_sha256": sha(path),
                    "targets_read_during_inference": False,
                    "prediction": prediction,
                    "candidate": binding,
                    "source_sha256": sha(Path(__file__)),
                },
            )
            print(row["query_id"], flush=True)
    finally:
        hook.remove()


if __name__ == "__main__":
    main()
