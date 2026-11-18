"""Fit the GEO_LEARNED head on all eAP TRAIN40 rows (seeds 7, 13, 21) for zero-shot transfer.

``python -m operational.second_benchmark.train_geo --work <geometry work dir> --out <dir>``

Recipe identical to the GEO_LEARNED variant of ``operational/phase_geometry`` (BAND mass,
``fusion_loss``, AdamW 3e-4, wd 1e-3, clip 1.0, batch 128, ``U = ceil(2500*N/18422)``), only on
eAP rows; no FCWD or TSTTC data are read.  CPU only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

SEEDS = (7, 13, 21)
EXPECTED_TRAIN_INPUTS_SHA256 = "1af688cf"  # prefix recorded in geometry/work/DATA_RECEIPT.json


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Train the three seeds and write state dicts plus a receipt."""
    import torch

    from operational.phase_geometry.heads import fit_head, parameter_count, scaled_updates
    from operational.phase_geometry.metrics import ttc_to_phase
    from operational.phase_geometry.variants import VARIANTS, design, training_mass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("artifacts/sota_20261118/geometry/work"))
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/sota_20261118/second_benchmark")
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    source = args.work / "train_inputs.npz"
    if not source.is_file():
        raise SystemExit(f"missing {source}; run operational.phase_geometry.data first")
    source_sha = sha256_file(source)
    if not source_sha.startswith(EXPECTED_TRAIN_INPUTS_SHA256):
        raise SystemExit(f"train_inputs.npz sha256 {source_sha} differs from the geometry receipt")
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    variant = VARIANTS["GEO_LEARNED"]
    rows = np.arange(len(arrays["target_ttc"]))
    x, anchor, width = design(variant, arrays, rows)
    truth = arrays["target_ttc"]
    mass = training_mass(variant, arrays["group_id"], truth)
    target = ttc_to_phase(truth).astype(np.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    fits: list[dict[str, Any]] = []
    for seed in SEEDS:
        started = time.perf_counter()
        fit = fit_head(x, anchor, width, target, mass, seed=seed)
        path = args.out / f"GEO_LEARNED_EAP_ALL_S{seed}.state.pt"
        torch.save(fit.state_dict, path)
        fits.append(
            {
                "seed": seed,
                "updates": fit.updates,
                "loss_mean_last100": fit.final_loss_mean_last100,
                "seconds": time.perf_counter() - started,
                "state_path": path.name,
                "state_sha256": sha256_file(path),
            }
        )
        print(json.dumps(fits[-1]), flush=True)
    receipt = {
        "schema": "second_benchmark_geo_learned_eap_v1",
        "variant": "GEO_LEARNED",
        "train_rows": int(len(rows)),
        "train_sequences": int(len(np.unique(arrays["group_id"]))),
        "expected_updates": scaled_updates(len(rows)),
        "parameters": parameter_count(x.shape[1]),
        "train_inputs_sha256": source_sha,
        "fits": fits,
        "data_read": "eAP TRAIN40 geometry work inputs only; no FCWD/TSTTC rows",
        "device": "cpu",
        "torch": torch.__version__,
    }
    (args.out / "GEO_LEARNED_EAP_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
