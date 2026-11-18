"""Fit the pre-registered primary on all 40 sequences and package label-free Test12 predictions.

``python -m operational.phase_geometry.final_package --work <dir> --docs <dir> --artifacts <dir>``
reads ``SELECTION.json`` (written by ``evaluate``), refits the primary with seed 7, predicts the
5,805 physical Test12 queries, expands the 957 aliases in official order, writes the candidate
ZIP plus a receipt and compares it with the submitted package as output disagreement only.
Nothing is uploaded or submitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import FILES, SOURCES, sha256_file, verified_file
from .metrics import json_safe
from .package import build_submission, disagreement, read_zip, verify_package, write_zip
from .run_cv import Job, run_job
from .variants import VARIANTS, design, input_count

FINAL_SEED = 7
CODE_FILES = (
    "__init__.py",
    "data.py",
    "evaluate.py",
    "final_package.py",
    "geometry.py",
    "heads.py",
    "metrics.py",
    "package.py",
    "report.py",
    "run_cv.py",
    "variants.py",
    "weights.py",
)


def _npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def predict_test(variant: str, state_path: Path, test: dict[str, Any]) -> np.ndarray:
    """Primary head predictions (TTC seconds) for the physical Test12 queries."""
    import torch

    from .heads import predict_head

    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    x, anchor, width = design(VARIANTS[variant], test, slice(None))
    return predict_head(state, x, anchor, width)["ttc"]


def main() -> None:
    """Refit, predict, package and compare without any external submission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    work, docs, artifacts = args.work.resolve(), args.docs.resolve(), args.artifacts.resolve()
    selection = json.loads((docs / "SELECTION.json").read_text(encoding="utf-8"))
    primary = str(selection["primary"])
    job = Job(primary, -1, FINAL_SEED)
    state_path = work / "jobs" / f"{job.job_id}.state.pt"
    if not state_path.exists():
        run_job(job, work)
    fit_record = json.loads((work / "jobs" / f"{job.job_id}.json").read_text(encoding="utf-8"))
    test = _npz(work / "test_inputs.npz")
    physical = predict_test(primary, state_path, test)
    position = {str(t): i for i, t in enumerate(test["sample_token"])}
    official = [str(t) for t in test["official_token"]]
    canonical = [str(t) for t in test["official_canonical"]]
    ttc = np.asarray([physical[position[c]] for c in canonical], dtype=np.float64)
    name = f"PHASE_GEOMETRY_{primary}_S{FINAL_SEED}_TRAIN40"
    out = artifacts / "test12_candidate"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "sample_token": official,
            "prediction": ttc,
            "rgb_available": True,
            "fallback_source": primary,
        }
    )
    frame.to_csv(out / "predictions.csv", index=False, lineterminator="\n")
    zip_path = out / f"{name}.zip"
    write_zip(zip_path, build_submission(official, ttc))
    verified = verify_package(zip_path, official)
    if not np.array_equal(verified, ttc):
        raise ValueError("Packaged predictions differ from computed predictions")
    submitted_path = verified_file("submitted_zip", root)
    submitted_payload = read_zip(submitted_path)["results"]
    if list(submitted_payload) != official:
        raise ValueError("Submitted package token order differs from official order")
    submitted = np.asarray([float(submitted_payload[t]["ttc"]) for t in official])
    replica_state = work / "jobs" / "REL__ALL__S7__M32.state.pt"
    replica_physical = predict_test("REL", replica_state, test)
    replica = np.asarray([replica_physical[position[c]] for c in canonical])
    captured = np.asarray([test["released_ttc"][position[c]] for c in canonical])
    released_physical = predict_test("REL", verified_file("released_checkpoint", root), test)
    released = np.asarray([released_physical[position[c]] for c in canonical])
    comparisons = {
        "candidate_vs_submitted": disagreement(ttc, submitted),
        "rel_cpu_replica_vs_submitted": disagreement(replica, submitted),
        "released_checkpoint_cpu_vs_submitted_max_abs_ttc": float(
            np.abs(released - submitted).max()
        ),
        "captured_diagnostic_output_vs_submitted_max_abs_ttc": float(
            np.abs(captured - submitted).max()
        ),
        "scope": "OUTPUT DISAGREEMENT ONLY; NOT ACCURACY; private Test12 labels not read",
    }
    (docs / "TEST12_DISAGREEMENT.json").write_text(
        json.dumps(json_safe(comparisons), indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        out / "candidate_predictions.npz",
        official_token=np.asarray(official),
        candidate_ttc=ttc,
        submitted_ttc=submitted,
        rel_replica_ttc=replica,
    )
    code = {f: sha256_file(Path(__file__).with_name(f)) for f in CODE_FILES}
    receipt: dict[str, Any] = {
        "schema": "phase_geometry_test12_candidate_receipt_v1",
        "status": "PACKAGED_NOT_SUBMITTED",
        "external_submission": False,
        "private_targets_read": False,
        "candidate": name,
        "primary_variant": primary,
        "selection_sha256": sha256_file(docs / "SELECTION.json"),
        "protocol_sha256": sha256_file(docs / "PROTOCOL.json"),
        "seed": FINAL_SEED,
        "updates": fit_record["updates"],
        "parameters": fit_record["parameters"],
        "inputs": input_count(VARIANTS[primary]),
        "train_rows": fit_record["train_rows"],
        "state_dict_sha256": sha256_file(state_path),
        "zip_path": str(zip_path.relative_to(root))
        if zip_path.is_relative_to(root)
        else str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "predictions_csv_sha256": sha256_file(out / "predictions.csv"),
        "population": len(official),
        "physical_queries": len(position),
        "aliases": len(official) - len(position),
        "token_order": "official test_inputs.parquet order",
        "finite_and_no_failures": True,
        "work_inputs_sha256": {
            n: sha256_file(work / n) for n in ("train_inputs.npz", "test_inputs.npz")
        },
        "source_members": SOURCES,
        "source_files": {k: v["sha256"] for k, v in FILES.items()},
        "code_sha256": code,
        "submitted_zip_sha256": sha256_file(submitted_path),
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (out / "CANDIDATE_RECEIPT.json").write_text(
        json.dumps(json_safe(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (docs / "TEST12_CANDIDATE_RECEIPT.json").write_text(
        json.dumps(json_safe(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            json_safe(
                {
                    "receipt": {k: receipt[k] for k in ("candidate", "zip_sha256", "population")},
                    **comparisons,
                }
            ),
            indent=1,
        )[:4000]
    )


if __name__ == "__main__":
    main()
