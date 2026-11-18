"""Fit the selected RGB+robust-geometry variant on all rows and package Test12 (NOT submitted).

``python -m operational.geometry_robust.test12``

Same package format as ``operational/phase_geometry/final_package.py``: 6,762 tokens in the
official order (957 aliases expanded), byte-reproducible ZIP, predictions CSV and a receipt.
The comparison with the scored 23.47 package is output disagreement only (no labels).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from operational.phase_geometry import geometry
from operational.phase_geometry.data import FILES, resolved_sources, sha256_file
from operational.phase_geometry.metrics import json_safe
from operational.phase_geometry.package import (
    build_submission,
    disagreement,
    read_zip,
    verify_package,
    write_zip,
)

from . import robust
from .variants import VARIANTS, full_design, select, training_mass

FINAL_SEED = 7
GEOMETRY_WORK = Path("artifacts/sota_20261118/geometry/work")
WORK = Path("artifacts/sota_20261118/geometry_robust/work")
OUT = Path("artifacts/sota_20261118/geometry_robust/test12_candidate")
DOCS = Path("docs/sota_20261118/geometry_robust")
SCORED_ZIP = Path("docs/sota_20261118/geometry/PHASE_GEOMETRY_RGB_GEO_BAND_S7_TRAIN40.zip")
SCORED_ZIP_SHA256 = "8590f1151bb8b8db5baa5ae364675356a47ca93a3ee58139421f3d80d0f831df"
SCORED_OVERALL = 23.4724
CODE = (
    "__init__.py",
    "data.py",
    "evaluate.py",
    "fcwd_eval.py",
    "report.py",
    "robust.py",
    "run_cv.py",
    "test12.py",
    "variants.py",
)


def _npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def fit_all(name: str) -> tuple[Path, dict[str, Any]]:
    """All-row fit of one variant with ``fit_head`` (seed 7), copies appended when augmented."""
    import torch

    from operational.phase_geometry.heads import fit_head, parameter_count, scaled_updates
    from operational.phase_geometry.metrics import ttc_to_phase

    from .run_cv import load_design

    torch.set_num_threads(2)
    variant = VARIANTS[name]
    d = load_design(GEOMETRY_WORK, WORK)
    truth = d["truth"]
    x, anchor, width = select(variant, d["x"], d["anchors"], d["widths"])
    target = truth
    if variant.augmented:
        xa, aa, wa = select(variant, d["x_aug"], d["anchors_aug"], d["widths"])
        x, anchor, width = (
            np.concatenate((x, xa)),
            np.concatenate((anchor, aa)),
            np.concatenate((width, wa)),
        )
        target = np.concatenate((truth, truth))
    fit = fit_head(
        x,
        anchor,
        width,
        ttc_to_phase(target).astype(np.float32),
        training_mass(truth, variant.augmented),
        seed=FINAL_SEED,
        updates=scaled_updates(len(truth)),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}__ALL__S{FINAL_SEED}.state.pt"
    torch.save(fit.state_dict, path)
    record = {
        "variant": name,
        "seed": FINAL_SEED,
        "updates": fit.updates,
        "parameters": parameter_count(len(variant.columns)),
        "inputs": len(variant.columns),
        "train_rows": int(len(truth)),
        "train_population": int(len(target)),
        "loss_mean_last100": fit.final_loss_mean_last100,
        "seconds": fit.seconds,
        "state_sha256": sha256_file(path),
    }
    return path, record


def subset_disagreement(
    candidate: NDArray[Any], reference: NDArray[Any], masks: dict[str, NDArray[np.bool_]]
) -> dict[str, Any]:
    """Output disagreement on label-free subsets of the official rows (NOT accuracy)."""
    out: dict[str, Any] = {}
    for label, mask in masks.items():
        if mask.any():
            out[label] = disagreement(candidate[mask], reference[mask])
        else:
            out[label] = {"rows": 0}
    return out


def main() -> None:
    """Refit, predict, package and compare without any external submission."""
    import torch

    from operational.phase_geometry.heads import predict_head

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    selection = json.loads((DOCS / "SELECTION.json").read_text(encoding="utf-8"))
    name = str(selection["candidate"])
    state_path, record = fit_all(name)
    test = _npz(GEOMETRY_WORK / "test_inputs.npz")
    rob = _npz(WORK / "robust_work.npz")
    x_full, anchors, widths = full_design(
        test["features"], test["rgb_width_logits"], test["geometry_raw"], rob["test_r_short"], None
    )
    x, anchor, width = select(VARIANTS[name], x_full, anchors, widths)
    physical = predict_head(torch.load(state_path, weights_only=True), x, anchor, width)["ttc"]
    position = {str(t): i for i, t in enumerate(test["sample_token"])}
    official = [str(t) for t in test["official_token"]]
    canonical_index = np.asarray([position[str(c)] for c in test["official_canonical"]])
    ttc = physical[canonical_index].astype(np.float64)
    package_name = f"PHASE_GEOMETRY_ROBUST_{name}_S{FINAL_SEED}_TRAIN40"
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_token": official,
            "prediction": ttc,
            "rgb_available": True,
            "fallback_source": name,
        }
    ).to_csv(OUT / "predictions.csv", index=False, lineterminator="\n")
    zip_path = OUT / f"{package_name}.zip"
    write_zip(zip_path, build_submission(official, ttc))
    if not np.array_equal(verify_package(zip_path, official), ttc):
        raise ValueError("packaged predictions differ from computed predictions")
    scored_path = root / SCORED_ZIP
    if sha256_file(scored_path) != SCORED_ZIP_SHA256:
        raise ValueError("the scored 23.47 package differs from its recorded SHA-256")
    scored_payload = read_zip(scored_path)["results"]
    if list(scored_payload) != official:
        raise ValueError("scored package token order differs from the official order")
    scored = np.asarray([float(scored_payload[t]["ttc"]) for t in official])
    from operational.phase_geometry.final_package import predict_test

    replica_physical = predict_test(
        "RGB_GEO_BAND", GEOMETRY_WORK / "jobs" / "RGB_GEO_BAND__ALL__S7__M32.state.pt", test
    )
    replica_gap = float(np.abs(replica_physical[canonical_index] - scored).max())
    g_off = test["official_geometry_raw"]
    trunc = (geometry.raw_column(g_off, "trunc_x_current") > 0) | (
        geometry.raw_column(g_off, "trunc_y_current") > 0
    )
    r_off = rob["test_official_r_short"]
    sat = robust.raw_r_column(r_off, "sat") > 0
    masks = {
        "all": np.ones(len(official), bool),
        "truncated_current": trunc,
        "not_truncated_current": ~trunc,
        "sat": sat,
        "window_not_clean": robust.raw_r_column(r_off, "src_clean") == 0,
        "robust_missing": robust.raw_r_column(r_off, "src_missing") > 0,
    }
    comparison = {
        "scope": "OUTPUT DISAGREEMENT ONLY; NOT ACCURACY; private Test12 labels not read",
        "reference": "scored package 1058935 (Overall 23.47)",
        "reference_sha256": SCORED_ZIP_SHA256,
        "geometry_study_state_reproduces_scored_package_max_abs_ttc": replica_gap,
        "row_counts": {k: int(v.sum()) for k, v in masks.items()},
        "by_subset": subset_disagreement(ttc, scored, masks),
    }
    (DOCS / "TEST12_DISAGREEMENT.json").write_text(
        json.dumps(json_safe(comparison), indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        OUT / "candidate_predictions.npz",
        official_token=np.asarray(official),
        candidate_ttc=ttc,
        scored_ttc=scored,
    )
    shutil.copyfile(zip_path, DOCS / zip_path.name)
    status = str(selection["candidate_status"])
    receipt: dict[str, Any] = {
        "schema": "geometry_robust_test12_candidate_receipt_v1",
        "status": "PACKAGED_NOT_SUBMITTED",
        "candidate_status": status,
        "recommend_test_submission_by_rule": bool(selection["recommend_test_submission_by_rule"]),
        "external_submission": False,
        "private_targets_read": False,
        "candidate": package_name,
        "variant": name,
        "seed": FINAL_SEED,
        **{
            k: record[k]
            for k in ("updates", "parameters", "inputs", "train_rows", "train_population")
        },
        "state_dict_sha256": record["state_sha256"],
        "package": {
            "file_name": zip_path.name,
            "sha256": sha256_file(zip_path),
            "size_bytes": zip_path.stat().st_size,
            "docs_copy": (DOCS / zip_path.name).as_posix(),
        },
        "predictions_csv_sha256": sha256_file(OUT / "predictions.csv"),
        "population": len(official),
        "physical_queries": len(position),
        "aliases": len(official) - len(position),
        "token_order": "official test_inputs.parquet order",
        "finite_and_no_failures": True,
        "recipe": {"file_name": "PROTOCOL.md", "sha256": sha256_file(DOCS / "PROTOCOL.md")},
        "selection_sha256": sha256_file(DOCS / "SELECTION.json"),
        "results_sha256": sha256_file(DOCS / "RESULTS.json"),
        "history_budget": "650 ms causal (600 ms box window), unchanged from 1058935",
        "hypothesis_draft": (
            f"Adding causal robust-geometry inputs (free-edge constant-velocity extrapolation and "
            f"saturation flags) to the RGB_GEO_BAND head changes Test12 Overall MiD relative to "
            f"{SCORED_OVERALL} mainly through queries with truncated boxes (10.0 % of physical "
            f"queries); development-fold change versus RGB_GEO_BAND: "
            f"{selection['candidate_delta_all_vs_RGB_GEO_BAND']:+.2f} MiD "
            f"[{selection['candidate_delta_all_ci95'][0]:+.2f}, "
            f"{selection['candidate_delta_all_ci95'][1]:+.2f}]. The result would be reported "
            f"whatever the score."
        ),
        "work_inputs_sha256": {
            "train_inputs.npz": sha256_file(GEOMETRY_WORK / "train_inputs.npz"),
            "test_inputs.npz": sha256_file(GEOMETRY_WORK / "test_inputs.npz"),
            "robust_work.npz": sha256_file(WORK / "robust_work.npz"),
        },
        "source_members": resolved_sources(),
        "source_files": {k: v["sha256"] for k, v in FILES.items()},
        "code_sha256": {f: sha256_file(Path(__file__).with_name(f)) for f in CODE},
        "upload": {
            "performed_by_this_tool": False,
            "note": "A formal pre-registration receipt (ADR-0047) is required before any upload.",
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    text = json.dumps(json_safe(receipt), indent=2, sort_keys=True) + "\n"
    (OUT / "CANDIDATE_RECEIPT.json").write_text(text, encoding="utf-8")
    (DOCS / "TEST12_CANDIDATE_RECEIPT.json").write_text(text, encoding="utf-8")
    print(
        json.dumps(json_safe({"receipt": receipt["package"], **comparison["row_counts"]}), indent=1)
    )


if __name__ == "__main__":
    main()
