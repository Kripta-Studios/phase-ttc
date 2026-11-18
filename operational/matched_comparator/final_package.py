"""Fit and package A (``GEO_LEARNED``) and, if the pre-registered gates pass, B; label-free only.

``python -m operational.matched_comparator.final_package --geometry-work <dir> --work <dir>
--docs <dir> --artifacts <dir>`` refits on all 88,744 rows (seed 7, U = 12,044, ``fit_head``),
predicts the 5,805 physical Test12 queries, expands the 957 aliases in official order, writes
byte-reproducible ZIPs plus receipts, the Test12 output-disagreement matrix and the additive
(level_gap A3) projections.  Nothing is uploaded or submitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from operational.phase_geometry import geometry
from operational.phase_geometry.metrics import band_index, json_safe, ttc_to_phase
from operational.phase_geometry.package import (
    build_submission,
    disagreement,
    read_zip,
    verify_package,
    write_zip,
)

from . import projection
from .features import design as garl_design
from .features import release_ttc
from .run_cv import load_npz

FINAL_SEED = 7
CODE_FILES = (
    "__init__.py",
    "evaluate.py",
    "feasibility.py",
    "features.py",
    "final_package.py",
    "garl_data.py",
    "garl_runner.py",
    "projection.py",
    "report.py",
    "run_cv.py",
    "run_s8.py",
    "sources.py",
)
GEOMETRY_CODE = ("geometry.py", "heads.py", "metrics.py", "package.py", "variants.py", "weights.py")
SUBMITTED_23 = "docs/sota_20261118/geometry/PHASE_GEOMETRY_RGB_GEO_BAND_S7_TRAIN40.zip"
SUBMITTED_23_SHA = "8590f1151bb8b8db5baa5ae364675356a47ca93a3ee58139421f3d80d0f831df"
FACT_SHEETS = {
    "A": {
        "method_name": "Box-geometry baseline (supplied regions only, 600 ms)",
        "uses_event": False,
        "uses_image": False,
        "note": (
            "Uses neither events nor images: only the benchmark-supplied 2D object regions of "
            "the same track in the causal 600 ms window (expansion slopes, truncation flags, "
            "frame count, size), fed to a 1,635-parameter MLP head trained on TRAIN40."
        ),
    },
    "B_FULL": {
        "method_name": "Garl-TTC + box-geometry head (supplied regions, 600 ms)",
        "uses_event": True,
        "uses_image": True,
        "note": (
            "Released Garl-TTC RGB+event model (paper_ours_full.pth, two observations 0.1 s "
            "apart) per-query prediction plus the same causal 600 ms box-expansion geometry, "
            "fed to a 1,731-parameter MLP head trained on the S8 subsample of TRAIN40 "
            "(11,935 queries; Garl in-sample there)."
        ),
    },
    "B_EVENT": {
        "method_name": "Garl-TTC event-only + box-geometry head (supplied regions, 600 ms)",
        "uses_event": True,
        "uses_image": False,
        "note": (
            "Released Garl-TTC event-only model (paper_event_only_lhr.pth) per-query prediction "
            "plus the same causal 600 ms box-expansion geometry, fed to a 1,731-parameter MLP "
            "head trained on TRAIN40 (Garl in-sample there). Not the 45.03 RGB+event model."
        ),
    },
}
HYPOTHESES = {
    "A": (
        "A head that reads only the supplied object regions of the last 600 ms (no image, no "
        "event model) is scored to quantify how much of 23.47 (RGB_GEO_BAND) is box history "
        "alone; reported whatever the score, including if it is below 23.47."
    ),
    "B": (
        "Giving Garl-TTC (45.03) the same 600 ms box history through the same head, loss and "
        "CV as RGB_GEO_BAND tests whether PHASE-TTC-Geo improves on Garl-TTC given the same "
        "information; reported whatever the score, including if it beats 23.47."
    ),
}


def sha256_file(path: Path) -> str:
    """SHA-256 of a regular file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit_and_predict(
    name: str,
    x_train: NDArray[Any],
    anchor_train: NDArray[Any],
    width_train: NDArray[Any],
    x_test: NDArray[Any],
    anchor_test: NDArray[Any],
    width_test: NDArray[Any],
    truth: NDArray[Any],
    work: Path,
) -> tuple[NDArray[np.float64], dict[str, Any], Path]:
    """All-row seed-7 ``fit_head`` with BAND mass; cached state; returns physical Test12 TTC."""
    import torch

    from operational.phase_geometry.heads import fit_head, parameter_count, predict_head
    from operational.phase_geometry.weights import band_mass

    torch.set_num_threads(2)
    state_path = work / "final" / f"{name}__ALL__S{FINAL_SEED}.state.pt"
    record_path = state_path.with_suffix("").with_suffix(".json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if not state_path.exists():
        fit = fit_head(
            x_train,
            anchor_train,
            width_train,
            ttc_to_phase(truth).astype(np.float32),
            band_mass(truth),
            seed=FINAL_SEED,
        )
        torch.save(fit.state_dict, state_path)
        record_path.write_text(
            json.dumps(
                {
                    "arm": name,
                    "seed": FINAL_SEED,
                    "updates": fit.updates,
                    "parameters": parameter_count(x_train.shape[1]),
                    "inputs": int(x_train.shape[1]),
                    "train_rows": len(truth),
                    "loss_mean_last100": fit.final_loss_mean_last100,
                    "seconds": fit.seconds,
                    "trainer": "fit_head",
                },
                indent=1,
            ),
            "utf-8",
        )
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    ttc = predict_head(state, x_test, anchor_test, width_test)["ttc"]
    return ttc, json.loads(record_path.read_text("utf-8")), state_path


def package(
    label: str,
    name: str,
    physical: NDArray[Any],
    test: dict[str, NDArray[Any]],
    out: Path,
    docs: Path,
) -> tuple[Path, NDArray[np.float64]]:
    """Expand aliases in official order, write and verify the ZIP (also copied to docs)."""
    position = {str(t): i for i, t in enumerate(test["sample_token"])}
    official = [str(t) for t in test["official_token"]]
    ttc = np.asarray(
        [physical[position[str(c)]] for c in test["official_canonical"]], dtype=np.float64
    )
    zip_path = out / f"{name}.zip"
    write_zip(zip_path, build_submission(official, ttc))
    if not np.array_equal(verify_package(zip_path, official), ttc):
        raise ValueError(f"Packaged predictions of {label} differ from computed predictions")
    shutil.copyfile(zip_path, docs / zip_path.name)
    return zip_path, ttc


def official_vector(path: Path, official: list[str]) -> NDArray[np.float64]:
    """Prediction vector of a package ZIP in official token order."""
    results = read_zip(path)["results"]
    if list(results) != official:
        raise ValueError(f"{path.name} token order differs from the official order")
    return np.asarray([float(results[t]["ttc"]) for t in official])


def main() -> None:  # noqa: PLR0915 - linear packaging pipeline
    """Refit, package, compare (output disagreement only) and write receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--geometry-work", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    from operational.phase_geometry.data import verified_file
    from operational.phase_geometry.variants import VARIANTS
    from operational.phase_geometry.variants import design as geometry_design

    root = args.root.resolve()
    gwork, work = args.geometry_work.resolve(), args.work.resolve()
    docs, artifacts = args.docs.resolve(), args.artifacts.resolve()
    out = artifacts / "test12_candidates"
    out.mkdir(parents=True, exist_ok=True)
    results = json.loads((docs / "RESULTS.json").read_text("utf-8"))
    decision = results["decision"]
    train = load_npz(gwork / "train_inputs.npz")
    test = load_npz(gwork / "test_inputs.npz")
    truth = train["target_ttc"]
    official = [str(t) for t in test["official_token"]]
    canonical = [str(t) for t in test["official_canonical"]]
    position = {str(t): i for i, t in enumerate(test["sample_token"])}
    multiplicity = np.bincount(
        [position[c] for c in canonical], minlength=len(test["sample_token"])
    ).astype(np.float64)

    packages: dict[str, dict[str, Any]] = {}
    variant = VARIANTS["GEO_LEARNED"]
    xa, aa, wa = geometry_design(variant, train, slice(None))
    xt, at, wt = geometry_design(variant, test, slice(None))
    phys_a, record_a, state_a = fit_and_predict("GEO_LEARNED", xa, aa, wa, xt, at, wt, truth, work)
    packages["A"] = {"arm": "GEO_LEARNED", "physical": phys_a, "record": record_a, "state": state_a}
    rows_of: dict[str, NDArray[np.int64] | slice] = {"A": slice(None)}
    if decision["package_B"]:
        source = decision["source"]
        arm = decision["arm"]
        if decision["training_rows"] == "S8":
            from .run_s8 import arm_design, combined, s8

            idx = s8(train)
            ev = load_npz(work / "garl_EVENT_heights.npz")
            fu = load_npz(work / "garl_FULL_heights.npz")
            xc, ac, wc = combined(train, ev["train_heights"], fu["train_heights"], idx)
            xb, ab, wb = arm_design(arm, xc, ac, wc)
            xtc, atc, wtc = combined(test, ev["test_heights"], fu["test_heights"], slice(None))
            xbt, abt, wbt = arm_design(arm, xtc, atc, wtc)
            truth_b = truth[idx]
            rows_of["B"] = idx
        else:
            h = load_npz(work / f"garl_{source}_heights.npz")
            phi_train = geometry.raw_column(train["geometry_raw"], "phi_area")
            phi_test = geometry.raw_column(test["geometry_raw"], "phi_area")
            xb, ab, wb = garl_design("GEO_R", train["geometry_g"], h["train_heights"], phi_train)
            xbt, abt, wbt = garl_design("GEO_R", test["geometry_g"], h["test_heights"], phi_test)
            truth_b = truth
            rows_of["B"] = slice(None)
        phys_b, record_b, state_b = fit_and_predict(arm, xb, ab, wb, xbt, abt, wbt, truth_b, work)
        packages["B"] = {"arm": arm, "physical": phys_b, "record": record_b, "state": state_b}

    sets: dict[str, NDArray[np.float64]] = {}
    for label, item in packages.items():
        name = f"MATCHED_{label}_{item['arm']}_S{FINAL_SEED}_TRAIN40"
        zip_path, ttc = package(label, name, item["physical"], test, out, docs)
        item.update(name=name, zip=zip_path, official_ttc=ttc)
        sets[f"PACKAGE_{label}"] = ttc
    submitted23 = root / SUBMITTED_23
    if sha256_file(submitted23) != SUBMITTED_23_SHA:
        raise ValueError("Submitted 23.47 package differs from its receipt")
    sets["SUBMITTED_23_47"] = official_vector(submitted23, official)
    sets["ORIGINAL_35_80"] = official_vector(verified_file("submitted_zip", root), official)
    for source in ("FULL", "EVENT"):
        path = work / f"garl_{source}_heights.npz"
        if path.exists():
            sets[f"GARL_{source}"] = release_ttc(load_npz(path)["official_heights"])

    names = list(sets)
    matrix = []
    for ref in names:
        for cand in names:
            if ref == cand:
                continue
            matrix.append(
                {"reference": ref, "candidate": cand, **disagreement(sets[cand], sets[ref])}
            )
    # Additive (A3) projections anchored at the scored 23.47 package, plus the fold-side inputs.
    from .evaluate import existing_oof
    from .run_cv import SEEDS

    fold_anchor = existing_oof(gwork, "RGB_GEO_BAND", FINAL_SEED, len(truth))
    official_position = {t: i for i, t in enumerate(official)}
    test_anchor = np.asarray(
        [sets["SUBMITTED_23_47"][official_position[str(t)]] for t in test["sample_token"]]
    )
    bands = band_index(truth)
    additive = []
    fold_sets: dict[str, NDArray[np.float64]] = {
        "A": existing_oof(gwork, "GEO_LEARNED", 7, len(truth))
    }
    if "B" in packages:
        from .run_cv import oof
        from .run_s8 import oof_s8

        if decision["training_rows"] == "S8":
            fold_sets["B"] = oof_s8(work, packages["B"]["arm"], FINAL_SEED, len(truth))
        else:
            fold_sets["B"] = oof(work, packages["B"]["arm"], FINAL_SEED, len(truth))
    for label, fold_ttc in fold_sets.items():
        r = rows_of[label]
        additive.append(
            {
                "package": label,
                "arm": packages[label]["arm"],
                "fold_rows": int(len(truth[r])),
                **projection.additive_disagreement(
                    fold_ttc[r],
                    fold_anchor[r],
                    truth[r],
                    bands[r],
                    train["group_id"][r],
                    packages[label]["physical"],
                    test_anchor,
                    multiplicity,
                ),
            }
        )
    descriptive = {}
    for label, fold_ttc in fold_sets.items():
        r = rows_of[label]
        train_dis = float(np.nanmean(projection.phase_mid(fold_ttc[r], fold_anchor[r])))
        test_dis = float(np.nanmean(projection.phase_mid(packages[label]["physical"], test_anchor)))
        descriptive[label] = {
            "train_oof_s7": train_dis,
            "test": test_dis,
            "ratio": test_dis / train_dis,
        }
    if "B" in packages:
        fold_a = fold_sets["A"]
        r = rows_of["B"]
        train_ab = float(np.nanmean(projection.phase_mid(fold_sets["B"][r], fold_a[r])))
        test_ab = float(np.nanmean(projection.phase_mid(packages["B"]["physical"], phys_a)))
        descriptive["B_vs_A"] = {
            "train_oof_s7": train_ab,
            "test": test_ab,
            "ratio": test_ab / train_ab,
        }

    code = {f: sha256_file(Path(__file__).with_name(f)) for f in CODE_FILES}
    geometry_dir = Path(geometry.__file__).parent
    code_geometry = {f: sha256_file(geometry_dir / f) for f in GEOMETRY_CODE}
    protocol_sha = sha256_file(docs / "PROTOCOL.md")
    receipts = {}
    for label, item in packages.items():
        receipt: dict[str, Any] = {
            "schema": "matched_comparator_test12_candidate_receipt_v1",
            "status": "PACKAGED_NOT_SUBMITTED",
            "external_submission": False,
            "private_targets_read": False,
            "package": label,
            "candidate": item["name"],
            "arm": item["arm"],
            "seed": FINAL_SEED,
            **{k: item["record"][k] for k in ("updates", "parameters", "inputs", "train_rows")},
            "trainer": item["record"]["trainer"],
            "loss": "BAND (w_band/n_band, 0.5/0.3/0.1/0.1) fusion_loss",
            "state_dict_sha256": sha256_file(item["state"]),
            "zip_file": item["zip"].name,
            "zip_sha256": sha256_file(item["zip"]),
            "population": len(official),
            "physical_queries": len(position),
            "aliases": len(official) - len(position),
            "token_order": "official test_inputs.parquet order",
            "finite_and_no_failures": True,
            "protocol_sha256": protocol_sha,
            "results_sha256": sha256_file(docs / "RESULTS.json"),
            "work_inputs_sha256": {
                n: sha256_file(gwork / n) for n in ("train_inputs.npz", "test_inputs.npz")
            },
            "code_sha256": code,
            "phase_geometry_code_sha256": code_geometry,
            "fact_sheet": FACT_SHEETS[label if label == "A" else f"B_{decision['source']}"],
            "proposed_hypothesis": HYPOTHESES[label],
        }
        if label == "B":
            receipt["garl_source"] = decision["source"]
            receipt["garl_heights_sha256"] = sha256_file(
                work / f"garl_{decision['source']}_heights.npz"
            )
            receipt["garl_pins"] = results["garl_inputs"]
        receipt["receipt_sha256"] = hashlib.sha256(
            json.dumps(json_safe(receipt), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        receipts[label] = receipt
        (docs / f"TEST12_CANDIDATE_RECEIPT_{label}.json").write_text(
            json.dumps(json_safe(receipt), indent=2, sort_keys=True) + "\n", "utf-8"
        )
    payload = {
        "scope": "OUTPUT DISAGREEMENT ONLY; NOT ACCURACY; private Test12 labels not read",
        "sets": names,
        "matrix": matrix,
        "additive_projection_anchor_RGB_GEO_BAND": additive,
        "disagreement_with_anchor_train_vs_test": descriptive,
        "seeds_used_for_fold_side": [FINAL_SEED],
        "cv_seeds": list(SEEDS),
    }
    (docs / "TEST12_DISAGREEMENT.json").write_text(
        json.dumps(json_safe(payload), indent=2) + "\n", "utf-8"
    )
    print(
        json.dumps(
            json_safe(
                {k: {"zip": v["zip_sha256"], "name": v["candidate"]} for k, v in receipts.items()}
            ),
            indent=1,
        )
    )
    print(json.dumps(json_safe(additive), indent=1)[:3000])


if __name__ == "__main__":
    main()
