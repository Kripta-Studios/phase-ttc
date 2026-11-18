"""OOF evaluation on eAP TRAIN40: strata, paired sequence-bootstrap CIs, decision and Test12 gate.

``python -m operational.saturation_scale.evaluate --work <dir> --docs <dir>``

Only eAP TRAIN40 rows are read (PROTOCOL.md §4); FCWD and Test12 are never read here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from operational.geometry_robust import robust
from operational.geometry_robust.evaluate import bootstrap, stratum_overall
from operational.phase_geometry import geometry
from operational.phase_geometry.metrics import BANDS, band_index, json_safe, row_mid, ttc_to_phase
from operational.second_benchmark.fcwd import phase_ttc

from . import content_scale as cs
from .run_cv import FOLDS, GEOMETRY_WORK, ROBUST_WORK, SEEDS, load_design
from .variants import (
    ARMS,
    BASELINE,
    GATE_ARMS,
    GATE_COMPARATOR,
    LEARNED,
    PRIMARY,
    content_fixed_phase,
)

BOOTSTRAP_SEED = 20261118
BOOTSTRAP_REPLICATES = 10_000
MARGIN_MID = 0.5
GATE_UPPER_MID = -2.0
REFERENCES = (BASELINE, "RGB_GEO_BAND", "GEO_LEARNED")
FIXED = ("CONTENT_FIXED", "ROBUST_FIXED")
REPLICA_ARMS = ("GEO_ROBUST_AUG", "GEO_LEARNED", "RGB_GEO_BAND", "RGB_ROBUST")


def _npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_predictions(jobs: Path, names: tuple[str, ...], rows: int) -> dict[str, dict[str, Any]]:
    """``{arm: {"orig": [S, N], "aug": [S, N]}}`` TTC predictions from CV job files."""
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        orig = np.full((len(SEEDS), rows), np.nan)
        aug = np.full((len(SEEDS), rows), np.nan)
        for s_index, seed in enumerate(SEEDS):
            for fold in FOLDS:
                job = _npz(jobs / f"{name}__F{fold}__S{seed}.npz")
                orig[s_index, job["rows"]] = job["ttc_orig"]
                aug[s_index, job["rows"]] = job["ttc_aug"]
        if not (np.isfinite(orig).all() and np.isfinite(aug).all()):
            raise ValueError(f"{name}: missing OOF predictions")
        out[name] = {"orig": orig, "aug": aug}
    return out


def rank_correlation(a: NDArray[Any], b: NDArray[Any]) -> float:
    """Spearman correlation (average ranks are not needed: ties are negligible here)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def signal_diagnostics(
    truth: NDArray[Any],
    masks: dict[str, NDArray[np.bool_]],
    sources: dict[str, dict[str, NDArray[Any]]],
) -> dict[str, Any]:
    """Descriptive: rank correlation of fixed phases with the target phase per stratum."""
    target = ttc_to_phase(truth)
    out: dict[str, Any] = {}
    for stratum, mask in masks.items():
        tag = "aug" if stratum.startswith("SYN") else "orig"
        valid = mask & (sources["valid"][tag] > 0)
        out[stratum] = {
            "rows": int(mask.sum()),
            "content_valid_rows": int(valid.sum()),
            "spearman_content_vs_target_on_valid": rank_correlation(
                sources["phi_content"][tag][valid], target[valid]
            ),
            "spearman_robust_vs_target_on_valid": rank_correlation(
                sources["phi_robust"][tag][valid], target[valid]
            ),
            "spearman_robust_vs_target_all": rank_correlation(
                sources["phi_robust"][tag][mask], target[mask]
            ),
        }
    return out


def replica_check(predictions: dict[str, Any], truth: NDArray[Any], root: Path) -> dict[str, Any]:
    """Descriptive: refits against the OOF jobs of ``geometry_robust``."""
    old = load_predictions(root / ROBUST_WORK / "jobs", REPLICA_ARMS, len(truth))
    bands = band_index(truth)
    out: dict[str, Any] = {}
    for name in REPLICA_ARMS:
        a, b = old[name]["orig"], predictions[name]["orig"]
        out[name] = {
            "max_abs_ttc_difference": float(np.abs(a - b).max()),
            "median_abs_ttc_difference": float(np.median(np.abs(a - b))),
            "overall_geometry_robust": stratum_overall(
                np.mean([row_mid(truth, p) for p in a], 0), bands
            )["overall"],
            "overall_refit": stratum_overall(np.mean([row_mid(truth, p) for p in b], 0), bands)[
                "overall"
            ],
        }
    return out


def official_checks(truth: NDArray[Any], predictions: dict, root: Path) -> dict[str, Any]:
    """Max |official - vectorized| Overall over every learned arm and seed (ALL rows)."""
    from operational.garl_comparison.mid import load_scorer
    from operational.phase_geometry.data import verified_file
    from operational.phase_geometry.metrics import official_check

    scorer = load_scorer(verified_file("official_scorer", root))
    diffs = [
        official_check(scorer, truth, predictions[n]["orig"][s])
        for n in LEARNED
        for s in range(len(SEEDS))
    ]
    if max(diffs) > 1e-6:
        raise ValueError(f"vectorized Overall differs from the official scorer by {max(diffs)}")
    return {"checks": len(diffs), "max_abs_overall_difference": float(max(diffs))}


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    """Evaluate every arm and write RESULTS.json, STRATA.csv, DELTAS.csv and DECISION.json."""
    from operational.phase_geometry.data import sha256_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    work, docs = args.work.resolve(), args.docs.resolve()
    d = load_design(work, root)
    train = _npz(root / GEOMETRY_WORK / "train_inputs.npz")
    rob = _npz(root / ROBUST_WORK / "robust_work.npz")
    content = _npz(work / "content_work.npz")
    truth = d["truth"]
    bands = band_index(truth)
    sequences, clusters = np.unique(d["groups"].astype(str), return_inverse=True)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(sequences), (BOOTSTRAP_REPLICATES, len(sequences))
    )
    g_raw = train["geometry_raw"]
    trunc = (geometry.raw_column(g_raw, "trunc_x_current") > 0) | (
        geometry.raw_column(g_raw, "trunc_y_current") > 0
    )
    sat = robust.raw_r_column(rob["r_short"], "sat") > 0
    sat_aug = robust.raw_r_column(rob["aug_r_short"], "sat") > 0
    c_valid = cs.raw_c_column(content["c_orig"], "valid") > 0
    c_valid_aug = cs.raw_c_column(content["c_aug"], "valid") > 0
    everything = np.ones(len(truth), bool)
    masks = {
        "ALL": everything,
        "TRUNC": trunc,
        "NOTRUNC": ~trunc,
        "SAT": sat,
        "NOSAT": ~sat,
        "C_VALID": c_valid,
        "C_INVALID": ~c_valid,
        "SYN_ALL": everything,
        "SYN_SAT": sat_aug,
        "SYN_NOSAT": ~sat_aug,
        "SYN_SAT_C_VALID": sat_aug & c_valid_aug,
    }
    predictions = load_predictions(work / "jobs", LEARNED, len(truth))
    fixed = {
        "CONTENT_FIXED": {
            "orig": phase_ttc(content_fixed_phase(rob["r_short"], content["c_orig"])),
            "aug": phase_ttc(content_fixed_phase(rob["aug_r_short"], content["c_aug"])),
        },
        "ROBUST_FIXED": {
            "orig": phase_ttc(robust.raw_r_column(rob["r_short"], "phi_robust")),
            "aug": phase_ttc(robust.raw_r_column(rob["aug_r_short"], "phi_robust")),
        },
    }
    mids: dict[str, dict[str, NDArray[np.float64]]] = {}
    seed_overall: dict[str, list[float]] = {}
    for name in LEARNED:
        per_seed = [row_mid(truth, p) for p in predictions[name]["orig"]]
        mids[name] = {
            "orig": np.mean(per_seed, 0),
            "aug": np.mean([row_mid(truth, p) for p in predictions[name]["aug"]], 0),
        }
        seed_overall[name] = [stratum_overall(m, bands)["overall"] for m in per_seed]
    for name, value in fixed.items():
        mids[name] = {tag: row_mid(truth, value[tag]) for tag in ("orig", "aug")}
    arms = (*LEARNED, *FIXED)
    strata: dict[str, dict[str, Any]] = {}
    deltas: dict[str, dict[str, dict[str, Any]]] = {}
    for name in arms:
        strata[name] = {}
        for stratum, mask in masks.items():
            tag = "aug" if stratum.startswith("SYN") else "orig"
            strata[name][stratum] = stratum_overall(
                np.where(mask, mids[name][tag], np.nan), np.where(mask, bands, -1)
            )
        deltas[name] = {}
        for reference in REFERENCES:
            if reference == name:
                continue
            deltas[name][reference] = {}
            for stratum, mask in masks.items():
                tag = "aug" if stratum.startswith("SYN") else "orig"
                diff = mids[name][tag] - mids[reference][tag]
                deltas[name][reference][stratum] = bootstrap(
                    np.where(mask, diff, np.nan), np.where(mask, bands, -1), clusters, draws
                )
    dd = deltas[PRIMARY][BASELINE]
    d1 = dd["ALL"]["point"] <= MARGIN_MID
    d2 = dd["SYN_SAT"]["ci95"][1] < 0
    d3 = dd["SAT"]["point"] <= 0
    decision = {
        "schema": "saturation_scale_decision_v1",
        "rule": "PROTOCOL.md §4.1",
        "primary": PRIMARY,
        "baseline": BASELINE,
        "D1_all_delta_le_0.5": bool(d1),
        "D1_delta": dd["ALL"]["point"],
        "D1_ci95": dd["ALL"]["ci95"],
        "D2_syn_sat_ci_upper_lt_0": bool(d2),
        "D2_delta": dd["SYN_SAT"]["point"],
        "D2_ci95": dd["SYN_SAT"]["ci95"],
        "D3_real_sat_point_le_0": bool(d3),
        "D3_delta": dd["SAT"]["point"],
        "D3_ci95_informative": dd["SAT"]["ci95"],
        "verdict": "SUPPORTED" if (d1 and d2 and d3) else "NOT_SUPPORTED",
        "test12_gate": {
            arm: {
                "delta_all_vs_RGB_GEO_BAND": deltas[arm][GATE_COMPARATOR]["ALL"]["point"],
                "ci95": deltas[arm][GATE_COMPARATOR]["ALL"]["ci95"],
                "passes": bool(deltas[arm][GATE_COMPARATOR]["ALL"]["ci95"][1] < GATE_UPPER_MID),
            }
            for arm in GATE_ARMS
        },
    }
    decision["test12_package_required"] = any(v["passes"] for v in decision["test12_gate"].values())
    counts = {
        stratum: {b: int(((bands == i) & mask).sum()) for i, b in enumerate(BANDS)}
        | {"rows": int(mask.sum()), "sequences": int(len(np.unique(clusters[mask])))}
        for stratum, mask in masks.items()
    }
    sources = {
        "valid": {
            "orig": cs.raw_c_column(content["c_orig"], "valid"),
            "aug": cs.raw_c_column(content["c_aug"], "valid"),
        },
        "phi_content": {
            "orig": cs.raw_c_column(content["c_orig"], "phi_content"),
            "aug": cs.raw_c_column(content["c_aug"], "phi_content"),
        },
        "phi_robust": {
            "orig": robust.raw_r_column(rob["r_short"], "phi_robust"),
            "aug": robust.raw_r_column(rob["aug_r_short"], "phi_robust"),
        },
    }
    results = {
        "schema": "saturation_scale_results_v1",
        "protocol_sha256": sha256_file(docs / "PROTOCOL.md"),
        "rows": int(len(truth)),
        "sequences": int(len(sequences)),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES},
        "stratum_counts_by_target_band": counts,
        "seed_overall_all": seed_overall,
        "strata": strata,
        "deltas": deltas,
        "decision": decision,
        "signal_diagnostics": signal_diagnostics(truth, masks, sources),
        "replica_check": replica_check(predictions, truth, root),
        "official_scorer": official_checks(truth, predictions, root),
        "parameters": {
            n: int(json.loads((work / "jobs" / f"{n}__F0__S7.json").read_text())["parameters"])
            for n in LEARNED
        },
        "content_receipt_sha256": sha256_file(work / "CONTENT_RECEIPT.json"),
        "arms": {n: {"role": a.role, "inputs": len(a.columns)} for n, a in ARMS.items()},
    }
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "RESULTS.json").write_text(
        json.dumps(json_safe(results), indent=1) + "\n", encoding="utf-8"
    )
    (docs / "DECISION.json").write_text(
        json.dumps(json_safe(decision), indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(
        docs / "STRATA.csv",
        ["arm", "stratum", "rows", "overall", *[f"MiD{b}" for b in BANDS]],
        [
            [a, s, int(v["rows"]), v["overall"], *[v[f"MiD{b}"] for b in BANDS]]
            for a, block in strata.items()
            for s, v in block.items()
        ],
    )
    _write_csv(
        docs / "DELTAS.csv",
        ["arm", "reference", "stratum", "delta", "ci_lo", "ci_hi", *[f"d_{b}" for b in BANDS]],
        [
            [a, c, s, v["point"], v["ci95"][0], v["ci95"][1], *[v["band_point"][b] for b in BANDS]]
            for a, block in deltas.items()
            for c, inner in block.items()
            for s, v in inner.items()
        ],
    )
    print(json.dumps(json_safe(decision), indent=1))


if __name__ == "__main__":
    main()
