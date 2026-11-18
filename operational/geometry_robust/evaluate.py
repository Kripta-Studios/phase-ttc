"""OOF evaluation, strata, paired sequence-bootstrap CIs, acceptance and Test12 selection.

``python -m operational.geometry_robust.evaluate --geometry-work <dir> --work <dir> --docs <dir>``

Selection uses eAP TRAIN40 OOF rows only (PROTOCOL.md §5); FCWD and Test12 are never read here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from operational.phase_geometry import geometry
from operational.phase_geometry.metrics import (
    BAND_WEIGHTS,
    BANDS,
    WEIGHT_VECTOR,
    band_index,
    json_safe,
    row_mid,
)
from operational.second_benchmark.fcwd import phase_ttc

from . import robust
from .run_cv import FOLDS, SEEDS, load_design
from .variants import LEARNED, TEST12_ELIGIBLE, VARIANTS

BOOTSTRAP_SEED = 20261118
BOOTSTRAP_REPLICATES = 10_000
MARGIN_MID = 0.5
TIE_MID = 0.5
RECOMMEND_GAIN_MID = -0.5
COMPARATORS = ("RGB_GEO_BAND", "GEO_LEARNED")
FIXED = ("GEO_AREA", "GEO_TRUNC_AWARE", "ROBUST_FIXED", "ROBUST_FIXED_MEM")
REAL_STRATA = ("ALL", "TRUNC", "NOTRUNC", "SAT", "NOSAT")
SYN_STRATA = ("SYN_ALL", "SYN_SAT", "SYN_NOSAT")


def _npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def stratum_overall(mid: NDArray[Any], bands: NDArray[Any]) -> dict[str, float]:
    """Band means and Overall with weights renormalized over the bands present."""
    m = np.asarray(mid, dtype=np.float64)
    out: dict[str, float] = {}
    total, weight = 0.0, 0.0
    for index, name in enumerate(BANDS):
        values = m[(bands == index) & np.isfinite(m)]
        out[f"MiD{name}"] = float(values.mean()) if len(values) else float("nan")
        out[f"n_{name}"] = float(len(values))
        if len(values):
            total += BAND_WEIGHTS[name] * float(values.mean())
            weight += BAND_WEIGHTS[name]
    out["overall"] = total / weight if weight else float("nan")
    out["rows"] = float(np.isfinite(m[bands >= 0]).sum())
    return out


def bootstrap(
    values: NDArray[Any], bands: NDArray[Any], clusters: NDArray[np.int64], draws: NDArray[Any]
) -> dict[str, Any]:
    """Renormalized band-weighted mean of ``values`` with a whole-sequence bootstrap CI."""
    n_clusters = draws.shape[1]
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v) & (bands >= 0)
    sums = np.zeros((n_clusters, len(BANDS)))
    counts = np.zeros((n_clusters, len(BANDS)))
    np.add.at(sums, (clusters[ok], bands[ok]), v[ok])
    np.add.at(counts, (clusters[ok], bands[ok]), 1.0)

    def overall(s: NDArray[Any], c: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        with np.errstate(invalid="ignore", divide="ignore"):
            means = s / c
        present = c > 0
        w = WEIGHT_VECTOR * present
        return np.nansum(np.where(present, means, 0.0) * WEIGHT_VECTOR, -1) / w.sum(-1), means

    point, band_point = overall(sums.sum(0), counts.sum(0))
    dist, band_dist = overall(sums[draws].sum(1), counts[draws].sum(1))
    low, high = np.nanquantile(dist, [0.025, 0.975])
    band_ci = np.nanquantile(band_dist, [0.025, 0.975], axis=0)
    return {
        "point": float(point),
        "ci95": [float(low), float(high)],
        "band_point": {b: float(band_point[i]) for i, b in enumerate(BANDS)},
        "band_ci95": {b: [float(band_ci[0, i]), float(band_ci[1, i])] for i, b in enumerate(BANDS)},
        "rows": int(ok.sum()),
        "clusters_with_rows": int((counts.sum(1) > 0).sum()),
    }


def load_predictions(work: Path, rows: int) -> dict[str, dict[str, NDArray[np.float64]]]:
    """``{variant: {"orig": [S, N], "aug": [S, N]}}`` TTC predictions from the CV jobs."""
    out: dict[str, dict[str, NDArray[np.float64]]] = {}
    for name in LEARNED:
        orig = np.full((len(SEEDS), rows), np.nan)
        aug = np.full((len(SEEDS), rows), np.nan)
        for s_index, seed in enumerate(SEEDS):
            for fold in FOLDS:
                job = _npz(work / "jobs" / f"{name}__F{fold}__S{seed}.npz")
                orig[s_index, job["rows"]] = job["ttc_orig"]
                aug[s_index, job["rows"]] = job["ttc_aug"]
        if not (np.isfinite(orig).all() and np.isfinite(aug).all()):
            raise ValueError(f"{name}: missing OOF predictions")
        out[name] = {"orig": orig, "aug": aug}
    return out


def fixed_predictions(g_raw: NDArray[Any], r_short: NDArray[Any], r_long: NDArray[Any]) -> dict:
    """Zero-learning experts (|T| <= 60 s, exact-zero phase -> +60 s)."""
    return {
        "GEO_AREA": phase_ttc(geometry.raw_column(g_raw, "phi_area")),
        "GEO_TRUNC_AWARE": phase_ttc(geometry.raw_column(g_raw, "phi_trunc_aware")),
        "ROBUST_FIXED": phase_ttc(robust.raw_r_column(r_short, "phi_robust")),
        "ROBUST_FIXED_MEM": phase_ttc(robust.raw_r_column(r_long, "phi_robust")),
    }


def replica_check(geometry_work: Path, predictions: dict, truth: NDArray[Any]) -> dict[str, Any]:
    """Descriptive: refit comparators against the OOF jobs of the original geometry study."""
    out: dict[str, Any] = {}
    for name in COMPARATORS:
        old = np.full((len(SEEDS), len(truth)), np.nan)
        for s_index, seed in enumerate(SEEDS):
            for fold in FOLDS:
                job = _npz(geometry_work / "jobs" / f"{name}__F{fold}__S{seed}__M32.npz")
                old[s_index, job["rows"]] = job["ttc"]
        new = predictions[name]["orig"]
        bands = band_index(truth)
        old_mid = np.nanmean([row_mid(truth, o) for o in old], 0)
        new_mid = np.nanmean([row_mid(truth, n) for n in new], 0)
        out[name] = {
            "max_abs_ttc_difference": float(np.abs(old - new).max()),
            "median_abs_ttc_difference": float(np.median(np.abs(old - new))),
            "overall_original_study": stratum_overall(old_mid, bands)["overall"],
            "overall_refit": stratum_overall(new_mid, bands)["overall"],
        }
    return out


def official_checks(truth: NDArray[Any], predictions: dict, root: Path) -> dict[str, Any]:
    """Max |official - vectorized| Overall over every learned variant and seed (ALL rows)."""
    from operational.garl_comparison.mid import load_scorer
    from operational.phase_geometry.data import verified_file
    from operational.phase_geometry.metrics import official_check

    scorer = load_scorer(verified_file("official_scorer", root))
    diffs = [
        official_check(scorer, truth, predictions[name]["orig"][s])
        for name in LEARNED
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
    """Evaluate every variant and write RESULTS.json, CSVs and SELECTION.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--geometry-work", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    work, docs = args.work.resolve(), args.docs.resolve()
    docs.mkdir(parents=True, exist_ok=True)
    d = load_design(args.geometry_work, work)
    train = _npz(args.geometry_work / "train_inputs.npz")
    rob = _npz(work / "robust_work.npz")
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
    masks = {
        "ALL": np.ones(len(truth), bool),
        "TRUNC": trunc,
        "NOTRUNC": ~trunc,
        "SAT": sat,
        "NOSAT": ~sat,
        "SYN_ALL": np.ones(len(truth), bool),
        "SYN_SAT": sat_aug,
        "SYN_NOSAT": ~sat_aug,
    }
    predictions = load_predictions(work, len(truth))
    fixed_orig = fixed_predictions(g_raw, rob["r_short"], rob["r_long"])
    fixed_aug = fixed_predictions(rob["aug_geometry_raw"], rob["aug_r_short"], rob["aug_r_long"])
    mids: dict[str, dict[str, NDArray[np.float64]]] = {}
    seed_overall: dict[str, list[float]] = {}
    for name in LEARNED:
        per_seed = [row_mid(truth, p) for p in predictions[name]["orig"]]
        mids[name] = {
            "orig": np.mean(per_seed, 0),
            "aug": np.mean([row_mid(truth, p) for p in predictions[name]["aug"]], 0),
        }
        seed_overall[name] = [stratum_overall(m, bands)["overall"] for m in per_seed]
    for name in FIXED:
        mids[name] = {
            "orig": row_mid(truth, fixed_orig[name]),
            "aug": row_mid(truth, fixed_aug[name]),
        }
    arms = (*LEARNED, *FIXED)
    strata: dict[str, dict[str, Any]] = {}
    for name in arms:
        strata[name] = {}
        for stratum, mask in masks.items():
            source = "aug" if stratum.startswith("SYN") else "orig"
            strata[name][stratum] = stratum_overall(
                np.where(mask, mids[name][source], np.nan), np.where(mask, bands, -1)
            )
    deltas: dict[str, dict[str, dict[str, Any]]] = {}
    for name in arms:
        deltas[name] = {}
        for comparator in COMPARATORS:
            if comparator == name:
                continue
            deltas[name][comparator] = {}
            for stratum, mask in masks.items():
                source = "aug" if stratum.startswith("SYN") else "orig"
                diff = mids[name][source] - mids[comparator][source]
                deltas[name][comparator][stratum] = bootstrap(
                    np.where(mask, diff, np.nan), np.where(mask, bands, -1), clusters, draws
                )
    acceptance: dict[str, dict[str, Any]] = {}
    for name in LEARNED:
        comparator = VARIANTS[name].comparator
        if not comparator:
            continue
        dd = deltas[name][comparator]
        a1 = dd["ALL"]["point"] <= MARGIN_MID
        a2 = dd["TRUNC"]["ci95"][1] < 0
        a3 = dd["SYN_SAT"]["ci95"][1] < 0
        acceptance[name] = {
            "comparator": comparator,
            "A1_all_delta_le_margin": bool(a1),
            "A1_delta": dd["ALL"]["point"],
            "A1_ci95": dd["ALL"]["ci95"],
            "A2_trunc_ci_upper_lt_0": bool(a2),
            "A2_delta": dd["TRUNC"]["point"],
            "A2_ci95": dd["TRUNC"]["ci95"],
            "A3_syn_sat_ci_upper_lt_0": bool(a3),
            "A3_delta": dd["SYN_SAT"]["point"],
            "A3_ci95": dd["SYN_SAT"]["ci95"],
            "real_sat_delta_informative_only": dd["SAT"]["point"],
            "real_sat_ci95_informative_only": dd["SAT"]["ci95"],
            "accepted": bool(a1 and a2 and a3),
        }
    accepted = [n for n in TEST12_ELIGIBLE if acceptance[n]["accepted"]]
    pool = accepted or list(TEST12_ELIGIBLE)
    mean_overall = {n: float(np.mean(seed_overall[n])) for n in LEARNED}
    best = min(pool, key=mean_overall.__getitem__)
    if "RGB_ROBUST" in pool and mean_overall["RGB_ROBUST"] <= mean_overall[best] + TIE_MID:
        best = "RGB_ROBUST"
    chosen_delta = deltas[best]["RGB_GEO_BAND"]["ALL"]
    recommend = bool(
        acceptance[best]["accepted"]
        and chosen_delta["point"] <= RECOMMEND_GAIN_MID
        and chosen_delta["ci95"][1] < 0
    )
    selection = {
        "schema": "geometry_robust_selection_v1",
        "eligible": list(TEST12_ELIGIBLE),
        "accepted": accepted,
        "candidate": best,
        "candidate_status": "ACCEPTED" if acceptance[best]["accepted"] else "NOT_ACCEPTED",
        "seed_mean_overall": {n: mean_overall[n] for n in TEST12_ELIGIBLE},
        "candidate_delta_all_vs_RGB_GEO_BAND": chosen_delta["point"],
        "candidate_delta_all_ci95": chosen_delta["ci95"],
        "recommend_test_submission_by_rule": recommend,
        "rule": "PROTOCOL.md §5.2",
    }
    counts = {
        stratum: {b: int(((bands == i) & mask).sum()) for i, b in enumerate(BANDS)}
        | {"rows": int(mask.sum()), "sequences": int(len(np.unique(clusters[mask])))}
        for stratum, mask in masks.items()
    }
    sources = {
        tag: {
            key: int((robust.raw_r_column(table, key) > 0).sum())
            for key in ("src_clean", "src_free_axis", "src_missing", "edge_valid", "sat")
        }
        for tag, table in (
            ("real_short", rob["r_short"]),
            ("real_long", rob["r_long"]),
            ("aug_short", rob["aug_r_short"]),
            ("aug_long", rob["aug_r_long"]),
        )
    }
    results = {
        "schema": "geometry_robust_results_v1",
        "protocol_sha256": _sha(docs / "PROTOCOL.md"),
        "rows": int(len(truth)),
        "sequences": int(len(sequences)),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES},
        "stratum_counts_by_target_band": counts,
        "robust_source_counts": sources,
        "seed_overall_all": seed_overall,
        "strata": strata,
        "deltas": deltas,
        "acceptance": acceptance,
        "selection": selection,
        "replica_check": replica_check(args.geometry_work, predictions, truth),
        "official_scorer": official_checks(truth, predictions, args.root.resolve()),
        "parameters": {
            n: int(json.loads((work / "jobs" / f"{n}__F0__S7.json").read_text())["parameters"])
            for n in LEARNED
        },
    }
    (docs / "RESULTS.json").write_text(
        json.dumps(json_safe(results), indent=1) + "\n", encoding="utf-8"
    )
    (docs / "SELECTION.json").write_text(
        json.dumps(json_safe(selection), indent=2) + "\n", encoding="utf-8"
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
        ["arm", "comparator", "stratum", "delta", "ci_lo", "ci_hi", *[f"d_{b}" for b in BANDS]],
        [
            [a, c, s, v["point"], v["ci95"][0], v["ci95"][1], *[v["band_point"][b] for b in BANDS]]
            for a, block in deltas.items()
            for c, inner in block.items()
            for s, v in inner.items()
        ],
    )
    print(json.dumps(json_safe({"acceptance": acceptance, "selection": selection}), indent=1))


def _sha(path: Path) -> str:
    from operational.phase_geometry.data import sha256_file

    return sha256_file(path)


__all__ = ["bootstrap", "stratum_overall"]


if __name__ == "__main__":
    main()
