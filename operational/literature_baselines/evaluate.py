"""Score every baseline row and comparator on the pre-registered subset (repository venv).

``python -m operational.literature_baselines.evaluate --artifacts <dir> --docs <dir>``
reads ``SUBSET_MANIFEST.npz``, ``runs/<METHOD>_OUTPUTS.npz``, the geometry study's
``OOF_PREDICTIONS.parquet`` and the pinned official scorer, and writes ``RESULTS.json`` and
``MAIN_TABLE.csv`` (docs) plus ``PER_QUERY.parquet`` (artifacts).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from operational.garl_comparison.mid import load_scorer
from operational.phase_geometry.data import sha256_file, verified_file
from operational.phase_geometry.metrics import (
    BANDS,
    band_index,
    bootstrap_overall,
    json_safe,
    official_check,
    overall_from_mid,
    row_mid,
    summarize,
    ttc_to_phase,
)

from .common import (
    box_ratio_rho,
    clean_rho,
    cross_fitted_scale,
    phase_to_ttc,
    rho_to_phase_cv,
    rho_to_phase_exp,
)

DENSE = ("OEXP", "SFPP")
MODELS = ("OEXP", "SFPP", "RAFT_AFF")
BASELINES = (*MODELS, "BOX2F")
SEEDED = ("REL", "RGB_GEO_BAND", "GEO_LEARNED")
FIXED_OOF = ("GEO_AREA",)
REFERENCES = ("RGB_GEO_BAND", "BOX2F")
SIZE_BINS = (0.0, 64.0, 128.0, 256.0, float("inf"))
OOF_PATH = "artifacts/sota_20261118/geometry/OOF_PREDICTIONS.parquet"
LABELS = {
    "OEXP": "Optical Expansion (VCN-exp, KITTI)",
    "SFPP": "ScaleFlow++ (KITTI FT)",
    "RAFT_AFF": "RAFT-large + affine fit",
    "BOX2F": "Two-frame box scale ratio",
    "GEO_AREA": "Fixed box-geometry estimator (<=600 ms)",
    "REL": "Released PHASE-TTC head (OOF)",
    "RGB_GEO_BAND": "RGB+geometry head (OOF)",
    "GEO_LEARNED": "Geometry-only learned head (OOF)",
}

FloatArray = NDArray[np.float64]


def load_npz(path: Path) -> dict[str, NDArray[Any]]:
    """Non-pickled NPZ as a dict."""
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def rankdata(values: FloatArray) -> FloatArray:
    """Average ranks (ties share their mean rank)."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse = np.unique(values, return_inverse=True)
    sums = np.zeros(len(unique))
    counts = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    np.add.at(counts, inverse, 1.0)
    return (sums / counts)[inverse]


def correlations(phase_hat: FloatArray, phase_true: FloatArray) -> dict[str, float]:
    """Descriptive Pearson/Spearman correlation and sign agreement with the target phase."""
    return {
        "pearson": float(np.corrcoef(phase_hat, phase_true)[0, 1]),
        "spearman": float(np.corrcoef(rankdata(phase_hat), rankdata(phase_true))[0, 1]),
        "sign_agreement": float(np.mean(np.sign(phase_hat) == np.sign(phase_true))),
    }


def baseline_rows(
    manifest: dict[str, NDArray[Any]], outputs: dict[str, dict[str, NDArray[Any]]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Every pre-registered baseline variant as predicted TTC plus bookkeeping."""
    delta = manifest["delta_us"].astype(np.float64) * 1e-6
    truth_phase = ttc_to_phase(manifest["ttc"])
    rows: dict[str, dict[str, Any]] = {}
    extra: dict[str, Any] = {"failures": {}, "calibration_scales": {}, "correlations": {}}
    rho_sources: dict[str, dict[str, FloatArray]] = {}
    for name in MODELS:
        out = outputs[name]
        if not np.array_equal(out["sample_token"], manifest["sample_token"]):
            raise ValueError(f"{name} outputs are not aligned with the manifest")
        rho_sources[name] = {"median": out["rho_median"], "inner": out["rho_inner"]}
        if name in DENSE:
            rho_sources[name]["mean"] = out["rho_mean"]
    rho_sources["BOX2F"] = {
        "median": np.asarray(
            [
                box_ratio_rho(p, c)
                for p, c in zip(manifest["box_previous"], manifest["box_current"], strict=True)
            ]
        )
    }
    for name in BASELINES:
        failures: dict[str, int] = {}
        cleaned: dict[str, FloatArray] = {}
        for how, raw in rho_sources[name].items():
            cleaned[how], failed = clean_rho(raw)
            failures[how] = int(failed.sum())
        extra["failures"][name] = failures
        primary_phase = rho_to_phase_cv(cleaned["median"], delta)
        variants: dict[str, tuple[FloatArray, str, str]] = {
            name: (primary_phase, "primary", "CV conversion, primary aggregation"),
            f"{name}__EXP": (
                rho_to_phase_exp(cleaned["median"], delta),
                "secondary",
                "EXP conversion, primary aggregation",
            ),
        }
        if "inner" in cleaned:
            variants[f"{name}__INNER"] = (
                rho_to_phase_cv(cleaned["inner"], delta),
                "secondary",
                "CV conversion, inner 50% box",
            )
        if "mean" in cleaned:
            variants[f"{name}__MEAN"] = (
                rho_to_phase_cv(cleaned["mean"], delta),
                "secondary",
                "CV conversion, mean over box",
            )
        calibrated, scales = cross_fitted_scale(
            truth_phase, primary_phase, manifest["band"], manifest["fold"]
        )
        variants[f"{name}__CAL"] = (
            calibrated,
            "calibrated",
            "a * CV phase, a fitted on the other 4 blocks (grouped CV)",
        )
        extra["calibration_scales"][name] = scales
        extra["correlations"][name] = correlations(primary_phase, truth_phase)
        for key, (phase, kind, description) in variants.items():
            rows[key] = {
                "ttc": [phase_to_ttc(phase)],
                "kind": kind,
                "base": name,
                "description": description,
            }
    return rows, extra


def comparator_rows(oof: pd.DataFrame, tokens: NDArray[Any]) -> dict[str, dict[str, Any]]:
    """OOF heads (three seeds) and the fixed geometry expert matched by sample token."""
    indexed = oof.set_index("sample_token").loc[tokens.astype(str)]
    rows: dict[str, dict[str, Any]] = {}
    for name in SEEDED:
        rows[name] = {
            "ttc": [indexed[f"{name}_S{s}_ttc"].to_numpy(np.float64) for s in (7, 13, 21)],
            "kind": "ours_oof",
            "base": name,
            "description": "sequence-grouped OOF, seed mean of per-row MiD (7/13/21)",
        }
    for name in FIXED_OOF:
        rows[name] = {
            "ttc": [indexed[f"{name}_ttc"].to_numpy(np.float64)],
            "kind": "fixed_geometry",
            "base": name,
            "description": "fixed causal box-geometry expert (<=600 ms OLS), no learning",
        }
    return rows


def group_overall(mid: FloatArray, bands: NDArray[Any], groups: NDArray[Any]) -> dict[str, Any]:
    """Overall per descriptive group label."""
    out: dict[str, Any] = {}
    for label in sorted(set(groups.tolist())):
        member = groups == label
        out[str(label)] = {
            "rows": int(member.sum()),
            "overall": overall_from_mid(mid[member], bands[member]),
        }
    return out


def evaluate(root: Path, artifacts: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Compute every table of the report."""
    manifest = load_npz(artifacts / "SUBSET_MANIFEST.npz")
    outputs = {m: load_npz(artifacts / "runs" / f"{m}_OUTPUTS.npz") for m in MODELS}
    oof = pd.read_parquet(root / OOF_PATH)
    scorer = load_scorer(verified_file("official_scorer", root))
    truth = manifest["ttc"].astype(np.float64)
    bands = manifest["band"]
    if not np.array_equal(band_index(truth), bands):
        raise ValueError("Manifest bands differ from the public targets")
    sequences = manifest["sequence_id"].astype(str)
    rows, extra = baseline_rows(manifest, outputs)
    rows.update(comparator_rows(oof, manifest["sample_token"]))
    checks: list[float] = []
    mids: dict[str, FloatArray] = {}
    table: list[dict[str, Any]] = []
    for name, spec in rows.items():
        per_seed = [row_mid(truth, t) for t in spec["ttc"]]
        for t in spec["ttc"]:
            checks.append(official_check(scorer, truth, t))
        mid = np.mean(per_seed, axis=0)
        mids[name] = mid
        seed_summaries = [summarize(truth, t) for t in spec["ttc"]]
        boot = bootstrap_overall(mid, bands, sequences)
        entry: dict[str, Any] = {
            "row": name,
            "base": spec["base"],
            "kind": spec["kind"],
            "description": spec["description"],
            "label": LABELS.get(spec["base"], spec["base"]),
            "overall": overall_from_mid(mid, bands),
            "overall_ci95": [boot["ci95_low"], boot["ci95_high"]],
            "overall_seed_sd": float(np.std([s["overall_MiD"] for s in seed_summaries], ddof=0)),
            "failed_rate": float(np.mean([s["failed_rate"] for s in seed_summaries])),
        }
        for b in BANDS:
            entry[f"MiD{b}"] = boot["band_point"][b]
        table.append(entry)
    max_check = max(checks)
    if max_check > 1e-6:
        raise ValueError(f"Vectorized Overall differs from the official scorer by {max_check}")
    deltas: dict[str, dict[str, Any]] = {}
    for reference in REFERENCES:
        deltas[reference] = {}
        for name in rows:
            if name == reference:
                continue
            boot = bootstrap_overall(mids[name] - mids[reference], bands, sequences)
            deltas[reference][name] = {
                "delta": boot["point"],
                "ci95": [boot["ci95_low"], boot["ci95_high"]],
                "band_delta": boot["band_point"],
                "band_ci95": boot["band_ci95"],
            }
    truncated = np.where(manifest["truncated"], "truncated", "not_truncated")
    size_bin = np.digitize(manifest["box_size"], SIZE_BINS[1:-1])
    size_labels = np.asarray([f"[{SIZE_BINS[i]:.0f},{SIZE_BINS[i + 1]:.0f})" for i in size_bin])
    breakdown_rows = (*BASELINES, *FIXED_OOF, *SEEDED)
    breakdowns = {
        "truncated": {n: group_overall(mids[n], bands, truncated) for n in breakdown_rows},
        "box_size": {n: group_overall(mids[n], bands, size_labels) for n in breakdown_rows},
    }
    labels_full = (
        pd.read_parquet(verified_file("train_annotations", root), columns=["sample_token", "ttc"])
        .set_index("sample_token")
        .loc[oof["sample_token"].astype(str), "ttc"]
        .to_numpy(np.float64)
    )
    full_bands = band_index(labels_full)
    population: dict[str, float] = {}
    for name in (*SEEDED, *FIXED_OOF):
        columns = [f"{name}_S{s}_ttc" for s in (7, 13, 21)] if name in SEEDED else [f"{name}_ttc"]
        mid = np.mean([row_mid(labels_full, oof[c].to_numpy(np.float64)) for c in columns], axis=0)
        population[name] = overall_from_mid(mid, full_bands)
    runtime: dict[str, Any] = {}
    for name in MODELS:
        receipt = json.loads((artifacts / "runs" / f"{name}_RECEIPT.json").read_text("utf-8"))
        out = outputs[name]
        runtime[name] = {
            "forward_ms_median": 1e3 * float(np.median(out["forward_s"])),
            "forward_ms_mean": 1e3 * float(np.mean(out["forward_s"])),
            "prep_ms_median": 1e3 * float(np.median(out["prep_s"])),
            "peak_cuda_memory_gb": receipt["peak_cuda_memory_gb"],
            "gpu": receipt["gpu"],
            "torch": receipt["torch"],
            "wall_seconds_last_invocation": receipt["wall_seconds_this_invocation"],
            "provenance": receipt["provenance"],
            "pilot_forward_ms_median": 1e3
            * json.loads((artifacts / "runs" / f"{name}_PILOT_RECEIPT.json").read_text("utf-8"))[
                "forward_seconds_median"
            ],
        }
    sanity: dict[str, Any] = {}
    for name in MODELS:
        sanity[name] = json.loads((artifacts / "sanity" / f"{name}_SANITY.json").read_text("utf-8"))
    maps = [np.load(artifacts / "sanity" / f"{m}_KITTI_LOGMID.npy").ravel() for m in DENSE]
    finite = np.isfinite(maps[0]) & np.isfinite(maps[1])
    sanity["kitti_oexp_sfpp_pearson"] = float(np.corrcoef(maps[0][finite], maps[1][finite])[0, 1])
    results = {
        "schema": "phase_literature_baselines_results_v1",
        "generated_unix": time.time(),
        "queries": len(truth),
        "sequences": int(len(set(sequences.tolist()))),
        "band_counts": {b: int((bands == i).sum()) for i, b in enumerate(BANDS)},
        "official_scorer_checks": len(checks),
        "official_scorer_max_abs_overall_difference": max_check,
        "table": table,
        "deltas": deltas,
        "breakdowns": breakdowns,
        "population_overall_88744": population,
        "runtime": runtime,
        "sanity": sanity,
        **extra,
        "inputs": {
            "manifest_sha256": sha256_file(artifacts / "SUBSET_MANIFEST.npz"),
            "outputs_sha256": {
                m: sha256_file(artifacts / "runs" / f"{m}_OUTPUTS.npz") for m in MODELS
            },
            "oof_predictions_sha256": sha256_file(root / OOF_PATH),
            "evaluate_source_sha256": sha256_file(Path(__file__)),
        },
        "test12_private_labels_read": False,
    }
    per_query = pd.DataFrame(
        {
            "sample_token": manifest["sample_token"],
            "sequence_id": sequences,
            "fold": manifest["fold"],
            "ttc_true": truth,
            "band": bands,
            "truncated": manifest["truncated"],
            "box_size": manifest["box_size"],
            **{f"{n}_ttc": rows[n]["ttc"][0] for n in rows if rows[n]["kind"] != "ours_oof"},
            **{f"{n}_mid": mids[n] for n in rows},
        }
    )
    return results, pd.DataFrame(table), per_query


def main() -> None:
    """Write RESULTS.json, MAIN_TABLE.csv and PER_QUERY.parquet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    results, table, per_query = evaluate(args.root.resolve(), args.artifacts)
    args.docs.mkdir(parents=True, exist_ok=True)
    (args.docs / "RESULTS.json").write_text(
        json.dumps(json_safe(results), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table.drop(columns=["overall_ci95"]).assign(
        ci95_low=[c[0] for c in table["overall_ci95"]],
        ci95_high=[c[1] for c in table["overall_ci95"]],
    ).to_csv(args.docs / "MAIN_TABLE.csv", index=False, float_format="%.4f")
    per_query.to_parquet(args.artifacts / "PER_QUERY.parquet", index=False)
    for entry in results["table"]:
        print(
            f"{entry['row']:<22} {entry['overall']:8.2f}  "
            f"[{entry['overall_ci95'][0]:.2f}, {entry['overall_ci95'][1]:.2f}]"
        )


if __name__ == "__main__":
    main()
