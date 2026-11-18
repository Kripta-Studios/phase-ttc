"""FCWD zero-shot evaluation of the PHASE-TTC-Geo geometry arms: predict, seal, score.

``python -m operational.second_benchmark.run predict`` builds the label-free cohort and writes
sealed predictions; ``python -m operational.second_benchmark.run score`` verifies the seal,
joins ``gt_ttc.csv`` and writes RESULTS.json and CSVs.  Requires ``PHASE_TTC_FCWD_ROOT``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import fcwd

ARTIFACTS = Path("artifacts/sota_20261118/second_benchmark")
DOCS = Path("docs/sota_20261118/second_benchmark")
GEOMETRY_WORK = Path("artifacts/sota_20261118/geometry/work")
PRIOR_SCORED = Path(
    "docs/sota_campaign_20261008/evidence/fcwd_inference/scoring/SCORED_PREDICTIONS.csv"
)
PRIOR_METHODS = ("public_Garl_event_lhr", "H8_seed7", "H8_seed13", "H8_seed23")
SEEDS = (7, 13, 21)
BOOTSTRAP_SEED = 20261120
BOOTSTRAP_DRAWS = 10_000
RANGES = (("0-3", 0.0, 3.0), ("3-6", 3.0, 6.0), ("6-20", 6.0, 20.0), (">20", 20.0, math.inf))
SOURCE_FILES = (
    "operational/second_benchmark/fcwd.py",
    "operational/second_benchmark/run.py",
    "operational/second_benchmark/train_geo.py",
    "operational/phase_geometry/geometry.py",
    "operational/phase_geometry/heads.py",
    "operational/phase_geometry/metrics.py",
    "operational/sota_eval/fcwd_score.py",
)
PUBLISHED_TABLE_VII = {
    "ETTCM": (15.5, 18.4, 19.0),
    "STRTTC": (9.8, 11.5, 14.0),
    "DeepScale": (25.4, 19.6, 21.7),
    "Garl visual-only baseline": (48.5, 47.6, 46.8),
    "Garl visual-only + LHR": (13.3, 16.7, 16.6),
    "Garl-TTC (Full)": (5.2, 6.1, 5.4),
}


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prior_rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{c: row[c] for c in columns} for row in csv.DictReader(handle)]


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _fmt(value: float) -> str:
    return repr(float(value))


# --------------------------------------------------------------------------------- predict


def predict(out: Path) -> dict[str, Any]:
    """Build the cohort, compute every arm and seal the predictions (no TTC target read)."""
    import torch

    from operational.phase_geometry.heads import predict_head

    root = fcwd.fcwd_root()
    receipt = json.loads((out / "GEO_LEARNED_EAP_RECEIPT.json").read_text(encoding="utf-8"))
    states = {}
    for fit in receipt["fits"]:
        path = out / fit["state_path"]
        if sha256_file(path) != fit["state_sha256"]:
            raise ValueError(f"state dict changed: {path}")
        states[int(fit["seed"])] = torch.load(path, weights_only=True)
    with np.load(GEOMETRY_WORK / "train_inputs.npz", allow_pickle=False) as archive:
        constant = float(np.median(archive["target_ttc"]))
    prior = _prior_rows(PRIOR_SCORED, ("query_id", "sequence_id", "anchor_us"))
    columns: dict[str, list[float]] = {}
    meta: list[tuple[str, str, int]] = []
    diagnostics: dict[str, list[float]] = {"n_frames_P20": [], "both_truncated_current": []}
    inputs: list[dict[str, Any]] = []
    for number in fcwd.SEQUENCES:
        sequence_id = f"FCWD{number}"
        bbox_path = root / "sealed_labels" / f"sequence_{number}" / "bounding_box.csv"
        hdf5_path = root / "inputs" / f"sequence_{number}" / fcwd.HDF5_NAMES[number]
        table = fcwd.load_boxes(bbox_path, sequence_id)
        start, end = fcwd.event_span_us(hdf5_path)
        anchors = fcwd.cohort_anchors(table, start, end)
        expected = [int(r["anchor_us"]) for r in prior if r["sequence_id"] == sequence_id]
        if anchors.tolist() != expected:
            raise ValueError(f"{sequence_id}: cohort anchors differ from the committed cohort")
        inputs.append(
            {
                "sequence_id": sequence_id,
                "bbox_csv": bbox_path.relative_to(root).as_posix(),
                "bbox_sha256": sha256_file(bbox_path),
                "bbox_rows": len(table.time_us),
                "hdf5": hdf5_path.relative_to(root).as_posix(),
                "event_span_us": [start, end],
                "queries": len(anchors),
            }
        )
        ids = [r["query_id"] for r in prior if r["sequence_id"] == sequence_id]
        meta.extend((qid, sequence_id, int(a)) for qid, a in zip(ids, anchors, strict=True))
        block: dict[str, NDArray[np.float64]] = {
            "CONST_EAP_MEDIAN": np.full(len(anchors), constant)
        }
        for latency_name, latency_us in fcwd.LATENCIES_US.items():
            suffix = "" if latency_name == "P" else f"__{latency_name}"
            for rate in fcwd.RATES:
                q = fcwd.query_geometry(table, anchors, latency_us, rate)
                rate_suffix = "" if rate == 20 else "_10HZ"
                for name, values in fcwd.fixed_arms(q).items():
                    if name == "SIZE_RATIO_LAST2" and rate == 10:
                        continue
                    block[f"{name}{rate_suffix}{suffix}"] = values
                x, anchor = fcwd.learned_design(q.raw)
                width = np.zeros((len(x), 2), dtype=np.float32)
                learned = "GEO_LEARNED_EAP_ZS" + ("_20HZ" if rate == 20 else "")
                for seed, state in states.items():
                    block[f"{learned}_S{seed}{suffix}"] = predict_head(state, x, anchor, width)[
                        "ttc"
                    ]
                if latency_name == "P" and rate == 20:
                    diagnostics["n_frames_P20"].extend(q.n_frames.astype(float).tolist())
                    diagnostics["both_truncated_current"].extend(
                        q.both_truncated_current.astype(float).tolist()
                    )
        for name, values in block.items():
            columns.setdefault(name, []).extend(np.asarray(values, dtype=np.float64).tolist())
    arms = sorted(columns)
    header = ["query_id", "sequence_id", "anchor_us", *diagnostics, *arms]
    rows = [
        [
            m[0],
            m[1],
            m[2],
            *[_fmt(diagnostics[d][i]) for d in diagnostics],
            *[_fmt(columns[a][i]) for a in arms],
        ]
        for i, m in enumerate(meta)
    ]
    for arm in arms:
        values = np.asarray(columns[arm])
        if not np.isfinite(values).all() or (np.abs(values) < 1e-9).any():
            raise ValueError(f"{arm}: non-finite or zero prediction")
    predictions = out / "PREDICTIONS_SEALED.csv"
    _write_csv(predictions, header, rows)
    seal = {
        "schema": "second_benchmark_fcwd_seal_v1",
        "predictions": predictions.name,
        "predictions_sha256": sha256_file(predictions),
        "queries": len(meta),
        "arms": arms,
        "constant_eap_median_ttc_s": constant,
        "inputs": inputs,
        "gt_ttc_opened": False,
        "cohort_matches_committed_anchors": True,
        "geo_learned_receipt_sha256": sha256_file(out / "GEO_LEARNED_EAP_RECEIPT.json"),
        "source_sha256": {p: sha256_file(Path(p)) for p in SOURCE_FILES},
        "protocol_sha256": sha256_file(DOCS / "PROTOCOL.md"),
    }
    (out / "SEAL.json").write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
    return seal


# ----------------------------------------------------------------------------------- score


def extrapolate_truth(times: NDArray[Any], targets: NDArray[Any], query: float) -> float:
    """MATLAB ``interp1(times, targets, query, 'linear', 'extrap')``."""
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    index = int(np.clip(np.searchsorted(t, query), 1, len(t) - 1))
    left, right = index - 1, index
    weight = (query - t[left]) / (t[right] - t[left])
    return float(y[left] + weight * (y[right] - y[left]))


def rte(prediction: NDArray[Any], truth: NDArray[Any]) -> NDArray[np.float64]:
    """Official ``metricRee``: ``100*|gt - pred|/|gt|``."""
    p = np.asarray(prediction, dtype=np.float64)
    g = np.asarray(truth, dtype=np.float64)
    return 100.0 * np.abs(g - p) / np.abs(g)


def sequence_means(values: NDArray[Any], sequences: NDArray[Any], order: list[str]) -> NDArray[Any]:
    """Arithmetic mean of ``values`` per sequence, in ``order``."""
    v = np.asarray(values, dtype=np.float64)
    return np.asarray([v[sequences == s].mean() for s in order], dtype=np.float64)


def bootstrap_macro(per_sequence: NDArray[Any], draws: NDArray[np.int64]) -> list[float]:
    """Percentile 95 % CI of the macro mean over resampled whole sequences."""
    stats = np.asarray(per_sequence, dtype=np.float64)[draws].mean(axis=1)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def summarize_arm(
    prediction: NDArray[Any], truth: NDArray[Any], sequences: NDArray[Any], order: list[str]
) -> dict[str, Any]:
    """RTE/MAE per sequence, macro and micro, per GT range, and cap hits."""
    p = np.asarray(prediction, dtype=np.float64)
    g = np.asarray(truth, dtype=np.float64)
    r = rte(p, g)
    per_seq = sequence_means(r, sequences, order)
    out: dict[str, Any] = {
        "per_sequence_rte": dict(zip(order, per_seq.tolist(), strict=True)),
        "macro_rte": float(per_seq.mean()),
        "micro_rte": float(r.mean()),
        "median_rte": float(np.median(r)),
        "mae_s": float(np.abs(p - g).mean()),
        "per_sequence_mae_s": dict(
            zip(order, sequence_means(np.abs(p - g), sequences, order).tolist(), strict=True)
        ),
        "cap_hits": int((np.abs(p) >= fcwd.TTC_CAP_S - 1e-9).sum()),
        "negative_predictions": int((p < 0).sum()),
        "rows": int(len(g)),
    }
    out["by_gt_range"] = {
        name: {
            "rows": int(((g > lo) & (g <= hi)).sum()),
            "mean_rte": float(r[(g > lo) & (g <= hi)].mean())
            if ((g > lo) & (g <= hi)).any()
            else None,
        }
        for name, lo, hi in RANGES
    }
    return out


def score(out: Path, docs: Path) -> dict[str, Any]:
    """Verify the seal, join the targets, compute every metric and write the results."""
    from operational.sota_eval.fcwd_score import interpolate_truth

    seal = json.loads((out / "SEAL.json").read_text(encoding="utf-8"))
    predictions_path = out / seal["predictions"]
    if sha256_file(predictions_path) != seal["predictions_sha256"]:
        raise ValueError("sealed predictions changed after sealing")
    root = fcwd.fcwd_root()
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sequences = np.asarray([r["sequence_id"] for r in rows])
    anchors = np.asarray([int(r["anchor_us"]) for r in rows], dtype=np.int64)
    order = [f"FCWD{n}" for n in fcwd.SEQUENCES]
    truth_in = np.full(len(rows), np.nan)
    truth_ex = np.full(len(rows), np.nan)
    status: list[str] = []
    gt_files = {}
    for number in fcwd.SEQUENCES:
        path = root / "sealed_labels" / f"sequence_{number}" / "gt_ttc.csv"
        table = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
        gt_files[f"FCWD{number}"] = {"sha256": sha256_file(path), "rows": int(len(table))}
        times, targets = table[:, 1], table[:, 4]
        for i in np.flatnonzero(sequences == f"FCWD{number}"):
            value, state = interpolate_truth(times, targets, anchors[i] / 1e6)
            truth_in[i] = value
            truth_ex[i] = extrapolate_truth(times, targets, anchors[i] / 1e6)
    for i in range(len(rows)):
        status.append("ELIGIBLE" if math.isfinite(truth_in[i]) and truth_in[i] != 0 else "OUTSIDE")
    prior = {
        r["query_id"]: r
        for r in _prior_rows(PRIOR_SCORED, ("query_id", "truth_ttc_seconds", *PRIOR_METHODS))
    }
    committed = np.asarray(
        [float(prior[r["query_id"]]["truth_ttc_seconds"] or "nan") for r in rows], dtype=np.float64
    )
    both = np.isfinite(committed) & np.isfinite(truth_in)
    truth_check = {
        "max_abs_diff_vs_committed": float(np.abs(committed[both] - truth_in[both]).max()),
        "finite_pattern_equal": bool(np.array_equal(np.isfinite(committed), np.isfinite(truth_in))),
    }
    if truth_check["max_abs_diff_vs_committed"] > 1e-9 or not truth_check["finite_pattern_equal"]:
        raise ValueError(f"joined truths differ from the committed cohort: {truth_check}")
    arms = list(seal["arms"])
    preds = {a: np.asarray([float(r[a]) for r in rows]) for a in arms}
    for method in PRIOR_METHODS:
        preds[f"PRIOR_{method}"] = np.asarray([float(prior[r["query_id"]][method]) for r in rows])
    eligible = np.asarray([s == "ELIGIBLE" for s in status])
    both_truncated = np.asarray([float(r["both_truncated_current"]) > 0 for r in rows])
    extrap_ok = np.isfinite(truth_ex) & (truth_ex != 0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(order), size=(BOOTSTRAP_DRAWS, len(order)))
    cohorts = {
        "primary_in_support": (eligible, truth_in),
        "secondary_extrapolated": (extrap_ok, truth_ex),
    }
    results: dict[str, Any] = {}
    for cohort, (mask, truth) in cohorts.items():
        cohort_out: dict[str, Any] = {"rows": int(mask.sum()), "arms": {}}
        per_seq_cache = {}
        for name, values in preds.items():
            summary = summarize_arm(values[mask], truth[mask], sequences[mask], order)
            # Post-hoc descriptive split (addendum after scoring; not used for any choice).
            row_rte = rte(values[mask], truth[mask])
            saturated = both_truncated[mask]
            summary["posthoc_by_current_box_both_axes_truncated"] = {
                "truncated": {
                    "rows": int(saturated.sum()),
                    "mean_rte": float(row_rte[saturated].mean()),
                },
                "not_truncated": {
                    "rows": int((~saturated).sum()),
                    "mean_rte": float(row_rte[~saturated].mean()),
                    "median_rte": float(np.median(row_rte[~saturated])),
                },
            }
            per_seq = np.asarray([summary["per_sequence_rte"][s] for s in order])
            per_seq_cache[name] = per_seq
            summary["macro_rte_ci95"] = bootstrap_macro(per_seq, draws)
            cohort_out["arms"][name] = summary
        for reference in ("GEO_AREA", "PRIOR_public_Garl_event_lhr"):
            deltas = {}
            for name, per_seq in per_seq_cache.items():
                if name == reference:
                    continue
                diff = per_seq - per_seq_cache[reference]
                deltas[name] = {
                    "delta_macro_rte": float(diff.mean()),
                    "ci95": bootstrap_macro(diff, draws),
                }
            cohort_out[f"paired_delta_vs_{reference}"] = deltas
        learned = {}
        for base in ("GEO_LEARNED_EAP_ZS", "GEO_LEARNED_EAP_ZS_20HZ"):
            for suffix in ("", "__L50"):
                macros = [cohort_out["arms"][f"{base}_S{s}{suffix}"]["macro_rte"] for s in SEEDS]
                learned[f"{base}{suffix}"] = {
                    "seed_macro_rte": dict(zip([str(s) for s in SEEDS], macros, strict=True)),
                    "mean": float(np.mean(macros)),
                    "sd": float(np.std(macros, ddof=1)),
                }
        cohort_out["learned_seed_summary"] = learned
        results[cohort] = cohort_out
    payload = {
        "schema": "second_benchmark_fcwd_results_v1",
        "protocol_sha256": sha256_file(DOCS / "PROTOCOL.md"),
        "seal_sha256": sha256_file(out / "SEAL.json"),
        "predictions_sha256": seal["predictions_sha256"],
        "queries": len(rows),
        "sequences": order,
        "constant_eap_median_ttc_s": seal["constant_eap_median_ttc_s"],
        "gt_files": gt_files,
        "truth_check": truth_check,
        "diagnostics": {
            "n_frames_P20_distribution": {
                str(int(k)): int(v)
                for k, v in zip(
                    *np.unique([float(r["n_frames_P20"]) for r in rows], return_counts=True),
                    strict=True,
                )
            },
            "both_truncated_current_rows": int(
                sum(float(r["both_truncated_current"]) for r in rows)
            ),
            "both_truncated_current_rows_eligible": int(
                sum(
                    float(r["both_truncated_current"])
                    for r, e in zip(rows, eligible, strict=True)
                    if e
                )
            ),
        },
        "bootstrap": {
            "unit": "whole sequence",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "distinct_multisets": 10,
            "note": "3 clusters only: descriptive intervals",
        },
        "published_table_vii_rte_reported_not_reproduced": {
            k: dict(zip(order, v, strict=True)) for k, v in PUBLISHED_TABLE_VII.items()
        },
        "results": results,
    }
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "RESULTS.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    scored_header = [
        "query_id",
        "sequence_id",
        "anchor_us",
        "truth_in_support",
        "truth_extrapolated",
        "status",
        "n_frames_P20",
        "both_truncated_current",
        *preds,
    ]
    _write_csv(
        docs / "PREDICTIONS_SCORED.csv",
        scored_header,
        [
            [
                r["query_id"],
                r["sequence_id"],
                r["anchor_us"],
                _fmt(truth_in[i]),
                _fmt(truth_ex[i]),
                status[i],
                r["n_frames_P20"],
                r["both_truncated_current"],
                *[_fmt(preds[a][i]) for a in preds],
            ]
            for i, r in enumerate(rows)
        ],
    )
    summary_rows = []
    for cohort, data in results.items():
        for name, s in data["arms"].items():
            summary_rows.append(
                [
                    cohort,
                    name,
                    data["rows"],
                    *[_fmt(s["per_sequence_rte"][q]) for q in order],
                    _fmt(s["macro_rte"]),
                    _fmt(s["macro_rte_ci95"][0]),
                    _fmt(s["macro_rte_ci95"][1]),
                    _fmt(s["micro_rte"]),
                    _fmt(s["median_rte"]),
                    _fmt(s["mae_s"]),
                    s["cap_hits"],
                ]
            )
    _write_csv(
        docs / "SUMMARY.csv",
        [
            "cohort",
            "arm",
            "rows",
            *[f"rte_{q}" for q in order],
            "macro_rte",
            "macro_ci_lo",
            "macro_ci_hi",
            "micro_rte",
            "median_rte",
            "mae_s",
            "cap_hits",
        ],
        summary_rows,
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "score"))
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument("--docs", type=Path, default=DOCS)
    args = parser.parse_args(argv)
    if args.stage == "predict":
        seal = predict(args.out)
        print(json.dumps({k: seal[k] for k in ("predictions_sha256", "queries")}, indent=2))
    else:
        payload = score(args.out, args.docs)
        print(json.dumps(payload["truth_check"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
