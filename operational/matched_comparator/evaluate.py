"""Score the matched arms on TRAIN40 OOF, paired CIs, in-sample risk (S1/S2) and fold projections.

``python -m operational.matched_comparator.evaluate --geometry-work <dir> --work <dir>
--docs <dir>``
reads only finished job files and label-free Test12 inputs (never private labels).  Every Overall
is cross-checked with the pinned official scorer.  Writes ``RESULTS.json`` and CSV tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from operational.phase_geometry import geometry
from operational.phase_geometry.metrics import (
    BANDS,
    band_index,
    bootstrap_overall,
    json_safe,
    official_check,
    phase_to_ttc,
    row_mid,
    summarize,
)

from . import projection
from .features import garl_phase, release_ttc
from .run_cv import FOLDS, SEEDS, arm_name, load_npz, oof

EXISTING = ("REL", "GEO_LEARNED", "RGB_GEO_BAND")
RHO_MAX = 1.25
PI_MIN = 0.75
FAILURE_MAX = 0.01


def existing_oof(geometry_work: Path, name: str, seed: int, n: int) -> NDArray[np.float64]:
    """OOF TTC of a phase_geometry variant (unchanged job files)."""
    out = np.full(n, np.nan)
    for fold in FOLDS:
        part = load_npz(geometry_work / "jobs" / f"{name}__F{fold}__S{seed}__M32.npz")
        out[part["rows"]] = part["ttc"]
    if not np.isfinite(out).all():
        raise ValueError(f"Incomplete OOF for {name}")
    return out


def job_hashes(geometry_work: Path, work: Path, names: list[str]) -> dict[str, str]:
    """SHA-256 of every OOF job file used."""
    out: dict[str, str] = {}
    for name in names:
        for fold in FOLDS:
            for seed in SEEDS:
                if name in EXISTING:
                    path = geometry_work / "jobs" / f"{name}__F{fold}__S{seed}__M32.npz"
                else:
                    path = work / "jobs" / f"{name}__F{fold}__S{seed}.npz"
                out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def mean_disagreement(a_phase: NDArray[Any], b_phase: NDArray[Any], ok: NDArray[Any]) -> float:
    """Mean |phase difference| x 1e4 over rows ``ok``."""
    return float(np.mean(np.abs(a_phase[ok] - b_phase[ok])) * 1e4)


def in_sample_s1(
    train: dict[str, NDArray[Any]],
    test: dict[str, NDArray[Any]],
    garl_train: NDArray[Any],
    garl_test: NDArray[Any],
) -> dict[str, Any]:
    """S1: test/train contraction of Garl pairs relative to independent reference pairs."""
    estimators = {}
    for side, arrays, heights in (("train", train, garl_train), ("test", test, garl_test)):
        phase, valid = garl_phase(heights)
        estimators[side] = {
            "GARL": phase,
            "GEO": geometry.raw_column(arrays["geometry_raw"], "phi_trunc_aware"),
            "RCTX": arrays["features"][:, 0].astype(np.float64),
            "ECTX": arrays["features"][:, 10].astype(np.float64),
            "valid": valid,
        }
    pairs = {}
    for a, b in (
        ("GARL", "GEO"),
        ("GARL", "RCTX"),
        ("GARL", "ECTX"),
        ("RCTX", "GEO"),
        ("ECTX", "GEO"),
        ("RCTX", "ECTX"),
    ):
        values = {}
        for side in ("train", "test"):
            est = estimators[side]
            ok = est["valid"] if "GARL" in (a, b) else np.ones(len(est["valid"]), dtype=bool)
            values[side] = mean_disagreement(est[a], est[b], ok)
            values[f"{side}_rows"] = int(ok.sum())
        values["ratio"] = values["test"] / values["train"]
        pairs[f"{a}-{b}"] = values
    garl_ratio = float(
        np.median([pairs[p]["ratio"] for p in ("GARL-GEO", "GARL-RCTX", "GARL-ECTX")])
    )
    reference = float(np.median([pairs[p]["ratio"] for p in ("RCTX-GEO", "ECTX-GEO", "RCTX-ECTX")]))
    return {
        "pairs": pairs,
        "garl_ratio_median": garl_ratio,
        "reference_ratio_median": reference,
        "rho": garl_ratio / reference,
        "threshold": RHO_MAX,
        "passed": bool(garl_ratio / reference <= RHO_MAX),
        "train_valid_fraction": float(estimators["train"]["valid"].mean()),
        "test_valid_fraction": float(estimators["test"]["valid"].mean()),
    }


def score_arms(
    scorer: ModuleType,
    learned: dict[str, dict[int, NDArray[np.float64]]],
    zero: dict[str, NDArray[np.float64]],
    truth: NDArray[Any],
    groups: NDArray[Any],
) -> tuple[list[dict[str, Any]], dict[str, NDArray[np.float64]], list[float]]:
    """Official-formula scores, seed-mean per-row MiD and scorer cross-check gaps."""
    bands = band_index(truth)
    checks: list[float] = []
    mids: dict[str, NDArray[np.float64]] = {}
    rows: list[dict[str, Any]] = []
    for name, by_seed in learned.items():
        scores, seed_mid = [], []
        for seed, ttc in by_seed.items():
            checks.append(official_check(scorer, truth, ttc))
            scores.append({"seed": seed, **summarize(truth, ttc)})
            seed_mid.append(row_mid(truth, ttc))
        mids[name] = np.mean(seed_mid, axis=0)
        overall = np.asarray([s["overall_MiD"] for s in scores])
        rows.append(
            {
                "arm": name,
                "kind": "learned",
                "overall_mean": float(overall.mean()),
                "overall_sd": float(overall.std(ddof=1)),
                **{f"overall_seed{s['seed']}": s["overall_MiD"] for s in scores},
                **{f"MiD{b}": float(np.mean([s[f"MiD{b}"] for s in scores])) for b in BANDS},
                "failed_rate": float(np.mean([s["failed_rate"] for s in scores])),
            }
        )
    for name, ttc in zero.items():
        checks.append(official_check(scorer, truth, ttc))
        score = summarize(truth, ttc)
        mids[name] = row_mid(truth, ttc)
        rows.append(
            {
                "arm": name,
                "kind": "zero_learning_in_sample" if name.startswith("GARL") else "zero_learning",
                "overall_mean": score["overall_MiD"],
                "overall_sd": None,
                **{f"MiD{b}": score[f"MiD{b}"] for b in BANDS},
                "failed_rate": score["failed_rate"],
                "invalid_mid_rows": score["invalid_mid"],
            }
        )
    for row in rows:
        boot = bootstrap_overall(mids[row["arm"]], bands, groups)
        row["overall_ci95"] = [boot["ci95_low"], boot["ci95_high"]]
    return rows, mids, checks


def paired(
    mids: dict[str, NDArray[np.float64]],
    pairs: list[tuple[str, str, str]],
    truth: NDArray[Any],
    groups: NDArray[Any],
) -> list[dict[str, Any]]:
    """Paired sequence-bootstrap contrasts and 12-sequence resolution."""
    bands = band_index(truth)
    out = []
    for variant, comparator, question in pairs:
        delta = mids[variant] - mids[comparator]
        out.append(
            {
                "variant": variant,
                "comparator": comparator,
                "question": question,
                "rows_defined": int(np.isfinite(delta).sum()),
                **bootstrap_overall(delta, bands, groups),
                "resolution12": projection.resolution12(delta, bands, groups),
            }
        )
    return out


def contrast_pairs(sources: list[str], suffix: str = "") -> list[tuple[str, str, str]]:
    """Pre-registered contrasts of PROTOCOL section 5 (``suffix`` = ``_S8`` for amendment 1)."""
    rgb, geo = f"RGB_GEO_BAND{suffix}", f"GEO_LEARNED{suffix}"
    pairs = [(rgb, geo, "RGB fields given geometry")]
    for source in sources:
        g = f"GARL_{source}"
        b, r = arm_name(source, "GEO_R") + suffix, arm_name(source, "R") + suffix
        pairs += [
            (b, geo, f"Garl {source} added to box geometry"),
            (rgb, b, f"PHASE RGB branch vs Garl {source}, same box information"),
            (b, g, f"box geometry added to Garl {source}"),
            (geo, g, f"box geometry alone vs Garl {source}"),
            (r, g, f"learned recalibration of Garl {source}"),
        ]
    return pairs


def projections_for(
    mids: dict[str, NDArray[np.float64]],
    truth: NDArray[Any],
    groups: NDArray[Any],
    anchors: dict[str, str],
) -> list[dict[str, Any]]:
    """Multiplicative projections of every arm with the REL and RGB_GEO_BAND anchors."""
    bands = band_index(truth)
    return [
        {"arm": name, **projection.multiplicative(mids[name], mids[key], bands, groups, anchor)}
        for name in mids
        for anchor, key in anchors.items()
    ]


def failure_summary(heights: dict[str, NDArray[Any]], rows: NDArray[Any] | slice) -> dict[str, Any]:
    """Missing and invalid Garl outputs on the given TRAIN40 rows and on Test12."""
    train = heights["train_heights"][rows]
    return {
        "train_rows": int(len(train)),
        "train_missing_rows": int(np.isnan(train).any(axis=1).sum()),
        "test_physical_missing_rows": int(np.isnan(heights["test_heights"]).any(axis=1).sum()),
        "train_invalid_fraction": float(1 - garl_phase(train)[1].mean()),
        "test_invalid_fraction": float(1 - garl_phase(heights["test_heights"])[1].mean()),
    }


def test_s8(test: dict[str, NDArray[Any]]) -> NDArray[np.int64]:
    """Amendment 2: the S8 rule on the physical Test12 queries (label-free public inputs)."""
    import io

    from operational.phase_geometry.data import read_member

    from .features import s8_rows

    columns = ["sample_token", "sequence_id", "track_id", "timestamp_us"]
    frame = pd.read_parquet(io.BytesIO(read_member("test_inputs")), columns=columns)
    frame = frame.set_index("sample_token").loc[test["sample_token"].astype(str)]
    return s8_rows(
        frame["sequence_id"].to_numpy(),
        frame["track_id"].to_numpy(),
        frame["timestamp_us"].to_numpy(),
        test["sample_token"],
    )


def s2_gate(projections: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    """S2: projected in-sample Garl FULL level versus its official 45.03."""
    values = {p["anchor"]: p["point"] for p in projections if p["arm"] == arm}
    official = projection.OFFICIAL["GARL_PAPER"]["overall"]
    pi = {k: v / official for k, v in values.items()}
    return {
        "arm": arm,
        "projected": values,
        "official": official,
        "pi": pi,
        "threshold": PI_MIN,
        "passed": bool(pi and all(v >= PI_MIN for v in pi.values())),
    }


DEVIATIONS = (
    "Enmienda 1 (PROTOCOL §10): FULL en TRAIN40 sólo en la submuestra S8 por coste medido; "
    "escrita después de ver los resultados EVENT en folds (declarado en el protocolo).",
    "Enmienda 2 (PROTOCOL §11): S1 de FULL con la regla S8 en ambos lados (TRAIN40 y Test12) "
    "tras detectar en una ejecución en seco con EVENT que la composición de S8 sola mueve ρ de "
    "1,105 a 1,298; ningún valor FULL de TRAIN40 se había visto.",
    "Durante la ejecución, `ruff format` reformateó los módulos (sin cambio de comportamiento) "
    "y `garl_runner.py` ganó la opción `--tag`; los hashes de los recibos son los del código "
    "final, no necesariamente los de cada proceso que ya estaba en marcha.",
    "Paridad del ejecutor y prueba de rendimiento (sección 3 del protocolo) se hicieron antes "
    "de escribirlo; no producen métricas de precisión.",
    "La lectura de B frente a la proyección de `RGB_GEO_BAND_S8` (referencia igualada en "
    "tamaño de entrenamiento, REPORT §7) se añadió tras la enmienda 1, al constatar en la "
    "ejecución en seco que S8 penaliza el entrenamiento; no cambia la regla de empaquetado.",
)


def main() -> None:  # noqa: PLR0915 - one linear pipeline of pre-registered outputs
    """Compute every table and write RESULTS.json and CSV files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--geometry-work", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--no-s8", action="store_true", help="skip the amendment-1 S8 section")
    args = parser.parse_args()
    from operational.garl_comparison.mid import load_scorer
    from operational.phase_geometry.data import verified_file

    from .run_s8 import ARMS as S8_ARMS
    from .run_s8 import oof_s8, s8

    root = args.root.resolve()
    gwork, work, docs = args.geometry_work.resolve(), args.work.resolve(), args.docs.resolve()
    scorer = load_scorer(verified_file("official_scorer", root))
    train = load_npz(gwork / "train_inputs.npz")
    test = load_npz(gwork / "test_inputs.npz")
    truth = train["target_ttc"]
    bands = band_index(truth)
    groups = train["group_id"]
    n = len(truth)
    anchors = {"REL": "REL", "RGB_GEO_BAND": "RGB_GEO_BAND"}

    # Section 5 on all 88,744 rows (EVENT only; FULL exists only on S8 after amendment 1).
    heights = {"EVENT": load_npz(work / "garl_EVENT_heights.npz")}
    receipts = {"EVENT": json.loads((work / "garl_EVENT_heights.json").read_text("utf-8"))}
    learned: dict[str, dict[int, NDArray[np.float64]]] = {
        name: {s: existing_oof(gwork, name, s, n) for s in SEEDS} for name in EXISTING
    }
    for arm in ("R", "GEO_R"):
        name = arm_name("EVENT", arm)
        learned[name] = {s: oof(work, name, s, n) for s in SEEDS}
    zero = {
        "GEO_AREA": phase_to_ttc(geometry.raw_column(train["geometry_raw"], "phi_area")),
        "GARL_EVENT": release_ttc(heights["EVENT"]["train_heights"]),
    }
    rows, mids, checks = score_arms(scorer, learned, zero, truth, groups)
    contrasts = paired(mids, contrast_pairs(["EVENT"]), truth, groups)
    projections = projections_for(mids, truth, groups, anchors)
    event_h = heights["EVENT"]
    s1 = {"EVENT": in_sample_s1(train, test, event_h["train_heights"], event_h["test_heights"])}
    failures = {"EVENT": failure_summary(event_h, slice(None))}
    hashes = job_hashes(gwork, work, list(learned))

    s8_block: dict[str, Any] = {}
    full_ok = False
    if not args.no_s8 and (work / "garl_FULL_heights.npz").exists():
        heights["FULL"] = load_npz(work / "garl_FULL_heights.npz")
        receipts["FULL"] = json.loads((work / "garl_FULL_heights.json").read_text("utf-8"))
        full_h = heights["FULL"]
        idx = s8(train)
        failures["FULL"] = failure_summary(full_h, idx)
        full_ok = (
            failures["FULL"]["train_missing_rows"] <= FAILURE_MAX * len(idx)
            and failures["FULL"]["train_invalid_fraction"] <= FAILURE_MAX
            and failures["FULL"]["test_invalid_fraction"] <= FAILURE_MAX
            and failures["FULL"]["test_physical_missing_rows"] == 0
        )
        sub = {k: v[idx] for k, v in train.items()}
        learned8 = {name: {s: oof_s8(work, name, s, n)[idx] for s in SEEDS} for name in S8_ARMS}
        for name in EXISTING:
            learned8[f"{name}_FULLDATA"] = {s: learned[name][s][idx] for s in SEEDS}
        zero8 = {
            "GARL_FULL": release_ttc(full_h["train_heights"][idx]),
            "GARL_EVENT": release_ttc(event_h["train_heights"][idx]),
        }
        rows8, mids8, checks8 = score_arms(scorer, learned8, zero8, truth[idx], groups[idx])
        checks += checks8
        pairs8 = contrast_pairs(["FULL", "EVENT"], "_S8")
        pairs8.append(("RGB_GEO_BAND_S8", "RGB_GEO_BAND_FULLDATA", "training-size cost of S8"))
        pairs8.append(("GEO_LEARNED_S8", "GEO_LEARNED_FULLDATA", "training-size cost of S8"))
        anchors8 = {"REL": "REL_FULLDATA", "RGB_GEO_BAND": "RGB_GEO_BAND_FULLDATA"}
        projections8 = projections_for(mids8, truth[idx], groups[idx], anchors8)
        # Amendment 2: the S8 rule on both sides (gate); amendment-1 literal variant descriptive.
        tidx = test_s8(test)
        tsub = {k: (v[tidx] if len(v) == len(test["sample_token"]) else v) for k, v in test.items()}
        for source, h in (("FULL", full_h), ("EVENT_S8", event_h)):
            s1[source] = in_sample_s1(sub, tsub, h["train_heights"][idx], h["test_heights"][tidx])
            s1[f"{source}_S8_TRAIN_ALL_TEST_DESCRIPTIVE"] = in_sample_s1(
                sub, test, h["train_heights"][idx], h["test_heights"]
            )
        folds, counts = np.unique(train["fold"][idx], return_counts=True)
        s8_block = {
            "rows": int(len(idx)),
            "test_physical_rows_s8": int(len(tidx)),
            "rows_by_fold": {int(k): int(v) for k, v in zip(folds, counts, strict=True)},
            "main": rows8,
            "contrasts": paired(mids8, pairs8, truth[idx], groups[idx]),
            "projections_multiplicative": projections8,
            "S2_official_anchor": s2_gate(projections8, "GARL_FULL"),
        }
        hashes.update(job_hashes(gwork, work, list(S8_ARMS)))
    max_check = float(max(checks))
    if max_check > 1e-6:
        raise ValueError(f"Vectorized Overall differs from the official scorer by {max_check}")
    chosen = "FULL" if full_ok else "EVENT"
    s2 = s8_block.get("S2_official_anchor", {}) if chosen == "FULL" else {}
    sound = bool(s1[chosen]["passed"] and (s2.get("passed", False) if chosen == "FULL" else True))
    decision = {
        "source": chosen,
        "full_available_and_within_failure_limit": bool(full_ok),
        "S1_passed": s1[chosen]["passed"],
        "S2_passed": s2.get("passed") if chosen == "FULL" else None,
        "package_B": sound,
        "arm": "GARL_FULL_GEO_BAND_S8" if chosen == "FULL" else arm_name("EVENT", "GEO_R"),
        "training_rows": "S8" if chosen == "FULL" else "ALL",
    }
    results = {
        "schema": "matched_comparator_results_v2",
        "protocol_sha256": hashlib.sha256((docs / "PROTOCOL.md").read_bytes()).hexdigest(),
        "official_scorer_max_abs_overall_difference": max_check,
        "official_scorer_checks": len(checks),
        "rows": n,
        "sequences": len(set(groups.tolist())),
        "band_counts": {b: int((bands == i).sum()) for i, b in enumerate(BANDS)},
        "main": rows,
        "contrasts": contrasts,
        "projections_multiplicative": projections,
        "S1_in_sample_contraction": s1,
        "S2_official_anchor": s2,
        "garl_failures": failures,
        "garl_inputs": receipts,
        "s8": s8_block,
        "decision": decision,
        "deviations": list(DEVIATIONS),
        "job_sha256": hashes,
        "test12_labels_read": False,
    }
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "RESULTS.json").write_text(json.dumps(json_safe(results), indent=2) + "\n", "utf-8")
    s8_rows_table = [{**r, "arm": r["arm"] + " [S8]"} for r in s8_block.get("main", [])]
    table = pd.DataFrame(rows + s8_rows_table)
    table.to_csv(docs / "MAIN_TABLE.csv", index=False)
    all_contrasts = contrasts + s8_block.get("contrasts", [])
    pd.DataFrame([{k: v for k, v in c.items() if k != "band_ci95"} for c in all_contrasts]).to_csv(
        docs / "CONTRASTS.csv", index=False
    )
    print(table[["arm", "overall_mean", "MiDc", "MiDs", "MiDl", "MiDn"]].to_string())
    summary = {"decision": decision, "rho": {k: v["rho"] for k, v in s1.items()}, "S2": s2}
    print(json.dumps(json_safe(summary), indent=1))


if __name__ == "__main__":
    main()
