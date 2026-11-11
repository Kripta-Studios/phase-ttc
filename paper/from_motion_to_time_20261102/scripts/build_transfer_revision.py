"""Regenerate corrected-label transfer evidence using bounded, saved projections only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

MODELS = ("PHASE_ONLY_TRAIN40_S7", "R_CTX_TRAIN40", "E_CTX_TRAIN40", "Garl_full")
NAMES = ("PHASE-TTC", "RGB context", "Event context", "Garl full")
ROOT = Path(__file__).resolve().parents[1]
Record = dict[str, Any]


def digest(path: Path) -> str:
    """Hash distributed bytes without normalizing line endings."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_rows(root: Path) -> tuple[list[Record], Record]:
    """Verify sealed projections, unchanged predictions, and bounded nearest targets."""
    audit = root / "audit_20261104"
    meta = json.loads((audit / "transfer_reassociation.json").read_text(encoding="utf-8"))
    paths = (audit / "corrected_transfer_candidates.csv", root / "data/transfer_queries.csv")
    for path, key in zip(paths, ("corrected_projection", "historical_projection"), strict=True):
        if digest(path) != meta[key + "_sha256"]:
            raise ValueError(f"{key} projection hash mismatch")
    data = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            data.append(list(csv.DictReader(handle)))
    rows, original = data
    identities = [row["query_id"] for row in rows]
    if identities != [row["query_id"] for row in original] or len(set(identities)) != len(rows):
        raise ValueError("Query identities/order changed or duplicated")
    for row, old in zip(rows, original, strict=True):
        if any(row[key] != old[key] for key in (*MODELS, "group_id", "cohort")):
            raise ValueError("Saved predictions or query grouping changed")
        if any(row["legacy_" + key] != old[key] for key in ("target_ttc", "target_exclusion")):
            raise ValueError("Historical target projection changed")
        if row["target_source_sha256"] != meta["corrected_target_source_sha256"][row["group_id"]]:
            raise ValueError("Target source hash differs from reassociation receipt")
        anchor = int(row["anchor_us"])
        before, after = (int(row[f"neighbor_{side}_us"]) for side in ("before", "after"))
        if anchor != int(row["query_id"].rsplit(":", 1)[1]) or before > after:
            raise ValueError("Invalid anchor or neighbor ordering")
        if before != after and not before < anchor <= after:
            raise ValueError("Neighbors do not bracket anchor")
        timestamp, target = min(
            [(int(row[f"neighbor_{s}_us"]), float(row[f"neighbor_{s}_ttc"]))
             for s in ("before", "after")], key=lambda point: (abs(point[0] - anchor), point[0]),
        )
        distance = abs(timestamp - anchor)
        reason = "TTC_TIMESTAMP_MISMATCH" if distance > 50000 else (
            "TTC_INVALID_OR_ZERO" if not math.isfinite(target) or target == 0 else ""
        )
        if (int(row["target_timestamp_us"]) != timestamp or int(row["target_delta_us"]) != distance
                or row["target_exclusion"] != reason or
                (row["target_ttc"] != "" if reason else float(row["target_ttc"]) != target)):
            raise ValueError(f"Incorrect nearest target or exclusion: {row['query_id']}")
        if not all(math.isfinite(float(row[key])) for key in MODELS):
            raise ValueError("Nonfinite saved prediction")
    return rows, meta


def error_summary(error: np.ndarray, target: np.ndarray) -> Record:
    """Compute pooled absolute and relative errors with linear quantiles."""
    return {"n": len(error), "MAE_s": float(error.mean()),
            "RTE_pct": float(np.mean(100 * error / np.abs(target))),
            "median_AE_s": float(np.median(error)),
            "p95_AE_s": float(np.quantile(error, .95, method="linear")),
            "max_AE_s": float(error.max()),
            "counts_above_s": {str(t): int((error > t).sum()) for t in (1, 3, 5, 10)}}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, groups: np.ndarray, *,
                     draws: int, seed: int, macro: bool = False) -> Record:
    """Resample complete sequence groups, pairing methods in each selected group."""
    indices = [np.flatnonzero(groups == group) for group in sorted(set(groups))]
    selected = np.random.default_rng(seed).integers(0, len(indices), (draws, len(indices)))
    if macro:
        values = np.array([(a[ix] - b[ix]).mean() for ix in indices])
        samples, point, metrics = values[selected].mean(1)[:, None], [values.mean()], ("mean",)
    else:
        def estimates(x: np.ndarray) -> np.ndarray:
            return np.array([x.mean(), np.median(x), np.quantile(x, .95, method="linear")])
        samples = np.array([estimates(a[ix]) - estimates(b[ix]) for ix in
                            (np.concatenate([indices[g] for g in pick]) for pick in selected)])
        point, metrics = estimates(a) - estimates(b), ("mean", "median", "p95")
    ci = np.quantile(samples, [.025, .975], axis=0, method="linear")
    return {"draws": draws, "seed": seed, "groups": len(indices),
            "unit": "complete sequence group", "rng": "numpy.default_rng / PCG64",
            "quantiles": "linear percentile 95%", "estimand": "equal-group macro" if macro
            else "pooled queries after whole-group resampling",
            **{key: {"point": float(point[i]), "CI95": ci[:, i].tolist()}
               for i, key in enumerate(metrics)}}


def cohort_statistics(rows: list[Record]) -> Record:
    """Keep eligibility, pooled scores, sequence scores, and estimands distinct."""
    valid = [row for row in rows if not row["target_exclusion"]]
    target = np.array([float(row["target_ttc"]) for row in valid])
    groups = np.array([row["group_id"] for row in valid])
    errors = {model: np.abs(np.array([float(row[model]) for row in valid]) - target)
              for model in MODELS}
    per_sequence = {group: {model: error_summary(error[groups == group], target[groups == group])
                           for model, error in errors.items()} for group in sorted(set(groups))}
    return {"candidates": len(rows), "eligible": len(valid), "sequences": len(per_sequence),
            "exclusions": dict(Counter(row["target_exclusion"] for row in rows
                                       if row["target_exclusion"])),
            "pooled": {model: error_summary(error, target) for model, error in errors.items()},
            "per_sequence": per_sequence,
            "phase_sequence_wins": {model: {key: sum(g[MODELS[0]][key] < g[model][key]
                                     for g in per_sequence.values())
                                     for key in ("MAE_s", "RTE_pct")}
                                     for model in MODELS[1:]},
            "equal_group_macro": {model: {key: float(np.mean([g[model][key]
                                   for g in per_sequence.values()]))
                                   for key in ("MAE_s", "RTE_pct")}
                                   for model in MODELS},
            "phase_minus_rgb_pooled": paired_bootstrap(errors[MODELS[0]], errors[MODELS[1]],
                                                       groups, draws=10000, seed=20261028),
            "phase_minus_garl_macro": {key: paired_bootstrap(errors[MODELS[0]] * scale,
                                      errors[MODELS[3]] * scale, groups, draws=2000,
                                      seed=7, macro=True) for key, scale in
                                      (("MAE_s", 1), ("RTE_pct", 100 / np.abs(target)))}}


def table(root: Path, name: str, columns: str, heading: str, rows: list[str],
          caption: str, *, long: bool = False) -> None:
    """Write an additive table; historical generated files are never opened for writing."""
    environment = "longtable" if long else "tabular"
    lines = ([r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}"] if long else
             [r"\begin{table}[htbp]", r"\centering\small", r"\resizebox{\linewidth}{!}{%"])
    lines += [rf"\begin{{{environment}}}{{{columns}}}"]
    if long:
        lines += [rf"\caption{{{caption}}}\label{{tab:audit-transfer-{name}}}\\"]
    lines += [r"\toprule", heading + r" \\", r"\midrule", *rows, r"\bottomrule",
              rf"\end{{{environment}}}"]
    if not long:
        lines += ["}", rf"\caption{{{caption}}}",
                  rf"\label{{tab:audit-transfer-{name}}}", r"\end{table}"]
    else:
        lines += [r"\endgroup"]
    filenames = {"sequences": "sequence_full"}
    (root / f"tables/audit_transfer_{filenames.get(name, name)}.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def render(root: Path, cohorts: Record, rows: list[Record]) -> None:
    """Render corrected-label tables and standalone PDF figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9})
    extended, historical = cohorts["extended_new"], cohorts["historical46"]
    macros = [rf"\newcommand{{\AuditTransferCount}}{{{extended['eligible']}}}"]
    for model, prefix in zip(MODELS[:2], ("Fusion", "Rgb"), strict=True):
        for suffix, key, decimals in (("MAE", "MAE_s", 3), ("Median", "median_AE_s", 3),
                                     ("Pninetyfive", "p95_AE_s", 3), ("RTE", "RTE_pct", 2),
                                     ("Max", "max_AE_s", 2)):
            value = f"{extended['pooled'][model][key]:.{decimals}f}"
            macros.append(r"\newcommand{\Audit" + prefix + suffix + "}{" + value
                          + (r"\%" if suffix == "RTE" else "") + "}")
    for key, prefix in (("mean", "Mean"), ("median", "Median"), ("p95", "Pninetyfive")):
        result = extended["phase_minus_rgb_pooled"][key]
        for suffix, value in zip(("Delta", "Low", "High"),
                                 (result["point"], *result["CI95"]), strict=True):
            macros.append(r"\newcommand{\Audit" + prefix + suffix + "}{" + f"{value:+.3f}" + "}")
    (root / "tables/audit_transfer_macros.tex").write_text("\n".join(macros) + "\n")
    summaries = []
    for scope, data in (("Corrected extended", extended), ("Historical46", historical)):
        for model, name in zip(MODELS, NAMES, strict=True):
            p, m = data["pooled"][model], data["equal_group_macro"][model]
            summaries.append(f"{scope}: {name} & {p['n']} & " + " & ".join(
                f"{v:.3f}" for v in (p["MAE_s"], p["median_AE_s"], p["p95_AE_s"],
                                     p["RTE_pct"], m["MAE_s"], m["RTE_pct"])) + r" \\")
    table(root, "summary", "lrrrrrrr",
          r"Cohort / model & $n$ & MAE & Median & p95 & RTE (\%) & Macro MAE & Macro RTE (\%)",
          summaries, "Corrected-label transfer: 460 eligible of 508 extended candidates "
          "across 32 sequences. Historical46 remains separate (46 queries, three sequences). "
          "Absolute errors are seconds; macro gives equal weight to sequences.")
    boot_rows = []
    for scope, data in (("Extended460", extended), ("Historical46", historical)):
        for metric in ("mean", "median", "p95"):
            item = data["phase_minus_rgb_pooled"][metric]
            boot_rows.append(f"{scope} & PHASE minus RGB & {metric} & {item['point']:+.3f} & "
                             f"[{item['CI95'][0]:+.3f}, {item['CI95'][1]:+.3f}]" + r" \\")
        for metric, result in data["phase_minus_garl_macro"].items():
            item = result["mean"]
            label = metric.replace("_", r"\_")
            boot_rows.append(f"{scope} & PHASE minus Garl & Macro {label} & "
                             f"{item['point']:+.3f} & "
                             f"[{item['CI95'][0]:+.3f}, {item['CI95'][1]:+.3f}]" + r" \\")
    table(root, "bootstrap", "lllrr",
          r"Cohort & Paired contrast & Estimand & Difference & 95\% interval",
          boot_rows, "Whole-sequence paired percentile intervals. PHASE--RGB uses 10,000 draws, "
          "seed 20261028, for pooled query errors; PHASE--Garl uses 2,000 draws, seed 7, "
          "for equal-sequence means. Linear quantiles; negative favors PHASE. MAE, median and "
          "p95 use seconds; RTE uses percentage points. Exposed development cohorts.")
    sequence_rows, eligibility = [], []
    for group, results in extended["per_sequence"].items():
        group_rows = [r for r in rows if r["cohort"] == "extended_new" and r["group_id"] == group]
        sequence_rows.append(group + " & " + str(results[MODELS[0]]["n"]) + " & " + " & ".join(
            f"{results[model][key]:.3f}" for model in MODELS
            for key in ("MAE_s", "RTE_pct")) + r" \\")
        eligibility.append(f"{group} & {len(group_rows)} & "
                           f"{sum(not row['legacy_target_exclusion'] for row in group_rows)} & "
                           f"{results[MODELS[0]]['n']}" + r" \\")
    table(root, "sequences", "lrrrrrrrrr",
          r"Sequence & $n$ & PHASE MAE & RTE & RGB MAE & RTE & Event MAE & RTE & Garl MAE & RTE",
          sequence_rows, "Corrected extended-cohort errors by all 32 sequence groups. "
          "MAE in seconds; RTE in percent.", long=True)
    table(root, "eligibility", "lrrr",
          "Sequence & Candidates & Historical eligible & Corrected eligible",
          eligibility, "All 508 extended candidates retained. Nearest TTC within 50 ms inclusive, "
          "earlier ties; corrected labels yield 460 eligible queries. "
          "Historical46 retains all 46 and is scored separately.", long=True)
    fig, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
    x = np.arange(3)
    for i, (model, name) in enumerate(zip(MODELS, NAMES, strict=True)):
        ax.bar(x + (i - 1.5) * .19, [extended["pooled"][model][k] for k in
               ("MAE_s", "median_AE_s", "p95_AE_s")], .19, label=name)
    ax.set(xticks=x, xticklabels=("Mean", "Median", "95th percentile"), ylabel="Absolute error (s)",
           title="Corrected labels: 460 eligible queries / 508 candidates")
    ax.legend(ncols=2)
    fig.savefig(root / "figures/audit_transfer_summary.pdf")
    plt.close(fig)
    valid = [r for r in rows if r["cohort"] == "extended_new" and not r["target_exclusion"]]
    fig, ax = plt.subplots(figsize=(6.6, 3.5), layout="constrained")
    for model, name in zip(MODELS, NAMES, strict=True):
        error = np.sort([abs(float(r[model]) - float(r["target_ttc"])) for r in valid])
        ax.step(error, np.arange(len(error), 0, -1) / len(error), where="pre", label=name)
    ax.set(xscale="log", yscale="log", xlabel="Absolute error (s)",
           ylabel="Fraction at or above error",
           title="Corrected labels: extended cohort (460 queries)")
    ax.legend()
    fig.savefig(root / "figures/audit_transfer_exceedance.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    groups = list(extended["per_sequence"])
    for model, name in zip(MODELS, NAMES, strict=True):
        ax.plot(range(len(groups)), [extended["per_sequence"][g][model]["RTE_pct"] for g in groups],
                ".-", label=name, linewidth=.8)
    ax.set(xticks=range(len(groups)), xticklabels=groups, yscale="log",
           ylabel="Mean relative TTC error (%)",
           title="Corrected labels: all 32 extended sequence groups")
    ax.tick_params(axis="x", labelrotation=75, labelsize=7)
    ax.legend(ncols=2)
    fig.savefig(root / "figures/audit_transfer_sequences.pdf")
    plt.close(fig)


def main() -> None:
    """Reproduce additive corrected evidence without changing historical artifacts."""
    rows, metadata = verified_rows(ROOT)
    cohorts = {cohort: cohort_statistics([r for r in rows if r["cohort"] == cohort])
               for cohort in ("historical46", "extended_new")}
    if [(d["candidates"], d["eligible"]) for d in cohorts.values()] != [(46, 46), (508, 460)]:
        raise ValueError("Corrected cohort counts differ from the audited reassociation")
    historical = json.loads((ROOT / "data/late_results.json").read_text())["rows"]
    for row in historical:
        if row.get("cohort") == "historical46" and row.get("model") in MODELS:
            for key in ("MAE_s", "RTE_pct", "median_AE_s", "p95_AE_s"):
                if not np.isclose(cohorts["historical46"]["pooled"][row["model"]][key],
                                  row[key], rtol=1e-12, atol=1e-12):
                    raise ValueError("Historical46 does not reproduce preserved metrics")
    report = {"schema": "corrected_transfer_statistics_v1", "cohorts": cohorts,
              "source_projection_sha256": metadata["corrected_projection_sha256"],
              "reassociation_sha256": digest(ROOT / "audit_20261104/transfer_reassociation.json"),
              "builder_sha256": digest(Path(__file__)), "predictions": "Unchanged saved values",
              "scope": "Corrected targets, complete fixed candidate set; no training or inference"}
    (ROOT / "audit_20261104/corrected_transfer_statistics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    render(ROOT, cohorts, rows)
    print(json.dumps({"eligible": {k: v["eligible"] for k, v in cohorts.items()},
                      "corrected_pooled": cohorts["extended_new"]["pooled"],
                      "phase_minus_rgb": cohorts["extended_new"]["phase_minus_rgb_pooled"]},
                     indent=2))


if __name__ == "__main__":
    main()
