"""Regenerate same-hardware tables from complete, hash-bound real-inference ledgers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/local_hardware_20261112"
LABELS = {
    "phase": "PHASE-TTC",
    "garl_full": "Garl full release",
    "garl_ttc_only": "Garl decoder bypass",
    "garl_event": "Garl event-only",
    "garl_visual": "Garl RGB-only",
    "phase_native_raw": "PHASE native RAW",
    "garl_full_raw": "Garl full RAW",
    "garl_ttc_only_raw": "Garl bypass RAW",
}


def digest(path: Path) -> str:
    """Hash exact bytes, including line endings."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Reject partial experiments and stale protocol or measurement files."""
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete":
        raise ValueError("Incomplete experiment")
    if (path / "profiles.json").exists() and json.loads(
        (path / "profiles.json").read_text(encoding="utf-8")
    ) != summary.get("profiles"):
        raise ValueError("Separate profiles and completed summary disagree")
    for name, key in (("protocol.json", "protocol_sha256"), ("trials.jsonl", "trials_sha256")):
        if digest(path / name) != summary[key]:
            raise ValueError(f"Changed experiment ledger or protocol: {name}")
    protocol = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
    for name, expected in protocol.get("source_sha256", {}).items():
        snapshot = path / "executed_source" / (name.replace("\\", "/") + ".txt")
        if not snapshot.is_file() or digest(snapshot) != expected:
            raise ValueError(f"Executed source snapshot differs: {name}")
    return (
        summary,
        protocol,
        [
            json.loads(line)
            for line in (path / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        ],
    )


def validate_population(
    rows: list[dict[str, Any]], models: list[str], queries: list[str], repeats: int
) -> None:
    """Require every planned model/query/repetition, including unavailable predictions."""
    expected = {(m, q, i) for m in models for q in queries for i in range(repeats)}
    actual = [(r["model"], r["query"], r["repeat"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Incomplete or duplicated experimental population")
    for row in rows:
        for metric in ("total_ms", "inference_ms", "preparation_ms"):
            if metric in row and (not np.isfinite(row[metric]) or row[metric] < 0):
                raise ValueError("Invalid measured timing")
        if {"preparation_ms", "inference_ms"} <= row.keys() and not np.isclose(
            row["total_ms"], row["preparation_ms"] + row["inference_ms"], rtol=1e-10, atol=1e-7
        ):
            raise ValueError("Total clock does not equal preparation plus inference")
        if "ttc_seconds" in row:
            value = row["ttc_seconds"]
            if row["status"] == "ok" and (value is None or not np.isfinite(value)):
                raise ValueError("Successful timing lacks a finite TTC output")
            if row["status"] != "ok" and value is not None:
                raise ValueError("Unavailable output has a numeric TTC")


def stats(rows: list[dict[str, Any]], metric: str) -> list[float]:
    """Return mean, median, p95 and p99 from actual attempts."""
    values = np.array([r[metric] for r in rows], np.float64)
    if not len(values):
        raise ValueError("Empty reporting population")
    return [float(values.mean()), *np.quantile(values, [0.5, 0.95, 0.99]).tolist()]


def validate_protocol(protocol: dict[str, Any], cohort: str) -> None:
    """Reject pilot or differently configured cohorts in the publication build."""
    expected = (
        ["phase", "garl_full", "garl_ttc_only", "garl_event", "garl_visual"]
        if cohort == "evttc"
        else ["phase_native_raw", "garl_full_raw", "garl_ttc_only_raw"]
    )
    if protocol["models"] != expected or protocol["repeats"] != 3:
        raise ValueError("Publication requires the registered models and three passes")
    counts = Counter(q["sequence_id"] for q in protocol["queries"])
    if cohort == "evttc":
        if len(counts) != 10 or sum(counts.values()) != 434:
            raise ValueError("Publication requires all 434 EvTTC queries")
    elif len(counts) != 12 or set(counts.values()) != {10}:
        raise ValueError("Publication requires 120 eAP queries: ten per sequence")
    if cohort == "eap" and any(
        r["native_delta_t_s"] != 0.1 for r in protocol["garl_rows"].values()
    ):
        raise ValueError("Changed native Garl conversion")


def validate_profiles(
    profiles: dict[str, Any], protocol: dict[str, Any], trials: list[dict[str, Any]]
) -> None:
    """Require supported real profile queries and finite, scope-consistent accounting."""
    if set(profiles) != set(protocol["models"]):
        raise ValueError("Incomplete profile population")
    supported = {(r["model"], r["query"]) for r in trials if r["status"] == "ok"}
    for name, profile in profiles.items():
        if (name, profile["query"]) not in supported:
            raise ValueError("Profile query lacks a successful measured prediction")
        for key in ("total_parameters", "executed_parameters", "parameter_bytes"):
            if not isinstance(profile[key], int) or profile[key] <= 0:
                raise ValueError("Invalid profile parameter accounting")
        if profile["executed_parameters"] > profile["total_parameters"]:
            raise ValueError("Executed count exceeds stored count")
        for key in (
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "baseline_allocated_bytes",
            "counted_operator_flops",
        ):
            if not np.isfinite(profile[key]) or profile[key] < 0:
                raise ValueError("Invalid profile memory or arithmetic")


def validate_parity(rows: list[dict[str, Any]], full: str, bypass: str) -> None:
    """Independently check repeat stability and the decoder control's actual inputs."""
    index = {(r["model"], r["query"], r["repeat"]): r for r in rows}

    def compare(a: dict[str, Any], b: dict[str, Any]) -> None:
        for key in ("status", "reason", "prepared_input_sha256"):
            if a.get(key) != b.get(key):
                raise ValueError(f"Parity mismatch: {key}")
        for key in ("ttc_seconds", "heights"):
            av = a.get(key, a.get("diagnostics", {}).get(key))
            bv = b.get(key, b.get("diagnostics", {}).get(key))
            if av is None and bv is None:
                continue
            if av is None or bv is None:
                raise ValueError(f"Parity availability mismatch: {key}")
            if key == "heights" and any(isinstance(x, str) for x in [*av, *bv]):
                equal = av == bv
            else:
                equal = np.allclose(av, bv, rtol=1e-5, atol=1e-5, equal_nan=False)
            if not equal:
                raise ValueError(f"Numeric parity mismatch: {key}")

    for row in rows:
        compare(index[row["model"], row["query"], 0], row)
        if row["model"] == full:
            compare(row, index[bypass, row["query"], row["repeat"]])


def detailed(rows: list[dict[str, Any]], models: list[str]) -> dict[str, Any]:
    """Retain failed attempts, all-model complete cases and pairwise populations."""
    common = paired(rows, models)
    result: dict[str, Any] = {}
    for model in models:
        subsets = {
            "all_attempts": [r for r in rows if r["model"] == model],
            "successful": [r for r in rows if r["model"] == model and r["status"] == "ok"],
            "all_models_complete_case": [r for r in common if r["model"] == model],
        }
        result[model] = {}
        for name, subset in subsets.items():
            result[model][name] = {
                "attempts": len(subset),
                "queries": len({r["query"] for r in subset}),
                "status_reason_counts": dict(
                    Counter(f"{r['status']}:{r.get('reason', '')}" for r in subset)
                ),
                "timing_mean_median_p95_p99_ms": {
                    metric: stats(subset, metric)
                    for metric in ("total_ms", "preparation_ms", "inference_ms")
                    if subset and metric in subset[0]
                },
            }
    return result


def table(name: str, header: str, rows: list[list[str]], columns: str) -> None:
    """Write a deterministic LaTeX table with generated numeric cells."""
    content = (
        "\\begin{tabular}{"
        + columns
        + "}\n\\toprule\n"
        + header
        + " \\\\\n\\midrule\n"
        + "\n".join(" & ".join(row) + " \\\\" for row in rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (ROOT / "tables" / name).write_text(content, encoding="utf-8")


def paired(rows: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    """Keep only jointly successful query/repeat identities for matched cost comparisons."""
    common = set.intersection(
        *[
            {(r["query"], r["repeat"]) for r in rows if r["model"] == m and r["status"] == "ok"}
            for m in models
        ]
    )
    return [r for r in rows if (r["query"], r["repeat"]) in common and r["model"] in models]


def plot_latency(evttc: list[dict[str, Any]], eap: list[dict[str, Any]]) -> None:
    """Show complete measured distributions; do not trim desktop scheduling outliers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=True)
    for ax, rows, names, title in zip(
        axes,
        (evttc, eap),
        (
            ("phase", "garl_full", "garl_ttc_only"),
            ("phase_native_raw", "garl_full_raw", "garl_ttc_only_raw"),
        ),
        ("EvTTC: portable sensor adapter", "eAP: native RAW replay"),
        strict=True,
    ):
        for name, label, color in zip(
            names,
            ("PHASE-TTC", "Garl full release", "Garl decoder bypass"),
            ("#285fa4", "#c17828", "#2c8068"),
            strict=True,
        ):
            values = np.sort([r["total_ms"] for r in rows if r["model"] == name])
            ax.step(
                values,
                100 * np.arange(1, len(values) + 1) / len(values),
                where="post",
                label=label,
                color=color,
                linewidth=1.6,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Preparation + native inference (ms; log scale)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Completed calls at or below latency (%)")
    axes[0].set_ylim(0, 101)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/local_latency_cdf.pdf", metadata={"CreationDate": None})
    fig.savefig(ROOT / "figures/local_latency_cdf.png", dpi=180)
    plt.close(fig)


def write_readable_report(
    summary: dict[str, Any], evttc: list[dict[str, Any]], eap: list[dict[str, Any]]
) -> None:
    """Generate a compact evidence-linked report and exact English comparison copy."""
    lines = [
        "# Local inference comparison — 12 November 2026",
        "",
        "RTX 5070 Ti Laptop, Python 3.11.15, PyTorch 2.11.0+cu128, FP32, TF32 off, "
        "two PyTorch CPU threads. Frozen weights; no labels read and no new accuracy score.",
        "",
        "## Parameter inventory and EvTTC inference",
        "",
        "All 434 queries, five arms, three passes: 6,510 attempts. Timings below use "
        "all-five complete cases and include native API transfer, synchronization and host output. "
        "Sensor preparation is excluded from this inference clock.",
        "",
        "| System | Stored parameters | Observed executed parameters | "
        "Median inference (ms) | p95 (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("phase", "garl_full", "garl_ttc_only", "garl_event", "garl_visual"):
        profile = summary["profiles"][name]
        values = stats([r for r in evttc if r["model"] == name], "inference_ms")
        lines.append(
            f"| {LABELS[name]} | {profile['total_parameters']:,} | "
            f"{profile['executed_parameters']:,} | {values[1]:.2f} | {values[2]:.2f} |"
        )
    lines += [
        "",
        "Executed counts come from one supported real profile query. The Garl bypass "
        "retains all checkpoint weights; this is not an exported pruned-model comparison.",
        "",
        "## Native eAP RAW replay",
        "",
        "120 physical queries across twelve sequences, three arms, three passes: 1,080 attempts. "
        "The clock includes real sensor reading and preparation, native inference and host output. "
        "PHASE reuses causal observations; each arm has separate per-sequence reader caches.",
        "",
        "| System | Median total (ms) | p95 (ms) | p99 (ms) |",
        "|---|---:|---:|---:|",
    ]
    eap_values = {}
    for name in ("phase_native_raw", "garl_full_raw", "garl_ttc_only_raw"):
        values = stats([r for r in eap if r["model"] == name], "total_ms")
        eap_values[name] = values
        lines.append(f"| {LABELS[name]} | {values[1]:.2f} | {values[2]:.2f} | {values[3]:.2f} |")
    phase = summary["profiles"]["phase"]["total_parameters"]
    garl = summary["profiles"]["garl_ttc_only"]["executed_parameters"]
    lines += [
        "",
        "The eAP population is a fixed chronological prefix, not a random sample of the "
        "full benchmark. EvTTC shares a sensor reader and cyclic order does not balance all "
        "ordered pairs. OS file caches and desktop scheduling are uncontrolled. All latency "
        "outliers remain in the ledgers. These paths differ from the previous paced 50 Hz service.",
        "",
        "## English architecture paragraph",
        "",
        f"PHASE-TTC brings motion estimation, causal temporal refinement and phase-based fusion "
        f"together in a complete system with {phase:,} parameters. Its compact 1,955-parameter "
        f"fusion head combines motion estimates from events and RGB. Garl-TTC uses two ResNet-50 "
        f"backbones with late feature fusion; its decoder-bypassed path involved "
        f"{garl:,} parameters "
        f"in our local execution profile. On the same RTX 5070 Ti Laptop GPU, median native "
        f"RAW-to-host time across our 120-query eAP cohort was "
        f"{eap_values['phase_native_raw'][1]:.2f} ms "
        f"for PHASE-TTC, {eap_values['garl_full_raw'][1]:.2f} ms for the full Garl release and "
        f"{eap_values['garl_ttc_only_raw'][1]:.2f} ms with Garl's auxiliary decoders disabled. "
        "The measured parameter footprint and runtime are separate results.",
        "",
        "[Protocols, per-call evidence and reproduction commands]"
        "(../data/local_hardware_20261112/README.md). "
        "The paper includes preparation costs, memory, supported-operator arithmetic, coverage, "
        "pairwise populations, recording-level distributions and repeat-level means.",
        "",
    ]
    (ROOT / "revision/local_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Validate both experiments and generate capacity, latency and sequence tables."""
    summary, protocol, trials = load_run(DATA / "evttc")
    eap_summary, eap_protocol, eap_trials = load_run(DATA / "eap")
    validate_protocol(protocol, "evttc")
    validate_protocol(eap_protocol, "eap")
    validate_population(
        trials,
        protocol["models"],
        [f"{q['sequence_id']}:{q['row_index']}" for q in protocol["queries"]],
        protocol["repeats"],
    )
    validate_population(
        eap_trials,
        eap_protocol["models"],
        [q["sample_token"] for q in eap_protocol["queries"]],
        eap_protocol["repeats"],
    )
    validate_parity(trials, "garl_full", "garl_ttc_only")
    validate_parity(eap_trials, "garl_full_raw", "garl_ttc_only_raw")
    validate_profiles(summary["profiles"], protocol, trials)
    common = paired(trials, protocol["models"])
    components = summary["profiles"]["phase"]["components"]
    component_names = {
        "E_A5_MATCHED": "Event transport producer",
        "E_C2F_MATCHED": "Event coarse-to-fine producer",
        "R_A5": "RGB transport producer",
        "R_C2F": "RGB coarse-to-fine producer",
        "PAIR_E_MATCHED": "Event pair head",
        "PAIR_R": "RGB pair head",
        "E_CTX_MATCHED": "Event temporal refiner",
        "R_CTX": "RGB temporal refiner",
        "PHASE_ONLY_EVENT_S7_TRAIN40": "Residual phase fusion head",
    }
    if (
        set(components) != set(component_names)
        or sum(components.values()) != summary["profiles"]["phase"]["total_parameters"]
    ):
        raise ValueError("Original nine-component capacity identity changed")
    table(
        "local_phase_components.tex",
        "Component & Parameters & Share",
        [
            [
                label,
                f"{components[name]:,}",
                f"{100 * components[name] / sum(components.values()):.2f}\\%",
            ]
            for name, label in component_names.items()
        ],
        "lrr",
    )
    capacities, latency, stages = [], [], []
    for model in protocol["models"]:
        p = summary["profiles"][model]
        capacities.append(
            [
                LABELS[model],
                f"{p['total_parameters'] / 1e6:.3f}",
                f"{p['executed_parameters'] / 1e6:.3f}",
                f"{p['peak_allocated_bytes'] / 1024**2:.1f}",
                f"{p['counted_operator_flops'] / 1e9:.2f}",
            ]
        )
        all_rows = [r for r in trials if r["model"] == model]
        matched = [r for r in common if r["model"] == model]
        n = sum(r["status"] == "ok" for r in all_rows)
        latency.append(
            [
                LABELS[model],
                f"{n}/{len(all_rows)}",
                *[f"{x:.2f}" for x in stats(matched, "inference_ms")],
            ]
        )
        stages.append(
            [
                LABELS[model],
                f"{stats(matched, 'preparation_ms')[0]:.2f}",
                *[f"{x:.2f}" for x in stats(matched, "total_ms")[:3]],
            ]
        )
    table(
        "local_capacity.tex",
        "Model & Stored M & Executed M & Peak MiB & Counted GFLOPs",
        capacities,
        "lrrrr",
    )
    table(
        "local_inference.tex",
        "Model & Outputs/attempts & Mean & Median & p95 & p99",
        latency,
        "lrrrrr",
    )
    table(
        "local_pipeline.tex",
        "Model & Prep. mean & Total mean & Total median & Total p95",
        stages,
        "lrrrr",
    )
    raw = []
    eap_common = paired(eap_trials, eap_protocol["models"])
    plot_latency(common, eap_common)
    write_readable_report(summary, common, eap_common)
    for model in eap_protocol["models"]:
        rows = [r for r in eap_trials if r["model"] == model]
        raw.append(
            [
                LABELS[model],
                f"{sum(r['status'] == 'ok' for r in rows)}/{len(rows)}",
                *[
                    f"{x:.2f}"
                    for x in stats([r for r in eap_common if r["model"] == model], "total_ms")
                ],
            ]
        )
    table(
        "local_eap_raw.tex", "Model & Outputs/attempts & Mean & Median & p95 & p99", raw, "lrrrrr"
    )
    sequence_rows, repeat_rows = [], []
    for cohort, rows, models in [
        ("EvTTC", common, ["phase", "garl_full", "garl_ttc_only"]),
        ("eAP", paired(eap_trials, eap_protocol["models"]), eap_protocol["models"]),
    ]:
        for sequence in sorted({r["sequence"] for r in rows}):
            selected = [r for r in rows if r["sequence"] == sequence]
            n = len({r["query"] for r in selected})
            sequence_rows.append(
                [
                    cohort,
                    sequence,
                    str(n),
                    *[
                        f"{stats([r for r in selected if r['model'] == m], 'total_ms')[1]:.2f}"
                        for m in models
                    ],
                ]
            )
        for model in models:
            repeat_rows.append(
                [
                    cohort,
                    LABELS[model],
                    *[
                        f"{stats([r for r in rows if (r['model'], r['repeat']) == (model, i)], 'total_ms')[0]:.2f}"  # noqa: E501
                        for i in range(3)
                    ],
                ]
            )
    table(
        "local_sequences.tex",
        "Cohort & Recording & Queries & PHASE & Garl full & Garl bypass",
        sequence_rows,
        "llrrrr",
    )
    table("local_repeats.tex", "Cohort & Model & Pass 1 & Pass 2 & Pass 3", repeat_rows, "llrrr")
    pair_rows, pair_analysis, all_attempt_rows = [], {}, []
    for cohort, rows, models in (
        ("EvTTC", trials, protocol["models"]),
        ("eAP", eap_trials, eap_protocol["models"]),
    ):
        phase_name = models[0]
        for comparator in models[1:]:
            subset = paired(rows, [phase_name, comparator])
            pair_analysis[f"{cohort}:{comparator}"] = detailed(subset, [phase_name, comparator])
            pair_rows.append(
                [
                    cohort,
                    LABELS[comparator],
                    str(len(subset) // 2),
                    f"{stats([r for r in subset if r['model'] == phase_name], 'total_ms')[1]:.2f}",
                    f"{stats([r for r in subset if r['model'] == comparator], 'total_ms')[1]:.2f}",
                ]
            )
        for model in models:
            subset = [r for r in rows if r["model"] == model]
            all_attempt_rows.append(
                [
                    cohort,
                    LABELS[model],
                    str(len(subset)),
                    *[f"{x:.2f}" for x in stats(subset, "total_ms")],
                ]
            )
    table(
        "local_pairwise.tex",
        "Cohort & Comparator & Paired calls & PHASE median & Garl median",
        pair_rows,
        "llrrr",
    )
    table(
        "local_all_attempts.tex",
        "Cohort & Model & Calls & Mean & Median & p95 & p99",
        all_attempt_rows,
        "llrrrrr",
    )
    phase = summary["profiles"]["phase"]["total_parameters"]
    phase_executed = summary["profiles"]["phase"]["executed_parameters"]
    garl = summary["profiles"]["garl_ttc_only"]["executed_parameters"]
    macros = {
        "LocalPhaseParameters": f"{phase:,}",
        "LocalGarlTTCParameters": f"{garl:,}",
        "LocalPhaseExecutedParameters": f"{phase_executed:,}",
        "LocalExecutedParameterReduction": f"{100 * (1 - phase_executed / garl):.2f}\\%",
        "LocalMatchedQueries": str(len({r["query"] for r in common})),
        "LocalEvTTCAttempts": str(len(trials)),
        "LocalEAPAttempts": str(len(eap_trials)),
    }
    for prefix, rows, models in (
        ("Ev", common, protocol["models"][:3]),
        ("EAP", eap_common, eap_protocol["models"]),
    ):
        for label, model in zip(("Phase", "GarlFull", "GarlBypass"), models, strict=True):
            subset = [r for r in rows if r["model"] == model]
            for metric, key in (("total_ms", "Total"), ("inference_ms", "Infer")):
                if metric in subset[0]:
                    for suffix, value in zip(
                        ("Mean", "Median", "Pninetyfive", "Pninetynine"),
                        stats(subset, metric),
                        strict=True,
                    ):
                        macros[f"Local{prefix}{label}{key}{suffix}"] = f"{value:.2f}"
    (ROOT / "tables/local_macros.tex").write_text(
        "\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in macros.items())
        + "\n",
        encoding="utf-8",
    )
    result = {
        "evttc_trials": len(trials),
        "eap_trials": len(eap_trials),
        "matched_evttc_queries": macros["LocalMatchedQueries"],
        "phase_parameters": phase,
        "garl_ttc_executed_parameters": garl,
        "phase_executed_parameters": phase_executed,
        "fewer_executed_parameters_percent": 100 * (1 - phase_executed / garl),
        "evttc": detailed(trials, protocol["models"]),
        "eap": detailed(eap_trials, eap_protocol["models"]),
        "pairwise_complete_cases": pair_analysis,
        "parity_validation": (
            "Independent input, availability, reason, heights and TTC checks passed"
        ),
        "input_sha256": {
            str(p.relative_to(DATA)): digest(p) for p in DATA.rglob("*") if p.is_file()
        },
    }
    (ROOT / "revision/local_comparison_validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"input_sha256", "evttc", "eap", "pairwise_complete_cases"}
            }
        )
    )


if __name__ == "__main__":
    main()
