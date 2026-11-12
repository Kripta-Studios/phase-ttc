"""Regenerate production-service comparisons from every frozen paced replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/persistent_service_20261112"
MODELS = {"phase": "PHASE-TTC", "garl": "Garl TTC-only"}


def sha(path: Path) -> str:
    """Hash exact evidence bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: list[float]) -> dict[str, float] | None:
    """Use the historical linear quantile convention."""
    if not values:
        return None
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("Invalid latency values")
    return {
        "mean_ms": float(x.mean()),
        "p50_ms": float(np.median(x)),
        "p95_ms": float(np.percentile(x, 95)),
        "p99_ms": float(np.percentile(x, 99)),
        "max_ms": float(x.max()),
    }


def summarize_paced(
    paced: dict[str, Any], expected: int = 250, *, queue_capacity: int = 2
) -> dict[str, Any]:
    """Recompute response clocks and coverage without silently removing failures."""
    if (
        paced["offered_updates"] != expected
        or paced["rate_hz"] != 50
        or paced["queue_capacity"] != queue_capacity
    ):
        raise ValueError("Wrong offered population/rate/queue")
    arrivals = paced["arrival_receipts"]
    ids = [f"update-{i}" for i in range(expected)]
    if [r["update_id"] for r in arrivals] != ids:
        raise ValueError("Missing, duplicate or reordered arrival population")
    clocks = {r["update_id"]: r["scheduled_ns"] for r in arrivals}
    first = clocks[ids[0]]
    if any(clocks[name] != first + i * 20_000_000 for i, name in enumerate(ids)):
        raise ValueError("Scheduled clock spacing changed")
    seen = set()
    deadline, availability, service = [], [], []
    failed = 0
    for row in paced["object_records"]:
        name = row["update_id"]
        if name in seen or name not in clocks or row["object_id"] != "roi_00":
            raise ValueError("Unexpected or duplicate object record")
        seen.add(name)
        ticks = [
            row[k]
            for k in ("scheduled_ns", "observation_available_ns", "started_ns", "completed_ns")
        ]
        if ticks != sorted(ticks) or ticks[0] != clocks[name]:
            raise ValueError("Invalid response clock ordering")
        if row["success"] is not True:
            failed += 1
            continue
        deadline.append((ticks[3] - ticks[0]) / 1e6)
        availability.append((ticks[3] - ticks[1]) / 1e6)
        service.append((ticks[3] - ticks[2]) / 1e6)
    drops = sum(r["status"] == "DROPPED_QUEUE_FULL" for r in arrivals)
    errors = sum(r["status"] == "SOURCE_ERROR" for r in arrivals)
    enqueued = {r["update_id"] for r in arrivals if r["status"] == "ENQUEUED"}
    if seen != enqueued or len(deadline) + failed + drops + errors != expected:
        raise ValueError("Arrival/output accounting is inconsistent")
    complete = len(deadline) == expected and not drops and not errors and not failed
    stats = distribution(deadline)
    return {
        "offered": expected,
        "completed": len(deadline),
        "drops": drops,
        "source_errors": errors,
        "failed_responses": failed,
        "complete": complete,
        "deadline": stats,
        "availability": distribution(availability),
        "service": distribution(service),
        "primary_p95_ms": stats["p95_ms"] if complete and stats else None,
        "under_20ms": sum(x < 20 for x in deadline),
        "at_least_20ms": sum(x >= 20 for x in deadline),
        "response_ms": deadline,
        "cuda_memory": paced.get("service_cuda_memory"),
    }


def load_evidence(data: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require all six sealed streams and their exact oracle/input parity receipts."""
    inventory = json.loads((data / "inventory.json").read_text(encoding="utf-8"))
    for name, expected in inventory["files"].items():
        path = data / name
        if not path.resolve().is_relative_to(data.resolve()) or sha(path) != expected:
            raise ValueError(f"Changed persistent evidence: {name}")
    protocol = json.loads((data / "protocol.json").read_text(encoding="utf-8"))
    results = json.loads((data / "results.json").read_text(encoding="utf-8"))
    expected_names = ["phase_1", "garl_1", "garl_2", "phase_2", "phase_3", "garl_3"]
    if [r["name"] for r in results] != expected_names or protocol["repeats"] != 3:
        raise ValueError("Incomplete or selected repeated-service population")
    if protocol["schema"] == "persistent_service_comparison_graph_v2":
        integrity = json.loads((data / "end_integrity.json").read_text(encoding="utf-8"))
        if not integrity["passed"]:
            raise ValueError("Frozen service inputs changed during measurement")
    if sha(data / "causal_manifest.json") != protocol["manifest_sha256"]:
        raise ValueError("Causal manifest changed")
    manifest = json.loads((data / "causal_manifest.json").read_text(encoding="utf-8"))
    queries = [r["query_id"] for r in manifest["rows"]]
    if len(queries) != 250 or len(set(queries)) != 250:
        raise ValueError("Expected complete 250-query causal workload")
    analyses = {}
    for result in results:
        name = result["name"]
        summary_path = data / name / "summary.json"
        if sha(summary_path) != result["summary_sha256"]:
            raise ValueError("Changed sealed service summary")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        parity = summary["separate_parity"]
        checks = parity["updates"]
        if [r["update_id"] for r in checks] != [f"update-{i}" for i in range(250)]:
            raise ValueError("Incomplete parity population")
        stats = summarize_paced(summary["paced"])
        checked = {row["update_id"]: row["passed"] for row in checks}
        if any(
            not checked[row["update_id"]]
            for row in summary["paced"]["object_records"]
            if row["success"]
        ):
            raise ValueError("A delivered TTC fails native parity")
        stats["parity_passed"] = parity.get("passed") is True and all(r["passed"] for r in checks)
        if stats["complete"] and not stats["parity_passed"]:
            raise ValueError("Completed service fails native output/input parity")
        if summary["targets_read"] is not False or summary["optimizer_updates"] != 0:
            raise ValueError("Label/training boundary changed")
        if name.startswith("phase"):
            if not summary["originals"]["passed"] or not summary["rgb_cache_exercise"]["passed"]:
                raise ValueError("PHASE source preservation/cache gate failed")
            stats["cache_hits"] = summary["rgb_cache_exercise"]["hit_updates"]
            stats["cache_misses"] = summary["rgb_cache_exercise"]["miss_updates"]
        else:
            if not summary["sources_unchanged"]:
                raise ValueError("Garl source preservation failed")
            stats["cache_hits"] = summary["rgb_branch_hits"]
            stats["cache_misses"] = summary["rgb_branch_misses"]
            if protocol["schema"] == "persistent_service_comparison_graph_v2":
                graph = data / name / "graph_startup.json"
                if sha(graph) != result["graph_startup_sha256"]:
                    raise ValueError("Graph execution receipt changed")
                if stats["complete"] and not (stats["cache_hits"] and stats["cache_misses"]):
                    raise ValueError("Complete graph run did not exercise both cache paths")
            service_rows = json.loads(
                (data / name / "service_updates.json").read_text(encoding="utf-8")
            )
            if protocol["schema"] == "persistent_service_comparison_graph_v2":
                references = {row["update_id"]: row for row in checks}
                for actual in service_rows:
                    ref = references[actual["update_id"]]
                    contract_hash = hashlib.sha256(
                        json.dumps(actual["contract"], sort_keys=True).encode()
                    ).hexdigest()
                    if (
                        actual["sensor_sha256"] != ref["reference_sensor_sha256"]
                        or contract_hash != ref["reference_contract_sha256"]
                        or not np.allclose(
                            actual["heights"][0], ref["reference_heights"], rtol=1e-5, atol=1e-5
                        )
                        or not np.isclose(actual["ttc"], ref["reference_ttc"], rtol=1e-5, atol=1e-5)
                    ):
                        raise ValueError("Independent numerical Garl parity reconstruction failed")
            if [r["query_id"] for r in service_rows] != queries and stats["complete"]:
                raise ValueError("Garl queries differ from the shared workload")
            stats["sensor_age_us"] = [r["contract"]["sensor_age_us"] for r in service_rows]
            stats["unique_endpoint_pairs"] = len(
                {tuple(r["contract"]["frame_ids"]) for r in service_rows}
            )
        analyses[name] = stats
    return analyses, inventory


def table(name: str, columns: str, header: str, rows: list[str]) -> None:
    """Write a natural-size publication table."""
    text = [
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *[row + r" \\" for row in rows],
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (ROOT / "tables" / name).write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    analyses, inventory = load_evidence(DATA)
    parameters = json.loads(
        (ROOT / "revision/local_comparison_validation.json").read_text(encoding="utf-8")
    )
    groups = {}
    rows, detail = [], []
    for model, label in MODELS.items():
        runs = [analyses[f"{model}_{i}"] for i in (1, 2, 3)]
        admitted = all(r["complete"] and r["parity_passed"] for r in runs)
        p95 = [r["primary_p95_ms"] for r in runs] if admitted else []
        groups[model] = {
            "complete_all_passes": admitted,
            "median_p95_ms": float(np.median(p95)) if admitted else None,
            "min_p95_ms": min(p95) if admitted else None,
            "max_p95_ms": max(p95) if admitted else None,
            "completed": sum(r["completed"] for r in runs),
            "drops": sum(r["drops"] for r in runs),
            "failed_responses": sum(r["failed_responses"] for r in runs),
            "under_20ms": sum(r["under_20ms"] for r in runs),
        }
        center = f"{np.median(p95):.2f}" if admitted else "---"
        spread = f"{min(p95):.2f}--{max(p95):.2f}" if admitted else "---"
        rows.append(
            f"{label} & {groups[model]['completed']}/750 & {center} & {spread} & "
            f"{groups[model]['under_20ms']}/750"
        )
        for i, run in enumerate(runs, 1):
            timing = run["deadline"]
            quantiles = (
                " & ".join(f"{timing[k]:.2f}" for k in ("p50_ms", "p95_ms", "p99_ms"))
                if timing
                else "--- & --- & ---"
            )
            detail.append(
                f"{label} & {i} & {run['completed']}/250 & {run['drops']} & "
                f"{run['failed_responses']} & {quantiles}"
            )
    table(
        "persistent_primary.tex",
        "lrrrr",
        r"Persistent service & Outputs & Median p95 & p95 range & $<20$\,ms",
        rows,
    )
    table(
        "persistent_repeats.tex",
        "lrrrrrrr",
        r"Service & Pass & Outputs & Drops & Errors & p50 & p95 & p99",
        detail,
    )
    secondary = []
    for name, run in analyses.items():
        model, repeat = name.split("_")
        clocks = [run["availability"], run["service"]]
        values = " & ".join(f"{clock['p95_ms']:.2f}" if clock else "---" for clock in clocks)
        secondary.append(f"{MODELS[model]} & {repeat} & {values}")
    table(
        "persistent_secondary.tex",
        "lrrr",
        "Service & Pass & Available p95 & Service p95",
        secondary,
    )
    control, _ = load_evidence(DATA / "eager_control")
    control_rows = []
    for name, run in control.items():
        model, repeat = name.split("_")
        primary = f"{run['primary_p95_ms']:.2f}" if run["primary_p95_ms"] is not None else "---"
        control_rows.append(
            f"{MODELS[model]} & {repeat} & {run['completed']}/250 & {run['drops']} & "
            f"{run['failed_responses']} & {primary}"
        )
    table(
        "persistent_eager_control.tex",
        "lrrrrr",
        "Service & Pass & Outputs & Drops & Errors & Complete p95",
        control_rows,
    )
    memory = []
    for name, run in analyses.items():
        model, repeat = name.split("_")
        mem = run["cuda_memory"]
        memory.append(
            f"{MODELS[model]} & {repeat} & {mem['peak_allocated_bytes'] / 1024**2:.1f} & "
            f"{mem['peak_reserved_bytes'] / 1024**2:.1f} & "
            f"{run['cache_hits']} & {run['cache_misses']}"
        )
    table(
        "persistent_memory.tex",
        "lrrrrr",
        "Service & Pass & Allocated MiB & Reserved MiB & RGB hits & Misses",
        memory,
    )
    manifest = json.loads((DATA / "causal_manifest.json").read_text(encoding="utf-8"))
    source_age = [row["anchor_us"] - row["rgb_timestamps_us"][-1] for row in manifest["rows"]]
    cadence = [
        row["rgb_timestamps_us"][-1] - row["rgb_timestamps_us"][-2] for row in manifest["rows"]
    ]
    pairs = {tuple(row["rgb_frame_indices"][-2:]) for row in manifest["rows"]}
    age_text = (
        f"Across all 250 offered anchors, Garl's declared latest RGB endpoint is "
        f"{min(source_age) / 1000:.2f}--{max(source_age) / 1000:.2f}\\,ms old "
        f"(median {np.median(source_age) / 1000:.2f}\\,ms). The workload contains "
        f"{len(pairs)} distinct paired endpoints, with actual endpoint spacing "
        f"{min(cadence) / 1000:.2f}--{max(cadence) / 1000:.2f}\\,ms. "
        "These describe sensor support, not the response clock "
        "or a new observation at every arrival.\n"
    )
    (ROOT / "tables/persistent_input_age.tex").write_text(age_text, encoding="utf-8")
    macros = []
    for model, value in groups.items():
        number = (
            f"{value['median_p95_ms']:.2f}" if value["complete_all_passes"] else "not established"
        )
        macros.append(rf"\newcommand{{\Persistent{model.title()}Pninetyfive}}{{{number}}}")
    (ROOT / "tables/persistent_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    sentences = []
    for model, value in groups.items():
        if value["complete_all_passes"]:
            sentences.append(
                f"{MODELS[model]} completes all 750 scheduled outputs across three replays, "
                f"with median per-pass p95 {value['median_p95_ms']:.2f}\\,ms."
            )
        else:
            drop_word = "queue drop" if value["drops"] == 1 else "queue drops"
            sentences.append(
                f"{MODELS[model]} completes {value['completed']}/750 outputs, with "
                f"{value['drops']} {drop_word} and {value['failed_responses']} failed responses; "
                "a complete-service repeated p95 is not established."
            )
            for index in (1, 2, 3):
                run = analyses[f"{model}_{index}"]
                if run["complete"] and run["parity_passed"]:
                    sentences.append(
                        f"Its fully covered pass {index} has p95 {run['primary_p95_ms']:.2f}\\,ms; "
                        "this is an individual replay, not a three-pass capacity result."
                    )
    (ROOT / "tables/persistent_findings.tex").write_text(
        " ".join(sentences) + "\n", encoding="utf-8"
    )
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    for model, color in (("phase", "#216869"), ("garl", "#a34d37")):
        for i, style in zip((1, 2, 3), ("-", "--", ":"), strict=True):
            x = np.sort(analyses[f"{model}_{i}"]["response_ms"])
            ax.step(
                x,
                np.arange(1, len(x) + 1) / 250,
                where="post",
                color=color,
                linestyle=style,
                label=f"{MODELS[model]}, pass {i}",
            )
    ax.axvline(20, color="#333333", linewidth=1, alpha=0.6)
    ax.set(
        xlabel="Scheduled-arrival to host TTC (ms)",
        ylabel="Fraction of 250 arrivals completed",
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8, loc="lower right")
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"figures/persistent_latency_cdf.{ext}", dpi=180)
    plt.close(fig)
    validation = {
        "status": "PASS",
        "scope": "Evidence integrity and accounting; not a claim that service capacity passed",
        "complete_three_pass_models": [
            model for model, group in groups.items() if group["complete_all_passes"]
        ],
        "groups": groups,
        "passes": analyses,
        "eager_control": control,
        "evidence_inventory_sha256": sha(DATA / "inventory.json"),
        "verified_files": len(inventory["files"]),
    }
    (ROOT / "revision/persistent_comparison_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Persistent PHASE/Garl service comparison",
        "",
        "Same RTX 5070 Ti Laptop, FP32/TF32 off, two PyTorch CPU threads. CPLA-low, one object, "
        "250 causal arrivals at 50 Hz over five seconds; three independent "
        "replays per model, alternating order. "
        "RAW is preloaded; models receive only causally released sensor data. "
        "Initialization/disk preload are excluded.",
        "",
        "| Service | Median of three p95s (ms) | Range of p95s (ms) | Complete outputs |",
        "|---|---:|---:|---:|",
    ]
    for model, g in groups.items():
        center = f"{g['median_p95_ms']:.2f}" if g["complete_all_passes"] else "Not established"
        spread = f"{g['min_p95_ms']:.2f}–{g['max_p95_ms']:.2f}" if g["complete_all_passes"] else "—"
        report.append(f"| {MODELS[model]} | {center} | {spread} | {g['completed']}/750 |")
    report += [
        "",
        "Every graph-service replay is retained:",
        "",
        "| Service/pass | Outputs | Complete-replay p95 (ms) | Drops | Failed responses |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, run in analyses.items():
        p95 = (
            f"{run['primary_p95_ms']:.2f}"
            if run["primary_p95_ms"] is not None
            else "Not established"
        )
        report.append(
            f"| {name} | {run['completed']}/250 | {p95} | {run['drops']} | "
            f"{run['failed_responses']} |"
        )
    report += [
        "",
        "The earlier eager-service control is retained in full:",
        "",
        "| Service/pass | Outputs | Queue drops | Failed responses |",
        "|---|---:|---:|---:|",
    ]
    for name, run in control.items():
        report.append(
            f"| {name} | {run['completed']}/250 | {run['drops']} | {run['failed_responses']} |"
        )
    report += [
        "",
        "Complete runs must pass every native parity check. Incomplete runs retain "
        "their drops and failed outputs; no complete-service p95 is assigned to a model "
        "unless all three runs complete. The primary clock includes "
        "acquisition/release, scheduling, "
        "queueing, preparation, inference and host delivery. These are repeats of"
        " the same 250 anchors, "
        "not 750 independent examples. Neither this experiment nor the historical"
        " 29.14 ms p95 proves "
        "a strict 20 ms deadline, physical-camera service, multi-object capacity "
        "or new accuracy ranking.",
        "",
        "For Garl, a missing RAW packet invalidates contiguous causal history and causes the "
        "service to refuse subsequent requests until a fresh replay. These remain capacity "
        "failures, but are correlated refusals rather than emitted inaccurate TTC estimates.",
        "",
        "Reference verification executes between measured streams, with different GPU workloads "
        "for each model and no enforced cooldown. Before/after telemetry documents but does not "
        "control power, thermal or cache carryover. These observations do not establish a causal "
        "speed ranking, or identify the cause of a capacity failure.",
        "",
        "Garl uses the two declared causal RGB endpoints from the shared "
        "manifest, exact cached native "
        "representations and paired RGB backbone features. Its event backbone and"
        " TTC heads run on every "
        "completed update, using separate native FP32 CUDA graphs. "
        "Auxiliary foreground decoders are disabled, with every delivered output checked against "
        "untouched native inference. Missing outputs remain failed checks. "
        "Input age and repeated sensor endpoints "
        "remain in the evidence.",
        "",
        "## English architecture paragraph",
        "",
        "PHASE-TTC builds its predictions from motion, causal temporal refinement and "
        "phase-based event–RGB fusion. It brings these stages together in a complete "
        f"{parameters['phase_parameters']:,}-parameter system, compared with "
        f"{parameters['garl_ttc_executed_parameters']:,} parameters in Garl-TTC's decoder-free "
        f"TTC path ({parameters['fewer_executed_parameters_percent']:.2f}% fewer). "
        "While Garl uses two ResNet-50 backbones and a height-ratio readout, PHASE uses "
        "compact motion producers, temporal refiners and an RGB-anchored residual fusion head. "
        "Its persistent service "
        "updates event state "
        "incrementally and reuses exact RGB features and transport summaries, preserving "
        "native predictions while avoiding repeated computation.",
        "",
        "The earlier query-oriented API experiment is historical evidence only "
        "and is excluded from "
        "the article's production-service latency comparison.",
    ]
    if all(g["complete_all_passes"] for g in groups.values()):
        report += [
            "",
            "For the measured single-object workload, all three replays completed without loss: "
            f"PHASE-TTC achieved {groups['phase']['median_p95_ms']:.2f} ms median per-run p95 "
            f"response, versus {groups['garl']['median_p95_ms']:.2f} ms for persistent TTC-only "
            "Garl, including causal RAW release, queueing, preparation and host delivery.",
        ]
    (ROOT / "revision/persistent_comparison_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "groups": groups}))


if __name__ == "__main__":
    main()
