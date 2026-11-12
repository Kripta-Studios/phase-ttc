"""Regenerate the expanded persistent inference chapter from every sealed run."""

from __future__ import annotations

import csv
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from build_persistent_comparison import distribution, sha, summarize_paced

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/extended_service_20261112"
NAMES = {"phase": "PHASE", "garl": "Garl"}


def read(path: Path) -> Any:  # noqa: ANN401
    """Load archived JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: float | None) -> str:
    """Render missing full-coverage measurements explicitly."""
    return "---" if value is None else f"{value:.2f}"


def table(name: str, columns: str, header: str, rows: list[str]) -> None:
    """Write deterministic booktabs tables."""
    text = "\\begin{tabular}{" + columns + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n"
    text += "\n".join(row + r" \\" for row in rows) + "\n\\bottomrule\n\\end{tabular}\n"
    (ROOT / "tables" / f"extended_{name}.tex").write_text(text, encoding="utf-8")


def validate_oracle_jobs(data: Path, protocol: dict[str, Any]) -> str:
    """Bind all native-reference jobs to the archived executed source identities."""
    jobs = read(data / "oracle_jobs.json")
    controller_sha = sha(data / "executed_sources/scripts__run_extended_oracles.py.txt")
    launcher_sha = sha(data / "executed_sources/scripts__reference_extended_service.py.txt")
    if (
        jobs["controller_sha256"] != controller_sha
        or jobs["frozen_launcher_sha256"] != launcher_sha
    ):
        raise ValueError("Native-reference execution source identity differs")
    expected = [
        (f"reference_s{w['id']}_{model}", str(w["id"]), model)
        for w in protocol["workloads"]
        for model in NAMES
    ]
    if [j["name"] for j in jobs["jobs"]] != [e[0] for e in expected]:
        raise ValueError("Incomplete or reordered native-reference job population")
    for job, (_, workload, model) in zip(jobs["jobs"], expected, strict=True):
        command = job["command"]
        if (
            job["returncode"] != 0
            or not isinstance(job["all_checks_passed"], bool)
            or len(command) != 8
            or command[2] != "--campaign"
            or command[4:] != ["--model", model, "--workload", workload]
            or command[1].replace("\\", "/").rsplit("/", 1)[-1] != "reference_extended_service.py"
        ):
            raise ValueError("Native-reference job failed or used a different command")
    return launcher_sha


def qualify_delivery(stats: dict[str, Any], parity: dict[str, Any]) -> None:
    """Keep delivery coverage distinct from strict native-parity admission."""
    stats["parity"] = parity["all_delivered_passed"] and parity["all_reference_valid"]
    stats["admitted"] = stats["complete"] and stats["parity"]
    stats["numerical_failures"] = sum(r["delivered"] and not r["passed"] for r in parity["updates"])
    stats["validated_outputs"] = sum(r["delivered"] and r["passed"] for r in parity["updates"])
    if not stats["admitted"]:
        stats["primary_p95_ms"] = None


def phase_parity_diagnostics(data: Path, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count repeated comparisons and unique failed anchors without relaxing tolerance."""
    diagnostics = []
    for scene in range(6):
        for component, label in (
            ("PHASE_ONLY_TRAIN40_S7", "Fused"),
            ("E_CTX_TRAIN40", "Event"),
            ("R_CTX_TRAIN40", "RGB"),
        ):
            checks, failed, anchors, repetitions, absolute, relative = 0, 0, set(), set(), [], []
            for run in runs:
                if run["model"] != "phase" or run["workload"] != scene:
                    continue
                parity = read(data / run["name"] / "deferred_parity.json")
                for row in parity["updates"]:
                    if not row["delivered"]:
                        continue
                    check = row["parity"]["per_query"][0]["components"][component]
                    checks += 1
                    if not check["passed"]:
                        failed += 1
                        anchors.add(row["update_id"])
                        repetitions.add(run["name"])
                    value = check.get("abs_error")
                    if value is not None and np.isfinite(value):
                        absolute.append(float(value))
                    value = row["parity"]["components"][component].get("max_relative_error")
                    if value is not None and np.isfinite(value):
                        relative.append(float(value))
            diagnostics.append(
                {
                    "scene": scene,
                    "component": component,
                    "label": label,
                    "checks": checks,
                    "failed_comparisons": failed,
                    "failed_anchors": sorted(anchors),
                    "affected_runs": sorted(repetitions),
                    "max_abs_error_s": max(absolute, default=None),
                    "max_relative_error": max(relative, default=None),
                }
            )
    return diagnostics


def analyze(data: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the sealed population, clock accounting, parity and source preservation."""
    inventory = read(data / "inventory.json")["files"]
    for name, digest in inventory.items():
        path = data / name
        if not path.resolve().is_relative_to(data.resolve()) or sha(path) != digest:
            raise ValueError(f"Changed extended evidence: {name}")
    publication = read(data / "publication_inventory.json")
    if sha(data / "inventory.json") != publication["campaign_inventory_sha256"]:
        raise ValueError("Campaign inventory changed")
    for name, digest in publication["files"].items():
        path = data / name
        if not path.resolve().is_relative_to(data.resolve()) or sha(path) != digest:
            raise ValueError(f"Changed publication evidence: {name}")
    archive_execution = read(data / "archive_execution.json")
    executed_sha = sha(data / "executed_sources/scripts__archive_extended_comparison.py.txt")
    if archive_execution["returncode"] != 0 or any(
        archive_execution[key] != executed_sha
        for key in ("expected_sha256", "executed_sha256_before", "executed_sha256_after")
    ):
        raise ValueError("Archiver execution identity differs from the frozen source")
    if (
        sha(data / "executed_sources/scripts__seal_extended_publication.py.txt")
        != archive_execution["sealer_sha256"]
    ):
        raise ValueError("Supplemental evidence sealer source changed")
    protocol, results = read(data / "protocol.json"), read(data / "results.json")
    oracle_sha = validate_oracle_jobs(data, protocol)

    # Load hash-verified frozen sources so the source ZIP also works without the repository.
    def archived_module(name: str, path: Path) -> Any:  # noqa: ANN401
        loader = importlib.machinery.SourceFileLoader(name, str(path))
        spec = importlib.util.spec_from_loader(name, loader)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module

    verifier = archived_module(
        "frozen_extended", data / "executed_sources/operational__local_comparison__extended.py.txt"
    )
    measurement = archived_module(
        "frozen_measurement",
        data / "historical_sources/operational__train40_runtime__measurement.py.txt",
    )
    verifier.validate_run_population(protocol, results)
    if protocol["pilot"] or len(protocol["workloads"]) != 6 or len(results) != 72:
        raise ValueError("Expected the entire prospective six-scene, 72-pass campaign")
    if [r["name"] for r in results] != [c["name"] for c in protocol["commands"]]:
        raise ValueError("Missing or reordered repeated population")
    for filename in ("end_integrity.json", "post_reference_integrity.json"):
        if not read(data / filename)["passed"]:
            raise ValueError("Frozen campaign sources changed")
    analyses = []
    for command, result in zip(protocol["commands"], results, strict=True):
        folder = data / command["name"]
        for filename, digest in result["files"].items():
            if sha(folder / filename) != digest:
                raise ValueError("A measured receipt changed after timing")
        workload = protocol["workloads"][command["workload"]]
        manifest_path = data / "cohorts" / f"manifest_{workload['id']}.json"
        if sha(manifest_path) != workload["manifest_sha256"]:
            raise ValueError("Causal cohort changed")
        manifest = read(manifest_path)
        if manifest["excluded"] or len(manifest["rows"]) != workload["updates"]:
            raise ValueError("Causal interval is incomplete")
        actual_path = folder / "service_updates.json"
        actual = read(actual_path) if actual_path.exists() else []
        parity = read(folder / "deferred_parity.json")
        reference_path = data / f"reference_s{workload['id']}_{command['model']}.json"
        if sha(reference_path) != parity["reference_sha256"]:
            raise ValueError("Native reference changed")
        if parity["delivered"] != len(actual) or parity["expected"] != workload["updates"]:
            raise ValueError("Native parity population differs")
        if not parity["all_reference_valid"]:
            raise ValueError("Independent native reference did not succeed at every anchor")
        ref = read(reference_path)
        if ref["oracle_source_sha256"] != oracle_sha:
            raise ValueError("Native reference was generated by a different source")
        if (
            ref["manifest_sha256"] != workload["manifest_sha256"]
            or ref["model"] != command["model"]
        ):
            raise ValueError("Native reference model/cohort differs")
        query_ids = [r["query_id"] for r in manifest["rows"]]
        if [r["query_id"] for r in ref["rows"]] != query_ids:
            raise ValueError("Native reference query identities differ")
        if any(r["query_id"] != query_ids[int(r["update_id"].split("-")[1])] for r in actual):
            raise ValueError("Measured query identity differs from the causal schedule")
        if [r["update_id"] for r in ref["rows"]] != [
            f"update-{i}" for i in range(workload["updates"])
        ]:
            raise ValueError("Incomplete independent reference population")
        # Verify comparison numerically from references and actual outputs on any machine.
        with patch.dict(sys.modules, {"operational.train40_runtime.measurement": measurement}):
            recomputed = verifier.compare_reference(command["model"], ref["rows"], actual)
        if any(recomputed[k] != parity[k] for k in recomputed):
            raise ValueError("Deferred parity cannot be reproduced")
        paced_path = folder / "paced.json"
        if not paced_path.exists():
            if actual or parity["delivered"]:
                raise ValueError("Outputs exist without measured arrival records")
            failure = {
                **command,
                "startup_failure": True,
                "offered": workload["updates"],
                "completed": 0,
                "complete": False,
                "primary_p95_ms": None,
            }
            qualify_delivery(failure, parity)
            analyses.append(failure)
            continue
        paced = read(paced_path)
        stats = summarize_paced(paced, workload["updates"], queue_capacity=command["queue"])
        qualify_delivery(stats, parity)
        if len(actual) != stats["completed"]:
            raise ValueError("Prediction ledger differs from measured successful outputs")
        summary = read(folder / "summary.json")
        binding_name = "binding.json" if command["model"] == "phase" else "protocol.json"
        run_binding = read(folder / binding_name)
        if run_binding["manifest_sha256"] != workload["manifest_sha256"]:
            raise ValueError("Timed service used a different manifest")
        if run_binding["precision"] != "FP32_TF32_OFF":
            raise ValueError("Timed precision differs from the declared comparison")
        intact = (
            summary["originals"]["passed"]
            if command["model"] == "phase"
            else summary["sources_unchanged"]
        )
        if not intact:
            raise ValueError("Model/source preservation failed")
        rest = read(folder / "rest.json")
        if rest["policy_seconds"] != 15 or rest["elapsed_seconds"] < 15:
            raise ValueError("Declared pre-stream rest was not applied")
        gpu_rows = list(csv.DictReader(rest["after"]["gpu"]["stdout"].splitlines()))
        gpu_state = {}
        if rest["after"]["gpu"]["returncode"] == 0 and len(gpu_rows) == 1:
            for key, value in gpu_rows[0].items():
                for prefix, label in (
                    ("temperature.gpu", "temperature_c"),
                    ("clocks.current.sm", "sm_mhz"),
                    ("power.draw", "power_w"),
                ):
                    if key.strip().startswith(prefix):
                        try:
                            gpu_state[label] = float(value.strip().split()[0])
                        except ValueError:
                            gpu_state[label] = None
        valid = [r for r in paced["object_records"] if r["success"]]
        first = paced["arrival_receipts"][0]["scheduled_ns"]
        last = max((r["completed_ns"] for r in valid), default=first)
        queue_wait = [(r["started_ns"] - r["observation_available_ns"]) / 1e6 for r in valid]
        release = [
            (r["observation_available_ns"] - r["scheduled_ns"]) / 1e6
            for r in paced["arrival_receipts"]
            if "observation_available_ns" in r
        ]
        deadline = np.asarray(stats["response_ms"])
        latter_half = [
            (r["completed_ns"] - r["scheduled_ns"]) / 1e6
            for r in valid
            if int(r["update_id"].split("-")[1]) >= stats["offered"] // 2
        ]
        completed_times = sorted(r["completed_ns"] for r in valid)
        pending = [
            sum(
                r["scheduled_ns"] < a["scheduled_ns"] < r["completed_ns"]
                for r in paced["object_records"]
            )
            for a in paced["arrival_receipts"]
        ]
        startup = read(folder / "startup.json")
        source_stats = paced["source_lifecycle"]["finalize"]["result"]
        hits = (
            summary["rgb_cache_exercise"]["hit_updates"]
            if command["model"] == "phase"
            else summary["rgb_branch_hits"]
        )
        misses = (
            summary["rgb_cache_exercise"]["miss_updates"]
            if command["model"] == "phase"
            else summary["rgb_branch_misses"]
        )
        stats.update(
            command,
            startup_failure=False,
            reference_valid=parity["all_reference_valid"],
            queue_wait=distribution(queue_wait),
            release=distribution(release),
            completion_hz=len(valid) / ((last - first) / 1e9) if last > first else 0,
            drain_ms=max(0.0, (last - first - stats["offered"] * 20_000_000) / 1e6),
            max_pending_at_arrival=max(pending, default=0),
            over_50ms=int((deadline > 50).sum()),
            over_100ms=int((deadline > 100).sum()),
            longest_output_gap_ms=max(np.diff(completed_times) / 1e6, default=0),
            source_archive_bytes=startup["raw_preload"]["archive_bytes"],
            source_envelope_bytes=startup["raw_preload"]["source_total_upper_bound_bytes"],
            startup=startup,
            history_failure=summary.get("history_failure"),
            source_stats=source_stats,
            cache_hits=hits,
            cache_misses=misses,
            gpu_state=gpu_state,
            first_response_ms=float(deadline[0]) if len(deadline) else None,
            latter_half=distribution(latter_half) if stats["admitted"] else None,
        )
        analyses.append(stats)
    for job in read(data / "oracle_jobs.json")["jobs"]:
        matching = [
            r for r in analyses if job["name"] == f"reference_s{r['workload']}_{r['model']}"
        ]
        if job["all_checks_passed"] != all(r["parity"] for r in matching):
            raise ValueError("Native-reference job outcome differs from recomputed parity")
    return protocol, analyses


def main() -> None:
    """Generate complete tables, per-repeat details, figures and validation JSON."""
    protocol, runs = analyze(DATA)
    scenes = protocol["workloads"]
    parity_details = phase_parity_diagnostics(DATA, runs)
    parity_rows = []
    for row in parity_details:
        absolute = row["max_abs_error_s"]
        relative = row["max_relative_error"]
        errors = (
            f"{absolute:.2e} & {relative:.2e}"
            if absolute is not None and relative is not None
            else "--- & ---"
        )
        parity_rows.append(
            f"S{row['scene'] + 1} & {row['label']} & {row['checks']} & "
            f"{row['failed_comparisons']} & {len(row['failed_anchors'])} & "
            f"{len(row['affected_runs'])} & {errors}"
        )
    table(
        "parity",
        "llrrrrrr",
        "Scene & Output & Checks & Failures & Anchors & Runs & Max abs. s & Max rel.",
        parity_rows,
    )
    parity_text = [
        "The frozen PHASE numerical contract uses relative and absolute tolerances "
        "of $10^{-5}$ for each of its three TTC outputs. "
        "These differences measure agreement with native inference, not error against TTC labels."
    ]
    for label in ("Fused", "Event", "RGB"):
        group = [r for r in parity_details if r["label"] == label]
        parity_text.append(
            f"For the {label.lower()} output, "
            f"{sum(r['failed_comparisons'] for r in group)} of "
            f"{sum(r['checks'] for r in group):,} delivered comparisons exceed the contract, "
            f"covering {sum(len(r['failed_anchors']) for r in group)} distinct scene/anchor pairs."
        )
    (ROOT / "tables/extended_parity_interpretation.tex").write_text(
        "\n\n".join(parity_text) + "\n", encoding="utf-8"
    )
    table(
        "cohorts",
        "llrrr",
        "ID & Sequence & Start (s) & Updates & Duration (s)",
        [
            f"S{w['id'] + 1} & {w['sequence']} & {w['bounds_us'][0] / 1e6:.2f} & {w['updates']} & "
            f"{w['updates'] / 50:.2f}"
            for w in scenes
        ],
    )
    for mode in ("bounded", "buffered"):
        rows = []
        for workload in scenes:
            for model in NAMES:
                group = [
                    r
                    for r in runs
                    if r["workload"] == workload["id"] and r["model"] == model and r["mode"] == mode
                ]
                complete = [r for r in group if r["complete"]]
                admitted = [r for r in group if r["admitted"]]
                values = [r["primary_p95_ms"] for r in group]
                p95 = number(float(np.median(values))) if len(admitted) == len(group) else "---"
                bounds = (
                    f"{min(values):.2f}--{max(values):.2f}"
                    if len(admitted) == len(group)
                    else "---"
                )
                delivered, offered = (
                    sum(r["completed"] for r in group),
                    sum(r["offered"] for r in group),
                )
                rows.append(
                    f"S{workload['id'] + 1} & {NAMES[model]} & {len(complete)}/{len(group)} & "
                    f"{len(admitted)}/{len(group)} & {delivered}/{offered} & {p95} & {bounds}"
                )
        table(
            mode,
            "llrrrrr",
            "Scene & Model & Delivered & Admitted & Outputs & p95 & p95 range",
            rows,
        )
    joint = []
    for w in scenes:
        for model in NAMES:
            cells = [f"S{w['id'] + 1}", NAMES[model]]
            for mode in ("bounded", "buffered"):
                group = [
                    r
                    for r in runs
                    if r["workload"] == w["id"] and r["model"] == model and r["mode"] == mode
                ]
                full = sum(r["complete"] for r in group)
                admitted = sum(r["admitted"] for r in group)
                p95 = (
                    float(np.median([r["primary_p95_ms"] for r in group]))
                    if admitted == len(group)
                    else None
                )
                cells += [f"{full}/{admitted}/{len(group)}", number(p95)]
            joint.append(" & ".join(cells))
    table("joint", "llrrrr", "Scene & Model & Q2 D/A/N & Q2 p95 & Q64 D/A/N & Q64 p95", joint)
    details = []
    diagnostics = []
    for r in runs:
        label = f"{'Q2' if r['mode'] == 'bounded' else 'Q64'} S{r['workload'] + 1} R{r['repeat']}"
        if r["startup_failure"]:
            details.append(
                f"{label} & {NAMES[r['model']]} & 0/{r['offered']} & "
                f"\\multicolumn{{6}}{{c}}{{Startup failure}}"
            )
            continue
        d = r["deadline"] or {}
        details.append(
            f"{label} & {NAMES[r['model']]} & {r['completed']}/{r['offered']} & "
            f"{number(d.get('p50_ms'))} & {number(d.get('p95_ms'))} & {number(d.get('p99_ms'))} & "
            f"{r['drops']} & {r['failed_responses']} & {r['numerical_failures']}"
        )
        diagnostics.append(
            f"{label} & {NAMES[r['model']]} & {number((r['service'] or {}).get('p95_ms'))} & "
            f"{number((r['queue_wait'] or {}).get('p95_ms'))} & "
            f"{r['under_20ms']}/{r['offered']} & {r['completion_hz']:.1f} & "
            f"{r['drain_ms']:.1f}"
        )
    for name, heading, columns, rows in [
        (
            "repeats",
            "Pass & Model & Outputs & p50 & p95 & p99 & Drops & Failed & Parity",
            "llrrrrrrr",
            details,
        ),
        (
            "clocks",
            "Pass & Model & Service p95 & Queue p95 & $<20$ ms & Outputs/s & Drain ms",
            "llrrrrr",
            diagnostics,
        ),
    ]:
        text = (
            "\\begin{longtable}{"
            + columns
            + "}\n\\toprule\n"
            + heading
            + " \\\\\n\\midrule\n\\endhead\n"
        )
        text += "\n".join(x + r" \\" for x in rows) + "\n\\bottomrule\n\\end{longtable}\n"
        (ROOT / "tables" / f"extended_{name}.tex").write_text(text, encoding="utf-8")
    summary_rows = []
    for mode in ("bounded", "buffered"):
        for model in NAMES:
            group = [r for r in runs if r["mode"] == mode and r["model"] == model]
            ready = [r for r in group if not r["startup_failure"]]
            memory = [r["cuda_memory"]["peak_allocated_bytes"] / 1024**2 for r in ready]
            summary_rows.append(
                f"{mode} & {NAMES[model]} & {sum(r['complete'] for r in group)}/{len(group)} & "
                f"{sum(r['admitted'] for r in group)}/{len(group)} & "
                f"{sum(r['completed'] for r in group)}/{sum(r['offered'] for r in group)} & "
                f"{min(memory):.0f}--{max(memory):.0f}"
            )
    table(
        "overview",
        "llrrrr",
        "Queue policy & Model & Delivered & Admitted & Outputs & CUDA peak MiB",
        summary_rows,
    )
    input_rows = []
    age_rows = []
    for w in scenes:
        run = next(
            r
            for r in runs
            if r["workload"] == w["id"]
            and r["model"] == "phase"
            and r["mode"] == "bounded"
            and r["repeat"] == 1
        )
        if run["startup_failure"]:
            continue
        source = run["source_stats"]
        # The source's owned event ABI is int32 x/y, int64 t, int8 p: 17 bytes/event.
        count, remainder = divmod(source["event_packet_copy_bytes"], 17)
        if remainder:
            raise ValueError("Unexpected copied-event ABI")
        reference = read(DATA / f"reference_s{w['id']}_garl.json")["rows"]
        valid = [r for r in reference if "error" not in r]
        ages = (
            [r["contract"]["sensor_age_us"] / 1000 for r in valid]
            if valid and "sensor_age_us" in valid[0]["contract"]
            else []
        )
        if ages:
            age_rows.append(
                f"S{w['id'] + 1} & {len(valid)}/{w['updates']} & "
                f"{min(ages):.2f}--{max(ages):.2f} & {np.median(ages):.2f} & "
                f"{len({tuple(r['contract']['frame_ids']) for r in valid})}"
            )
        else:
            age_rows.append(f"S{w['id'] + 1} & 0/{w['updates']} & --- & --- & ---")
        input_rows.append(
            f"S{w['id'] + 1} & {source['acquired_updates']}/{w['updates']} & "
            f"{count / 1e6:.2f} & {count / (w['updates'] / 50) / 1e6:.2f} & "
            f"{run['source_archive_bytes'] / 1024**2:.0f}"
        )
    table(
        "input_load",
        "lrrrr",
        "Scene & RAW packets & Events (M) & M events/s & Archive MiB",
        input_rows,
    )
    table(
        "input_age",
        "lrrrr",
        "Scene & Native references & Age range ms & Median ms & RGB pairs",
        age_rows,
    )
    execution_rows = []
    host_rows = []
    interpretation = []
    for mode in ("bounded", "buffered"):
        mode_text = "two-packet" if mode == "bounded" else "64-packet"
        for model in NAMES:
            group = [r for r in runs if r["mode"] == mode and r["model"] == model]
            ready = [r for r in group if not r["startup_failure"]]
            full = sum(r["complete"] for r in group)
            delivered, offered = (
                sum(r["completed"] for r in group),
                sum(r["offered"] for r in group),
            )
            under = sum(r["under_20ms"] for r in ready)
            interpretation.append(
                f"With the {mode_text} queue, {NAMES[model]} completed "
                f"{full}/{len(group)} full replays "
                f"and delivered {delivered:,}/{offered:,} outputs "
                f"({100 * delivered / offered:.1f}\\%). "
                f"Of all offered updates, {under:,} ({100 * under / offered:.1f}\\%) "
                "completed below 20\\,ms. "
                f"Strict native parity admitted {sum(r['admitted'] for r in group)}/{len(group)} "
                f"full replays; {sum(r['numerical_failures'] for r in group)} delivered outputs "
                "failed at least one required numerical check."
            )
            preloads = [r["startup"]["raw_preload"]["preload_seconds"] for r in ready]
            init = [
                (
                    r["startup"].get("model_and_service_seconds", 0)
                    + r["startup"].get("synthetic_shape_priming_seconds", 0)
                )
                if model == "phase"
                else r["startup"]["model_and_synthetic_warmup_seconds"]
                for r in ready
            ]
            hits, misses = (
                sum(r["cache_hits"] for r in ready),
                sum(r["cache_misses"] for r in ready),
            )
            reserved = [r["cuda_memory"]["peak_reserved_bytes"] / 1024**2 for r in ready]
            execution_rows.append(
                f"{mode} & {NAMES[model]} & {np.median(preloads):.2f} & {np.median(init):.2f} & "
                f"{hits}/{hits + misses} & {min(reserved):.0f}--{max(reserved):.0f}"
            )
            ranges = []
            for field in ("temperature_c", "sm_mhz", "power_w"):
                values = [r["gpu_state"].get(field) for r in ready]
                values = [v for v in values if v is not None]
                ranges.append(f"{min(values):.1f}--{max(values):.1f}" if values else "---")
            host_rows.append(f"{mode} & {NAMES[model]} & " + " & ".join(ranges))
    table(
        "execution",
        "llrrrr",
        "Policy & Model & Preload s & Model/prime s & RGB hits & Reserved MiB",
        execution_rows,
    )
    table(
        "host", "llrrr", "Policy & Model & GPU temperature $^\\circ$C & SM MHz & Power W", host_rows
    )
    tail_rows = []
    for w in scenes:
        for model in NAMES:
            group = [
                r
                for r in runs
                if r["mode"] == "buffered" and r["model"] == model and r["workload"] == w["id"]
            ]
            full = sum(r["complete"] for r in group)
            admitted = sum(r["admitted"] for r in group)
            if admitted == len(group):
                first = float(np.median([r["first_response_ms"] for r in group]))
                entire = float(np.median([r["primary_p95_ms"] for r in group]))
                late = float(np.median([r["latter_half"]["p95_ms"] for r in group]))
                values = f"{first:.2f} & {entire:.2f} & {late:.2f}"
            else:
                values = "--- & --- & ---"
            tail_rows.append(
                f"S{w['id'] + 1} & {NAMES[model]} & {full}/{admitted}/{len(group)} & {values}"
            )
    table(
        "initial_late",
        "llrrrr",
        "Scene & Model & D/A/N & First response & Full p95 & Latter-half p95",
        tail_rows,
    )
    (ROOT / "tables/extended_interpretation.tex").write_text(
        "\n\n".join(interpretation) + "\n", encoding="utf-8"
    )
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.3), constrained_layout=True)
    for w, ax in zip(scenes, axes.flat, strict=True):
        for model, color in [("phase", "#137c66"), ("garl", "#bc5c38")]:
            for r in runs:
                if (
                    r["mode"] == "bounded"
                    and r["model"] == model
                    and r["workload"] == w["id"]
                    and not r["startup_failure"]
                ):
                    x = np.sort(r["response_ms"])
                    ax.step(
                        x,
                        np.arange(1, len(x) + 1) / r["offered"],
                        color=color,
                        alpha=0.65,
                        label=NAMES[model] if r["repeat"] == 1 else None,
                    )
        ax.axvline(20, color="gray", ls=":")
        ax.set(
            title=f"S{w['id'] + 1}: {w['sequence']}",
            xlabel="Response (ms)",
            ylabel="Fraction of offered updates",
            ylim=(0, 1.03),
        )
        ax.legend(fontsize=8)
    fig.savefig(ROOT / "figures/extended_coverage_cdf.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.3), constrained_layout=True)
    for w, ax in zip(scenes, axes.flat, strict=True):
        for model, color in [("phase", "#137c66"), ("garl", "#bc5c38")]:
            for r in runs:
                if (
                    r["mode"] == "buffered"
                    and r["model"] == model
                    and r["workload"] == w["id"]
                    and not r["startup_failure"]
                ):
                    paced = read(DATA / r["name"] / "paced.json")
                    valid = [p for p in paced["object_records"] if p["success"]]
                    ax.plot(
                        [int(p["update_id"].split("-")[1]) / 50 for p in valid],
                        r["response_ms"],
                        color=color,
                        alpha=0.7,
                        label=NAMES[model] if r["repeat"] == 1 else None,
                    )
        ax.axhline(20, color="gray", ls=":")
        ax.set(title=f"S{w['id'] + 1}", xlabel="Sensor replay (s)", ylabel="Response (ms)")
        ax.legend(fontsize=8)
    fig.savefig(ROOT / "figures/extended_buffered_traces.pdf")
    plt.close(fig)
    compact = [
        {k: v for k, v in r.items() if k not in ("argv", "response_ms", "startup")} for r in runs
    ]
    validation = {
        "status": "PASS",
        "scope": "Evidence accounting; numerical failures retained, not capacity certification",
        "numerical_parity_passed": all(r["parity"] for r in runs),
        "parity_failed_outputs": sum(r["numerical_failures"] for r in runs),
        "phase_component_parity": parity_details,
        "runs": compact,
        "workloads": scenes,
        "unique_anchors": sum(w["updates"] for w in scenes),
        "offered_total": sum(r["offered"] for r in runs),
        "completed_total": sum(r["completed"] for r in runs),
    }
    (ROOT / "revision/extended_inference_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    overview = []
    for mode in ("bounded", "buffered"):
        for model in NAMES:
            group = [r for r in runs if r["mode"] == mode and r["model"] == model]
            full = sum(r["complete"] for r in group)
            delivered = sum(r["completed"] for r in group)
            overview.append(
                f"- {mode}/{model}: {full} complete deliveries, "
                f"{sum(r['admitted'] for r in group)} admitted; {delivered} outputs; "
                f"{sum(r['numerical_failures'] for r in group)} failed parity checks."
            )
    (ROOT / "revision/extended_inference_report.md").write_text(
        "# Extended persistent inference\n\nAll 72 declared runs retained.\n\n"
        + "\n".join(overview)
        + "\n\nFull per-run statistics and parity are in extended_inference_validation.json.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in validation.items()
                if k not in ("runs", "workloads", "phase_component_parity")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
