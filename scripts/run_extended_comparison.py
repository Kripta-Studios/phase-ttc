"""Freeze and execute a prospective multi-sequence persistent-service campaign."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from operational.local_comparison.extended import longest_interval, write  # noqa: E402
from operational.local_comparison.persistent import materialize_runtime  # noqa: E402
from operational.local_comparison.service_audit import sha, verify_files  # noqa: E402
from operational.train40_runtime.replay_inputs import generate_sensor_clock_replay  # noqa: E402

SEQUENCES = [
    "CPLA-low",
    "CCRm-medium-0-overlap-0",
    "CCRs-3-low-100-overlap-100",
    "CCRs-side-high",
    "CPNA-high",
    "CPNAO-high",
]


def main() -> None:
    """All measurements precede all independent numerical reference jobs."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--scout", type=Path, required=True)
    cli.add_argument("--pilot", action="store_true")
    args = cli.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    archive = ROOT / (
        "docs/train40_runtime_council_20261028/evidence/council_v1/artifacts/"
        "train40_runtime_council_20261028/paced/memory_cache_overlap_reverse1_250_v1"
    )
    binding = json.loads((archive / "binding.json").read_text(encoding="utf-8"))
    runtime = out / "runtime"
    receipt = materialize_runtime(ROOT, runtime, archive / "sources", binding)
    overlay = "operational/train40_runtime/paced.py"
    shutil.copyfile(ROOT / overlay, runtime / overlay)
    receipt["measurement_overlay"] = {overlay: sha(runtime / overlay)}
    receipt["tree_source_sha256"][overlay] = sha(runtime / overlay)
    write(out / "runtime_sources.json", receipt)
    shutil.copyfile(args.scout, out / "scout_manifest.json")
    scout = json.loads(args.scout.read_text(encoding="utf-8"))
    config = binding["configuration"]
    base_path = ROOT / config["base_manifest"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    # Rebuild all clock/annotation-only slots, so an edited scout cannot select inputs.
    import h5py

    bounds = {}
    for row in base["rows"]:
        sequence = row.get("physical_sequence_id", row["sequence_id"])
        if sequence not in bounds:
            with h5py.File(row["raw_path"], "r") as handle:
                stamps = cast(h5py.Dataset, handle["prophesee/event_cam_left/t"])
                bounds[sequence] = (max(700_000, int(stamps[0]) + 620_000), int(stamps[-1]))
    rebuilt = generate_sensor_clock_replay(base, bounds_us=bounds)
    if len(bounds) != 32 or json.dumps(rebuilt, sort_keys=True) != json.dumps(
        scout, sort_keys=True
    ):
        raise ValueError("Scout differs from the complete 32-sequence label-free reconstruction")
    freeze = ROOT / "output/garl_evttc_competition_20261028/RUN_FREEZE.json"
    (out / "cohorts").mkdir()
    workloads = []
    for index, sequence in enumerate(SEQUENCES[:1] if args.pilot else SEQUENCES):
        interval = longest_interval(
            [r for r in scout["rows"] if r["physical_sequence_id"] == sequence]
        )
        if args.pilot:
            interval = interval[:20]
        bounds = (interval[0]["anchor_us"], interval[-1]["anchor_us"])
        manifest = generate_sensor_clock_replay(
            base, sequence_ids=[sequence], bounds_us={sequence: bounds}
        )
        if manifest["excluded"] or json.dumps(manifest["rows"], sort_keys=True) != json.dumps(
            interval, sort_keys=True
        ):
            raise ValueError("Selected interval changed after scout")
        path = out / "cohorts" / f"manifest_{index}.json"
        write(path, manifest)
        workloads.append(
            {
                "id": index,
                "sequence": sequence,
                "updates": len(interval),
                "bounds_us": bounds,
                "manifest": str(path),
                "manifest_sha256": sha(path),
            }
        )
    commands = []
    for mode, repeats, queue in [
        ("bounded", 1 if args.pilot else 4, 2),
        ("buffered", 1 if args.pilot else 2, 64),
    ]:
        for repeat in range(repeats):
            order = workloads[repeat % len(workloads) :] + workloads[: repeat % len(workloads)]
            for workload in order:
                for model in ("phase", "garl") if repeat % 2 == 0 else ("garl", "phase"):
                    name = f"{mode}_s{workload['id']}_r{repeat + 1}_{model}"
                    argv = [
                        sys.executable,
                        str(runtime / f"scripts/run_extended_{model}.py"),
                        "--runtime-root",
                        str(runtime),
                        "--manifest",
                        workload["manifest"],
                        "--sequence",
                        workload["sequence"],
                        "--updates",
                        str(workload["updates"]),
                        "--queue-capacity",
                        str(queue),
                        "--rest-seconds",
                        "15",
                        "--output",
                        str(out / name),
                    ]
                    if model == "garl":
                        argv += ["--garl-freeze", str(freeze)]
                    else:
                        argv += ["--workspace-root", str(ROOT)]
                        for key in ("candidate", "reference", "base_manifest"):
                            path = Path(config[key])
                            argv += [
                                "--" + key.replace("_", "-"),
                                str(path if path.is_absolute() else ROOT / path),
                            ]
                        argv += [
                            "--objects",
                            "1",
                            "--raw-source",
                            "memory",
                            "--max-preloaded-source-bytes",
                            str(12 * 1024**3),
                            "--backend",
                            "prefix",
                            "--event-query-graphs",
                            "--event-normalization",
                            "topk",
                            "--rgb-workers",
                            "8",
                            "--prime-native-histories",
                            "--rgb-feature-cache",
                            "--transport-reuse",
                            "reverse-summaries",
                            "--overlap-event-with-rgb-preparation",
                            "--wait-strategy",
                            "sleep",
                        ]
                    commands.append(
                        {
                            "name": name,
                            "model": model,
                            "workload": workload["id"],
                            "repeat": repeat + 1,
                            "mode": mode,
                            "queue": queue,
                            "argv": argv,
                        }
                    )
    protocol = {
        "schema": "extended_persistent_service_v1",
        "pilot": args.pilot,
        "workloads": workloads,
        "commands": commands,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "rate_hz": 50,
        "seed": 7,
        "rest_seconds": 15,
        "reference_phase": "AFTER_ALL_TIMING",
        "precision": "FP32_TF32_OFF",
        "targets_read": False,
        "optimizer_updates": 0,
        "selection": (
            "Declared six scenarios; longest valid interval, earliest tie; "
            "no timing/target selection"
        ),
        "scope": (
            "Causal preloaded RAW CPU arrivals; one annotated target; no physical sensor claim"
        ),
        "runtime_receipt_sha256": sha(out / "runtime_sources.json"),
        "garl_freeze": str(freeze),
        "garl_freeze_sha256": sha(freeze),
        "config": config,
    }
    write(out / "protocol.json", protocol)
    paths = [
        out / "protocol.json",
        out / "runtime_sources.json",
        out / "scout_manifest.json",
        freeze,
        archive / "binding.json",
        base_path,
        *[Path(w["manifest"]) for w in workloads],
    ]
    paths += [
        runtime / name
        for key in ("tree_source_sha256", "configuration_sha256")
        for name in receipt[key]
    ]
    native = json.loads(freeze.read_text(encoding="utf-8"))["inventory"]
    paths += [Path(r["path"]) for r in native["upstream"]["native_import_graph"]]
    paths += [
        Path(v[k]["path"]) for v in native["variants"].values() for k in ("checkpoint", "config")
    ]
    from scripts.benchmark_train40_runtime import _candidate_inventory

    paths += list(_candidate_inventory(Path(config["candidate"])))
    inventory = {str(p): sha(p) for p in paths}
    write(out / "input_inventory.json", inventory)
    results = []
    env = {**os.environ, "PYTHONUTF8": "1", "UV_PROJECT_ENVIRONMENT": str(ROOT / ".venv-transfer")}
    for command in commands:
        name = command["name"]
        print(f"START {name}", flush=True)
        with (out / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command["argv"],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        files = {p.name: sha(p) for p in (out / name).glob("*.json")}
        results.append({"name": name, "returncode": result.returncode, "files": files})
        write(out / "results.json", results)
        print(f"END {name}: {result.returncode}", flush=True)
    write(out / "end_integrity.json", verify_files(inventory))


if __name__ == "__main__":
    main()
