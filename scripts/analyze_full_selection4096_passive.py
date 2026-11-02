#!/usr/bin/env python3
"""Analyze the passive full-selection profile without treating samples as time."""

# Generated report paragraphs and category rules stay on one source line for audit.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "phase_ttc_full_selection4096_profile_analysis_v1"
CATEGORY_RULES = (
    (
        "raw_tar",
        (
            "tarfile.py",
            "indexed_rgb_store.py",
            "read_member",
            "_shard (e_jepa_ttc\\detection\\eap_rgb_data.py:127",
        ),
    ),
    ("png_decode", ("PngImagePlugin", "ImageFile.py", "PIL\\Image.py")),
    ("ground_truth", ("evaluation_ground_truth.py", "build_evaluation_ground_truth")),
    (
        "prepare",
        (
            "prepare_sample",
            "model_inputs_to_cuda",
            "eap_rgb_data.py:194",
            "eap_rgb_data.py:199",
            "eap_rgb_data.py:212",
        ),
    ),
    (
        "finite_checks",
        ("require_finite", "finite_matrix", "isfinite", "is_finite", "nan_to_num"),
    ),
    ("decode", ("decode_eap", "monodgp_eap_adapter.py:302")),
    (
        "forward",
        (
            "evaluate (train_detection_campaign.py:226)",
            "forward (",
            "portable_attention.py",
            "ms_deform_attn",
        ),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def classify_folded(path: Path) -> dict[str, Any]:
    """Assign every py-spy sample to exactly one ordered category."""

    counts = {name: 0 for name, _ in CATEGORY_RULES}
    counts["unresolved"] = 0
    total = 0
    malformed_lines = 0
    phase_samples = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.search(r" (\d+)$", line)
        if match is None:
            malformed_lines += 1
            continue
        count = int(match.group(1))
        stack = line[: match.start()]
        total += count
        if "run (train_detection_campaign.py:768)" in stack:
            phase_samples += count
        for category, patterns in CATEGORY_RULES:
            if any(pattern in stack for pattern in patterns):
                counts[category] += count
                break
        else:
            counts["unresolved"] += count
    if total <= 0:
        raise ValueError("no folded samples found")
    if sum(counts.values()) != total:
        raise AssertionError("exclusive category accounting failed")
    return {
        "total_samples": total,
        "malformed_lines": malformed_lines,
        "run_line_768_samples": phase_samples,
        "run_line_768_fraction": phase_samples / total,
        "exclusive_categories": {
            name: {"samples": count, "sample_fraction": count / total}
            for name, count in counts.items()
        },
        "interpretation": "sample fractions, not wall-time or GPU-kernel fractions",
    }


def build_analysis(repository: Path, tool_path: Path) -> dict[str, Any]:
    passive_path = repository / "output/performance/full_selection4096_passive_v1.json"
    profile_dir = repository / "output/performance/full_selection4096_pyspy_v1"
    raw_path = profile_dir / "pyspy_raw.txt"
    console_path = profile_dir / "console_live.txt"
    identity_path = profile_dir / "process_identity.json"
    for path in (passive_path, raw_path, console_path, identity_path, tool_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    passive = _json(passive_path)
    identity = _json(identity_path)
    folded = classify_folded(raw_path)
    console = console_path.read_text(encoding="utf-8")
    console_match = re.search(r"Samples:\s*(\d+)\s+Errors:\s*(\d+)", console)
    if console_match is None:
        raise ValueError("live console lacks sample/error receipt")
    console_samples = int(console_match.group(1))
    sampling_errors = int(console_match.group(2))
    if console_samples != folded["total_samples"]:
        raise ValueError("console and folded sample counts differ")
    trainer = passive["trainer"]
    gpu = passive["gpu"]
    elapsed = float(passive["elapsed_seconds"])
    trainer_reads = int(trainer["io_delta"]["read_bytes"])
    system_reads = int(passive["system_disk_delta"]["read_bytes"])
    if passive["metrics"]["updates_completed"] != 0:
        raise ValueError("passive window unexpectedly includes training updates")
    if passive["evaluation"]["start"] != passive["evaluation"]["end"]:
        raise ValueError("evaluation artifacts changed during passive window")
    return {
        "schema": SCHEMA,
        "scope": "passive 60-second sample inside update-4096 full-selection evaluation",
        "phase": {
            "confirmed_by_stack": folded["run_line_768_fraction"] >= 0.99,
            "trainer_run_line": 768,
            "updates_completed": 0,
            "evaluation_artifacts_unchanged": True,
            "full_selection_finished_in_window": False,
        },
        "pyspy": {
            **folded,
            "sampling_errors": sampling_errors,
            "console_captured_live": True,
            "gpu_kernels_visible": False,
        },
        "passive": {
            "elapsed_seconds": elapsed,
            "trainer_cpu_seconds": float(trainer["cpu_seconds_delta"]),
            "mean_logical_core_equivalents": float(trainer["mean_logical_core_equivalents"]),
            "trainer_read_bytes": trainer_reads,
            "trainer_read_mib_per_second": trainer_reads / elapsed / (1024**2),
            "system_read_bytes": system_reads,
            "system_read_mib_per_second": system_reads / elapsed / (1024**2),
            "gpu_utilization_percent_mean": float(gpu["utilization_gpu_percent"]["mean"]),
            "gpu_utilization_percent_peak": float(gpu["utilization_gpu_percent"]["max"]),
            "nonzero_gpu_throttle_samples": int(gpu["nonzero_throttle_samples"]),
            "rss_bytes_min": int(trainer["rss_bytes"]["min"]),
            "rss_bytes_max": int(trainer["rss_bytes"]["max"]),
        },
        "conclusion": {
            "dominant_evidence": "HOST_CPU_WORK_WITH_CONCURRENT_READ_IO",
            "physical_reading": (
                "Trainer and system read counters increased materially, so reading occurred. "
                "The receipts do not distinguish physical media latency from OS cache/native I/O."
            ),
            "host_cpu": (
                "The trainer consumed 217.64 CPU-seconds in 60.04 seconds (3.62 logical-core "
                "equivalents), while PNG decode, preparation and Python-visible forward stacks "
                "account for most sampled stacks."
            ),
            "gpu_boundary": (
                "Mean GPU utilization was 13.38% with no throttle samples, but low utilization "
                "alone does not prove starvation and py-spy cannot measure GPU kernel time."
            ),
            "action": (
                "Profile reader/decode/preparation and forward submission separately before "
                "changing GPU kernels; preserve a separate end-to-end full-selection timing."
            ),
        },
        "limitations": [
            "py-spy sample fractions are not wall-time fractions.",
            "Python stacks cannot measure asynchronous CUDA kernel duration.",
            "The 60-second window did not finish or write a new full-selection artifact.",
            "No causal speedup or training-rate claim follows from this profile.",
        ],
        "sources": {
            "passive": {
                "path": "output/performance/full_selection4096_passive_v1.json",
                "sha256": _sha256(passive_path),
            },
            "pyspy_raw": {
                "path": "output/performance/full_selection4096_pyspy_v1/pyspy_raw.txt",
                "sha256": _sha256(raw_path),
            },
            "console_live": {
                "path": "output/performance/full_selection4096_pyspy_v1/console_live.txt",
                "sha256": _sha256(console_path),
            },
            "process_identity": {
                "path": "output/performance/full_selection4096_pyspy_v1/process_identity.json",
                "sha256": _sha256(identity_path),
                "process_id": int(identity["ProcessId"]),
            },
            "pyspy_tool": {
                "path": "<RGB_PORT_EVIDENCE_ROOT>/tools/py-spy.exe",
                "sha256": _sha256(tool_path),
            },
        },
    }


def render_section(analysis: dict[str, Any]) -> str:
    def decimal(value: float) -> str:
        return f"{value:.2f}".replace(".", ",")

    categories = analysis["pyspy"]["exclusive_categories"]
    passive = analysis["passive"]
    sample_phase = f"{analysis['pyspy']['run_line_768_samples']:,}".replace(",", ".")
    sample_total = f"{analysis['pyspy']['total_samples']:,}".replace(",", ".")
    phase_fraction = f"{analysis['pyspy']['run_line_768_fraction']:.2%}".replace(".", ",")
    lines = [
        "## Perfil pasivo de selección completa en update 4096",
        "",
        f"La pila `run:768` aparece en {sample_phase} de {sample_total} muestras ({phase_fraction}); junto con la identidad del proceso, confirma que la ventana corresponde al bloque de selección completa, no a entrenamiento. Durante 60,04 s no avanzó ningún update ni apareció un artefacto nuevo de evaluación.",
        "",
        "| Categoría exclusiva | Muestras | Fracción de muestras |",
        "|---|---:|---:|",
    ]
    labels = {
        "forward": "Forward visible desde Python",
        "png_decode": "Decodificación PNG",
        "prepare": "Preparación/H2D",
        "decode": "Decode de predicciones",
        "finite_checks": "Comprobaciones finitas",
        "ground_truth": "Construcción GT",
        "raw_tar": "Lectura TAR visible",
        "unresolved": "No resuelto",
    }
    for category in (
        "forward",
        "png_decode",
        "prepare",
        "decode",
        "finite_checks",
        "ground_truth",
        "raw_tar",
        "unresolved",
    ):
        value = categories[category]
        lines.append(
            f"| {labels[category]} | {value['samples']} | {value['sample_fraction']:.2%} |"
        )
    lines += [
        "",
        "Son fracciones de muestras de stacks, no fracciones de tiempo de pared ni de kernels GPU. El tee contemporáneo registra 1.469 muestras y 29 errores de muestreo.",
        "",
        f"El proceso consumió {decimal(passive['trainer_cpu_seconds'])} CPU-s en {decimal(passive['elapsed_seconds'])} s ({decimal(passive['mean_logical_core_equivalents'])} cores lógicos equivalentes). Leyó {decimal(passive['trainer_read_mib_per_second'])} MiB/s según el contador del trainer y {decimal(passive['system_read_mib_per_second'])} MiB/s a nivel de sistema. Esto confirma CPU host activo y lecturas concurrentes, pero no distingue latencia de medio físico frente a caché del SO o I/O nativo.",
        "",
        f"La GPU promedió {decimal(passive['gpu_utilization_percent_mean'])}% y alcanzó {passive['gpu_utilization_percent_peak']:.0f}%, sin muestras de throttling. La baja utilización por sí sola no demuestra starvation: py-spy no observa duración de kernels CUDA asíncronos.",
        "",
        "La prioridad práctica es medir por separado lectura/PNG/preparación y submission/forward, conservando aparte el tiempo end-to-end de selección completa. Este perfil no acredita un speedup ni una mejora del entrenamiento.",
        "",
        "Fuentes hash-bound: `full_selection4096_passive_v1.json`, `full_selection4096_pyspy_v1/pyspy_raw.txt`, `console_live.txt`, `process_identity.json` y el binario `py-spy.exe`; sus SHA-256 están en `output/performance/full_selection4096_profile_analysis_v1.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--pyspy-tool", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/performance/full_selection4096_profile_analysis_v1.json"),
    )
    parser.add_argument("--section-output", type=Path)
    args = parser.parse_args()
    repository = args.repository.resolve()
    analysis = build_analysis(repository, args.pyspy_tool.resolve())
    output = args.output if args.output.is_absolute() else repository / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.section_output is not None:
        section = (
            args.section_output
            if args.section_output.is_absolute()
            else repository / args.section_output
        )
        if section.exists():
            raise FileExistsError(f"refusing to overwrite: {section}")
        section.write_text(render_section(analysis), encoding="utf-8")


if __name__ == "__main__":
    main()
