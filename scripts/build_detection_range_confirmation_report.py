"""Render the verified full-selection range comparison; never turn partials into results."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.benchmark_evaluation import CLASSES  # noqa: E402
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def main() -> None:
    """Generate tables and a figure only from completed, hash-bound diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--aligned", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wait-seconds", type=float, default=0)
    args = parser.parse_args()
    if not np.isfinite(args.wait_seconds) or args.wait_seconds < 0:
        raise ValueError("Finite nonnegative wait budget required")
    start = time.perf_counter()
    complete = args.diagnostics / "COMPLETE.json"
    while not complete.is_file():
        if (args.diagnostics / "FAILURE.json").exists():
            raise RuntimeError("Diagnostic failed; do not build an accepted report")
        if time.perf_counter() - start >= args.wait_seconds:
            raise TimeoutError("Verified full diagnostics are not yet available")
        time.sleep(10)
    receipt = json.loads(complete.read_text())
    if (
        receipt["status"] != "VERIFIED"
        or receipt["frames"] != 23958
        or receipt["full_metric_exact_parity"] is not True
    ):
        raise ValueError("Full population/parity evidence is missing")
    for name, expected in receipt["files"].items():
        if digest(args.diagnostics / name) != expected:
            raise ValueError("Diagnostic results changed")
    comparison = json.loads((args.diagnostics / "COMPARISON.json").read_text())
    manifest = json.loads((args.aligned / "MANIFEST.json").read_text())
    old_path = args.aligned / "ranked.json"
    if digest(old_path) != manifest["ranked.json"]:
        raise ValueError("Reference metrics changed")
    old = json.loads(old_path.read_text())
    new = comparison["metrics"]
    if old["mAP"] != comparison["incumbent_mAP"]:
        raise ValueError("Reference summary differs")
    sequences = sorted(comparison["candidate_by_sequence"])
    if sequences != sorted(comparison["incumbent_by_sequence"]) or len(sequences) != 8:
        raise ValueError("Expected the same eight public sequences")
    deltas = np.array(
        [
            comparison["candidate_by_sequence"][s]["mAP"]
            - comparison["incumbent_by_sequence"][s]["mAP"]
            for s in sequences
        ]
    )
    # This explicitly estimates the mean of sequence mAP deltas, not pooled mAP.
    draws = np.random.default_rng(7).integers(0, len(sequences), size=(2000, len(sequences)))
    interval = np.quantile(deltas[draws].mean(axis=1), [0.025, 0.975])
    bootstrap = {
        "estimand": "Unweighted mean of eight per-sequence mAP differences; not pooled mAP",
        "seed": 7,
        "resamples": 2000,
        "resampling_unit": "sequence",
        "point": float(deltas.mean()),
        "percentile95": interval.tolist(),
        "caveat": "Exploratory interval with eight exposed sequences; independence unverified",
    }
    lines = [
        "# Confirmación del checkpoint de 150 m",
        "",
        "Mismas 23.958 imágenes públicas, GT raw, ocho clases y perfil alineado. "
        "Las métricas se reconstruyeron desde los shards con paridad exacta.",
        "",
        "| Variante | mAP completa |",
        "|---|---:|",
        f"| Control 60 m, update cero | {comparison['control_mAP']:.8f} |",
        f"| Incumbente con ranking | {old['mAP']:.8f} |",
        f"| Candidato 150 m, update 1.024 | {new['mAP']:.8f} |",
        "",
        f"Diferencia frente al control: {comparison['delta_control']:+.8f}; "
        f"frente al incumbente: {comparison['delta_incumbent']:+.8f}.",
        f"Umbral macro del plan original cumplido: **{comparison['macro_gate_pass']}**. "
        "No implica promoción automática ni excelencia.",
        "",
        "| Clase | GT | Incumbente AP | Candidato AP | Diferencia |",
        "|---|---:|---:|---:|---:|",
    ]
    class_deltas = []
    for cls in CLASSES:
        delta = new["classes"][cls]["AP"] - old["classes"][cls]["AP"]
        class_deltas.append(delta)
        # Support comes from the metric dictionary, never from the filtered train domain.
        support = new["classes"][cls]["gt_count"]
        lines.append(
            f"| {cls} | {support} | {old['classes'][cls]['AP']:.8f} | "
            f"{new['classes'][cls]['AP']:.8f} | {delta:+.8f} |"
        )
    lines += [
        "",
        "Caídas mayores de 0,01 AP: " + json.dumps(comparison["class_declines_over_0_01"]) + ".",
        "",
        "| Profundidad óptica | GT | Recall máximo 2 m anterior | Candidato |",
        "|---|---:|---:|---:|",
    ]
    for key, values in comparison["range_recall"].items():
        a, b = values["reference"], values["candidate"]
        if a["gt"] != b["gt"]:
            raise ValueError("Range denominators differ")
        metric = "same_class_maximum_2m"
        lines.append(
            f"| {key.removeprefix('optical_depth_m/')} | {b['gt']} | "
            f"{a['recall'][metric]:.6f} | {b['recall'][metric]:.6f} |"
        )
    lines += [
        "",
        "El máximo usa asignación de cardinalidad máxima con clase correcta; "
        "no depende del orden de scores. El JSON conserva también greedy y umbrales "
        "0,5/1/2/4 m. No es AP por rango.",
        "",
        "| Secuencia | Incumbente mAP | Candidato mAP | Diferencia |",
        "|---|---:|---:|---:|",
    ]
    for s, delta in zip(sequences, deltas, strict=True):
        a, b = comparison["incumbent_by_sequence"][s], comparison["candidate_by_sequence"][s]
        lines.append(f"| {s} | {a['mAP']:.8f} | {b['mAP']:.8f} | {delta:+.8f} |")
    lines += [
        "",
        f"Media de diferencias por secuencia: {bootstrap['point']:+.8f}. "
        f"Intervalo percentil exploratorio 95 %: [{interval[0]:+.8f}, {interval[1]:+.8f}] "
        "con 2.000 remuestreos por secuencia, seed 7. Este estimando difiere de la "
        "mAP global agrupada. Ocho secuencias expuestas y sin independencia certificada "
        "no sostienen una afirmación de significancia o generalización independiente.",
        "",
        "El control reutiliza predicciones verificadas de su update cero; la equivalencia "
        "cubre detecciones decodificadas, no logits auxiliares ni entrenamiento. "
        "No hay nuevo resultado oficial ni etiquetas privadas.",
        "",
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(args.output / "SEQUENCE_BOOTSTRAP.json", bootstrap)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    colors = ["#23784c" if value >= 0 else "#b63a38" for value in class_deltas]
    ax.barh(list(CLASSES), class_deltas, color=colors)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("AP candidate − incumbent")
    ax.set_title("150 m checkpoint · full public selection · 8 classes")
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(args.output / "class_ap_delta.png", dpi=160)
    plt.close(fig)
    write_json(
        args.output / "MANIFEST.json",
        {
            "diagnostic_complete_sha256": digest(complete),
            "comparison_sha256": digest(args.diagnostics / "COMPARISON.json"),
            "ranked_metrics_sha256": digest(old_path),
            "generator_sha256": digest(Path(__file__)),
            "files": {p.name: digest(p) for p in args.output.iterdir() if p.is_file()},
            "promotion": False,
        },
    )


if __name__ == "__main__":
    main()
