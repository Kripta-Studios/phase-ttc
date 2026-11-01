"""Regenerate the TRAIN distance/size report from the verified census JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.range_campaign import digest  # noqa: E402


def main() -> None:
    """Render measured counts, without presenting a target audit as model improvement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.census.read_text())
    if result["status"] != "VERIFIED" or result["test_labels_read"] is not False:
        raise ValueError("Verified TRAIN census required")
    counts = result["counts"]
    total150 = sum(counts["admitted150"].values())
    total400 = sum(counts["admitted400"].values())
    delta = total400 - total150
    truck_delta = counts["admitted400"]["truck"] - counts["admitted150"]["truck"]
    lines = [
        "# Auditoría del rango de entrenamiento — 2026-11-01",
        "",
        f"Fuente: `{args.census.as_posix()}`.",
        f"SHA-256: `{digest(args.census)}`.",
        "",
        f"Se auditaron {result['frames']:,} fotogramas TRAIN de "
        f"{len(result['sequences'])} secuencias. No se leyeron etiquetas de test.",
        f"Máximo de profundidad óptica anotada: "
        f"{result['raw_center_depth_extrema_m'][1]:.6f} m; máximo de distancia "
        f"horizontal ego: {result['raw_horizontal_range_extrema_m'][1]:.6f} m.",
        "No hay anotaciones TRAIN en 300–400 m ni por encima de 400 m.",
        "",
        "## Anotaciones por clase",
        "",
        "| Clase | Raw | Admitidas <60 m | <150 m | <400 m | Ganancia 150→400 m |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cls in sorted(counts["raw"]):
        values = [
            counts[key].get(cls, 0) for key in ("raw", "admitted60", "admitted150", "admitted400")
        ]
        lines.append(
            "| "
            + cls
            + " | "
            + " | ".join(f"{v:,}" for v in values)
            + f" | {values[3] - values[2]:,} |"
        )
    lines.extend(
        [
            "",
            f"Ampliar de 150 m a todo el rango observado añade {delta:,} anotaciones "
            f"({delta / total150:.2%}); {truck_delta:,} son camiones "
            f"({truck_delta / delta:.2%} del incremento). Son apariciones correlacionadas, "
            "no objetos o escenas independientes.",
            "",
            "## Distribución de profundidad y tamaño",
            "",
            "| Profundidad óptica [m) | Raw | Admitidas por geometría |",
            "|---|---:|---:|",
        ]
    )
    for interval, row in result["optical_depth_bins_m"].items():
        lines.append(
            f"| {interval.replace('_', '–')} | {sum(row['raw_counts'].values()):,} "
            f"| {sum(row['geometry_admissible_counts'].values()):,} |"
        )
    lines.extend(
        [
            "",
            "Altura proyectada mediana, tras resize a 640×384:",
            "",
            "| Profundidad [m) | Clase | Apariciones | Altura mediana [px] |",
            "|---|---|---:|---:|",
        ]
    )
    for interval, row in result["optical_depth_bins_m"].items():
        if float(interval.split("_")[0]) < 150:
            continue
        for cls, value in sorted(row["pixel_size_by_class"].items()):
            height = value["height_px"]
            lines.append(
                f"| {interval.replace('_', '–')} | {cls} | {height['count']:,} "
                f"| {height['quantiles_min_p10_p50_p90_max'][2]:.3f} |"
            )
    lines.extend(
        [
            "",
            f"Máximo de targets admitidos por imagen: "
            f"{result['capacity']['admitted400']['max']}; "
            f"fotogramas con más de 50: {len(result['capacity']['admitted400']['over_50'])}.",
            "",
            "## Comprobaciones y límites",
            "",
            f"- {result['parity']}.",
            "- Dos tests comparan la proyección vectorizada con el productor original, "
            "incluidos límites de profundidad y exclusiones.",
            "- Los filtros reproducen la aproximación pinhole existente: centro dentro "
            "de imagen, esquinas delante de cámara y caja recortada mayor de un píxel. "
            "No certifican ausencia de oclusión ni detectabilidad visual.",
            "- Los límites son profundidad óptica, con máximo exclusivo, no distancia euclídea.",
            "- Incluir más camiones lejanos no aporta soporte a bicycle u other_vehicle.",
            "- Esto mide targets disponibles; no demuestra una mejora de mAP.",
            "",
            "Siguiente comparación de rango: cubrir todos los targets observados con "
            "un margen declarado (300 m), manteniendo población, presupuesto y "
            "evaluación de ocho clases. No hacen falta etiquetas sintéticas a 400 m.",
            "",
            "Reproducción:",
            "",
            "```powershell",
            "python scripts/audit_detection_long_range.py --release RELEASE --dataset DATASET "
            "--reference-census CENSUS_60_150 --output NUEVO_DIRECTORIO",
            "python scripts/build_detection_long_range_report.py "
            "--census NUEVO_DIRECTORIO/CENSUS.json --output INFORME.md",
            "```",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
