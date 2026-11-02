"""Regenerate the raw-selection diagnostic tables from verified JSON evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluate_detection_benchmark_local import read, sha


def main() -> None:
    """Verify receipts before emitting tables; never type metrics into documents."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = read(args.evidence / "COMPLETE.json")
    if receipt["status"] != "VERIFIED":
        raise ValueError("Unverified evidence")
    for name, digest in receipt["files"].items():
        if sha(args.evidence / name) != digest:
            raise ValueError(f"Changed evidence: {name}")
    diagnostics = read(args.evidence / "ERROR_DIAGNOSTICS.json")
    variants = {
        name: read(args.evidence / f"{name}.json")
        for name in ("ranked", "oracle_class", "oracle_depth", "oracle_size", "oracle_rank")
    }
    args.output.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Diagnóstico del detector eAP sobre selección pública raw",
        "",
        "Las sustituciones oracle usan GT público. No son modelos entrenados, mejoras",
        "conseguidas ni cotas superiores generales. Mismo pool de propuestas; sin test privado.",
        "",
        f"Población: {receipt['frames']} frames; {variants['ranked']['gt_boxes']} GT; "
        f"{variants['ranked']['pred_boxes']} propuestas. "
        "Paridad completa con el JSON alineado verificada.",
        "",
        "## Sustituciones controladas",
        "",
        "| Variante | mAP | ATE | ASE | AOE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in variants.items():
        errors = result["tp_errors"]
        lines.append(
            f"| {name} | {result['mAP']:.8f} | {errors['trans_err']:.6f} | "
            f"{errors['scale_err']:.6f} | {errors['orient_err']:.6f} |"
        )
    lines += [
        "",
        "Clase/tamaño: asignación agnóstica de cardinalidad máxima y mínima distancia XY <4 m.",
        "Profundidad: asignación de misma clase por rayos de cámara normalizados con norma <0,02;",
        "se conserva el rayo predicho y se sustituye su profundidad óptica. No se cambian cajas",
        "sin asignación. Ranking: score=1/(1+distancia al GT más próximo de igual clase),",
        "sin eliminar duplicados. Ninguna de estas variantes es inferencia desplegable.",
        "",
        "## Clases",
        "",
        "| Clase | GT | AP real | AP oracle profundidad | "
        "Recall 2m clase | Recall 2m agnóstico máximo |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cls, result in variants["ranked"]["classes"].items():
        group = diagnostics["groups"][f"class/{cls}"]
        lines.append(
            f"| {cls} | {group['gt']} | {result['AP']:.6f} | "
            f"{variants['oracle_depth']['classes'][cls]['AP']:.6f} | "
            f"{group['recall']['same_class_greedy_2m']:.6f} | "
            f"{group['recall']['agnostic_maximum_2m']:.6f} |"
        )
    for prefix, title in (
        ("optical_depth_m/", "Profundidad óptica"),
        ("radial_ego_xy_m/", "Distancia radial ego XY"),
        ("clipped_height_720px/", "Altura proyectada en imagen original de 720 px"),
        ("historical_domain/", "Dominio histórico"),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "| Estrato | GT | Recall clase 0,5m | 1m | 2m | 4m |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, group in diagnostics["groups"].items():
            if name.startswith(prefix):
                recalls = " | ".join(
                    f"{group['recall'][f'same_class_greedy_{t}m']:.6f}"
                    for t in ("0.5", "1", "2", "4")
                )
                lines.append(f"| {name.removeprefix(prefix)} | {group['gt']} | {recalls} |")
    lines += [
        "",
        "## AP por secuencia",
        "",
        "| Secuencia | Frames | GT | mAP |",
        "|---|---:|---:|---:|",
    ]
    for name, row in read(args.evidence / "BY_SEQUENCE.json").items():
        lines.append(f"| {name} | {row['samples']} | {row['gt_boxes']} | {row['mAP']:.6f} |")
    lines += [
        "",
        "## Límites",
        "",
        "- Recall por rango tras matching global por frame; no es AP por rango.",
        "- Cardinalidad máxima: cota solo para el pool y grafo de distancia fijos.",
        "- Recall agnóstico comparte propuestas; no es una cota independiente por clase.",
        "- Ocho secuencias expuestas a desarrollo; independencia no certificada.",
        "- Errores geométricos condicionados a matches <2 m están en ERROR_DIAGNOSTICS.json;",
        "  no sustituyen los errores TP y defaults del benchmark.",
        "- Iluminación, visibilidad e identidades de adquisición no certificadas por estos shards.",
        "- No hay nuevos pesos ni nuevo envío oficial.",
        "",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    with (args.output / "recall_strata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stratum", "gt", "matching", "matched", "recall"])
        for name, group in diagnostics["groups"].items():
            for metric, recall in group["recall"].items():
                writer.writerow([name, group["gt"], metric, group["matched"][metric], recall])


if __name__ == "__main__":
    main()
