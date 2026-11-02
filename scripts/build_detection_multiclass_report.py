"""Regenerate a compact multiclass report from receipt-bound full-selection JSON."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.benchmark_evaluation import CLASSES, PROFILE_SHA256  # noqa: E402
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def main() -> None:
    """Publish exact source snapshots and preserve pending/negative outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    aligned_path = args.aligned / "COMPARISON.json"
    aligned = json.loads(aligned_path.read_text())
    if aligned["profile_sha256"] != PROFILE_SHA256:
        raise ValueError("Reference profile differs")
    manifest = json.loads((args.aligned / "MANIFEST.json").read_text())
    if manifest["COMPARISON.json"] != digest(aligned_path):
        raise ValueError("Reference comparison bytes differ")
    variants = {}
    source_hashes = {str(aligned_path): digest(aligned_path)}
    for name, summary in aligned["variants"].items():
        path = args.aligned / f"{name}.json"
        metrics = json.loads(path.read_text())
        if digest(path) != manifest[path.name] or any(metrics[k] != v for k, v in summary.items()):
            raise ValueError("Reference metrics differ from their summary")
        variants[name] = metrics
        source_hashes[str(path)] = digest(path)
    states = {}
    for folder_name in ("v1", "top50_v1"):
        folder = args.campaign / folder_name
        result_path = folder / "RESULTS.json"
        if not result_path.exists():
            states[folder_name] = "PENDING"
            continue
        intake_path = folder / "INTAKE.json"
        result = json.loads(result_path.read_text())
        intake = json.loads(intake_path.read_text())
        if (
            result["intake_sha256"] != digest(intake_path)
            or intake["aligned_sha256"] != digest(aligned_path)
            or intake["checkpoint_sha256"] != aligned["checkpoint_sha256"]
            or result["metrics"]["samples"] != aligned["samples"]
            or result["metrics"]["profile_sha256"] != PROFILE_SHA256
            or result["reference_metrics"] != aligned["variants"]
        ):
            raise ValueError("Multiclass experiment and reference differ")
        for name, expected in result["shard_receipt_sha256"].items():
            complete_path = folder / "shards" / name / "COMPLETE.json"
            complete = json.loads(complete_path.read_text())
            if (
                digest(complete_path) != expected
                or digest(complete_path.parent / "matches.npz") != complete["matches_sha256"]
            ):
                raise ValueError("Saved matching records differ")
        for path in (result_path, intake_path):
            source_hashes[str(path)] = digest(path)
            shutil.copy2(path, args.output / f"{folder_name}_{path.name}")
        variants[folder_name] = result["metrics"]
        states[folder_name] = "EVALUATED_NOT_PROMOTED"
    rows = []
    for name, metrics in variants.items():
        rows.append(f"| {name} | {metrics['mAP']:.8f} | {metrics['pred_boxes']} |")
    lines = [
        "# Hipótesis de clase sobre geometría congelada",
        "",
        "Tabla regenerada de JSON verificados. Todas las variantes conservan las ocho clases",
        "en la media y usan las mismas 23.958 imágenes públicas de selección y GT raw.",
        "La selección ya se ha consultado durante el desarrollo; no es test independiente.",
        "",
        "| Variante | mAP raw alineada | Propuestas |",
        "|---|---:|---:|",
        *rows,
        "",
        "| Clase | " + " | ".join(variants) + " |",
        "|---|" + "---:|" * len(variants),
    ]
    for cls in CLASSES:
        lines.append(
            f"| {cls} | "
            + " | ".join(f"{m['classes'][cls]['AP']:.8f}" for m in variants.values())
            + " |"
        )
    lines += [
        "",
        "Estados: " + ", ".join(f"{k}={v}" for k, v in states.items()) + ".",
        "",
        "La expansión conserva los centros, tamaños, orientación y velocidad. No aumenta",
        "la cobertura geométrica agnóstica de clase. Los logits alternativos no usan el",
        "ranker TRAIN congelado: aplicarlo sin una receta nueva mezclaría semánticas.",
        "",
        "Ninguna fila certifica equivalencia con el scorer privado ni excelencia. La regla",
        "de promoción exige +0,005 frente al incumbent ranked, revisar caídas >0,01 por",
        "clase y validar un productor ejecutable sin etiquetas antes de empaquetar.",
        "",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        args.output / "SNAPSHOT_RECEIPT.json",
        {
            "sources": source_hashes,
            "generator_sha256": digest(Path(__file__)),
            "report_sha256": digest(args.output / "REPORT.md"),
            "states": states,
            "promotion": False,
        },
    )


if __name__ == "__main__":
    main()
