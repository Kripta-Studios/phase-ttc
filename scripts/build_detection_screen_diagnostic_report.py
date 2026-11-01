"""Regenerate fixed-panel class, depth and oracle diagnostic tables from JSON."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def source_bytes(path: Path) -> bytes:
    """Read original JSON or its lossless repository archive."""
    if path.is_file():
        return path.read_bytes()
    return gzip.decompress(path.with_name(path.name + ".gz").read_bytes())


def digest(path: Path) -> str:
    """Hash source evidence without changing it."""
    return hashlib.sha256(source_bytes(path)).hexdigest()


def main() -> None:
    """Keep counterfactual GT substitutions distinct from trained model quality."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("New report destination required")
    root = args.diagnostics
    receipt = json.loads((root / "COMPLETE.json").read_text())
    if receipt["status"] != "VERIFIED" or not receipt["raw_gt_external_report_parity"]:
        raise ValueError("External raw-GT parity is required")
    for name, expected in receipt["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Diagnostic evidence changed: {name}")
    metrics = {
        name: json.loads(source_bytes(root / f"{name}.json"))
        for name in (
            "candidate",
            "incumbent",
            "oracle_class",
            "oracle_depth",
            "oracle_rank",
            "oracle_size",
        )
    }
    diagnostics = {
        name: json.loads(source_bytes(root / f"{name}_diagnostics.json"))
        for name in ("candidate", "incumbent")
    }
    lines = [
        "# Diagnóstico de predicciones en el panel fijo",
        "",
        f"Fuente: `{receipt['evaluation']}`; SHA-256 `{receipt['evaluation_sha256']}`.",
        "",
        "512 imágenes de selección pública. Las sustituciones oracle usan GT; "
        "no son modelos entrenados ni mejoras obtenidas y no se incorporan a inferencia. "
        "La reevaluación independiente reproduce exactamente las métricas originales.",
        "",
        "| Variante | mAP alineada |",
        "|---|---:|",
    ]
    lines += [f"| {name} | {row['mAP']:.8f} |" for name, row in metrics.items()]
    lines += [
        "",
        "| Clase | GT | AP candidato | AP incumbente | AP oracle clase | AP oracle profundidad |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cls, row in metrics["candidate"]["classes"].items():
        lines.append(
            f"| {cls} | {row['gt_count']} | {row['AP']:.8f} | "
            f"{metrics['incumbent']['classes'][cls]['AP']:.8f} | "
            f"{metrics['oracle_class']['classes'][cls]['AP']:.8f} | "
            f"{metrics['oracle_depth']['classes'][cls]['AP']:.8f} |"
        )
    lines += [
        "",
        "| Profundidad óptica GT | GT | Recall candidato 2 m | Recall incumbente 2 m | "
        "Recall candidato 4 m | Recall incumbente 4 m |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in diagnostics["candidate"]["groups"].items():
        if name.startswith("optical_depth_m/"):
            control = diagnostics["incumbent"]["groups"][name]
            if row["gt"] != control["gt"]:
                raise ValueError("Diagnostic strata differ")
            lines.append(
                f"| {name.split('/', 1)[1]} | {row['gt']} | "
                f"{row['recall']['same_class_greedy_2m']:.6f} | "
                f"{control['recall']['same_class_greedy_2m']:.6f} | "
                f"{row['recall']['same_class_greedy_4m']:.6f} | "
                f"{control['recall']['same_class_greedy_4m']:.6f} |"
            )
    lines += [
        "",
        "El matching se realiza en el frame completo antes de estratificar el GT. "
        "No se presenta AP por rango ni se trata cada frame correlacionado como una "
        "réplica independiente. Las clases con poco soporte requieren confirmación "
        "sobre más ejemplos; los cambios pequeños del panel no demuestran generalización.",
    ]
    args.output.mkdir(parents=True)
    report = args.output / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output / "MANIFEST.json").write_text(
        json.dumps(
            {
                "diagnostic_receipt": (root / "COMPLETE.json").as_posix(),
                "diagnostic_receipt_sha256": digest(root / "COMPLETE.json"),
                "report_sha256": digest(report),
                "generator_sha256": digest(Path(__file__)),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
