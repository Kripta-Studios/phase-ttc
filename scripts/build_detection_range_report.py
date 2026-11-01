"""Snapshot range-campaign evidence and regenerate tables from original JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def main() -> None:
    """Keep live work explicitly provisional and bind every snapshot to source bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for source, target in (
        ("SMOKE_AUDIT.json", "SMOKE_AUDIT.json"),
        ("screen_config_v1/ADMISSION.json", "ADMISSION.json"),
        ("screen_config_v1/range60.json", "CONFIG_RANGE60.json"),
        ("screen_config_v1/range150.json", "CONFIG_RANGE150.json"),
        ("screen_config_v1/QUEUE.json", "QUEUE.json"),
        ("RAW_PANEL_PARITY.json", "RAW_PANEL_PARITY.json"),
    ):
        path = args.campaign / source
        inputs[source] = digest(path)
        shutil.copy2(path, args.output / target)
    identity_path = args.campaign / "TRACK_IDENTITY_v2.json"
    identity = json.loads(identity_path.read_text())
    inputs[identity_path.name] = digest(identity_path)
    write_json(
        args.output / "TRACK_IDENTITY_SUMMARY.json",
        {k: v for k, v in identity.items() if k != "frames_per_instance"},
    )
    runs = {}
    for name in ("screen_range60_v1", "screen_range150_v1"):
        folder = args.campaign / name
        status = folder / "STATUS.json"
        if not status.exists():
            runs[name] = {"status": {"status": "NOT_STARTED"}, "evaluations": {}}
            continue
        status_bytes = status.read_bytes()
        inputs[f"{name}/STATUS.json"] = hashlib.sha256(status_bytes).hexdigest()
        row = {"status": json.loads(status_bytes), "evaluations": {}}
        for path in sorted((folder / "evaluation").glob("*.json")):
            value = json.loads(path.read_text())
            inputs[f"{name}/evaluation/{path.name}"] = digest(path)
            row["evaluations"][str(int(path.stem))] = {
                "metrics": value["metrics"],
                "seconds": value["seconds"],
                "label_free_forward": value["label_free_forward"],
                "gt_population": value["gt_population"],
            }
        runs[name] = row
    write_json(args.output / "SCREENING_RESULTS.json", runs)
    panel_path = args.campaign / "PANEL_COMPARISON.json"
    if panel_path.exists():
        inputs[panel_path.name] = digest(panel_path)
        shutil.copy2(panel_path, args.output / panel_path.name)
    census_path = args.campaign / "train_census_v1/CENSUS.json"
    census = None
    if census_path.exists():
        census = json.loads(census_path.read_text())
        inputs["train_census_v1/CENSUS.json"] = digest(census_path)
        shutil.copy2(census_path, args.output / "CENSUS.json")
    lines = [
        "# Evidencia del ensayo de rango",
        "",
        "Generado desde JSON original. Screening de una semilla en selección ya expuesta; "
        "ningún resultado de esta tabla implica promoción o excelencia.",
        "",
        "## Panel de selección",
        "",
        "| Brazo | Estado del run | Update evaluado | mAP raw | Segundos de validación |",
        "|---|---|---:|---:|---:|",
    ]
    for name, run in runs.items():
        for update, result in run["evaluations"].items():
            lines.append(
                f"| {name} | {run['status']['status']} | {update} | "
                f"{result['metrics']['mAP']:.8f} | {result['seconds']:.2f} |"
            )
        if not run["evaluations"]:
            lines.append(f"| {name} | {run['status']['status']} | — | — | — |")
    first = runs["screen_range60_v1"]["evaluations"].get("0")
    if first is not None:
        lines += [
            "",
            "## Soporte del panel de selección",
            "",
            "| Clase | Anotaciones raw |",
            "|---|---:|",
        ]
        for cls, row in first["metrics"]["classes"].items():
            lines.append(f"| {cls} | {row['gt_count']} |")
        lines += [
            "",
            "Las clases sin GT conservan AP=0 en la macro de ocho clases. "
            "Este panel no permite confirmar su calidad; se exige selección completa.",
        ]
    lines += [
        "",
        "## Soporte raw TRAIN",
        "",
        "Los identificadores se cuentan dentro de secuencia. "
        "No certifican adquisiciones independientes.",
        "",
        "| Clase | Anotaciones | Instancias por secuencia | Secuencias |",
        "|---|---:|---:|---:|",
    ]
    for cls, count in sorted(identity["raw_objects"].items()):
        lines.append(
            f"| {cls} | {count} | {identity['sequence_instance_counts'][cls]} | "
            f"{len(identity['sequence_support'][cls])} |"
        )
    lines += [
        "",
        f"Unión clase/centro: {identity['status']}; colisiones: "
        f"{len(identity['class_center_join_collisions'])}.",
    ]
    if census is not None:
        if census["counts"]["raw"] != identity["raw_objects"]:
            raise ValueError("Independent raw census counts differ")
        lines += [
            "",
            "## Admisión de targets TRAIN",
            "",
            "| Clase | Raw | Admitidos 60 m | Admitidos 150 m |",
            "|---|---:|---:|---:|",
        ]
        for cls in sorted(identity["raw_objects"]):
            lines.append(
                f"| {cls} | {census['counts']['raw'].get(cls, 0)} | "
                f"{census['counts']['admitted60'].get(cls, 0)} | "
                f"{census['counts']['admitted150'].get(cls, 0)} |"
            )
    else:
        lines += ["", "Censo de admisión geométrica completa: pendiente."]
    lines += [
        "",
        "## Reproducción",
        "",
        "```powershell",
        "python scripts/build_detection_range_report.py `",
        f"  --campaign {args.campaign.as_posix()} `",
        f"  --output {args.output.as_posix()}",
        "```",
        "",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        args.output / "SNAPSHOT_RECEIPT.json",
        {
            "source_root": str(args.campaign),
            "source_sha256": inputs,
            "report_builder_sha256": digest(Path(__file__)),
            "source_population_sha256": digest(args.campaign / "screen_config_v1/population.json"),
            "files": {
                p.name: digest(p)
                for p in args.output.iterdir()
                if p.is_file() and p.name != "SNAPSHOT_RECEIPT.json"
            },
            "promotion": False,
        },
    )


if __name__ == "__main__":
    main()
