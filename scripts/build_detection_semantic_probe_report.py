"""Regenerate the small TRAIN semantic probe report from its retained evidence."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def main() -> None:
    """Preserve the executed source and distinguish this probe from benchmark quality."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--visuals", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = json.loads((args.probe / "RESULTS.json").read_text())
    plan = json.loads((args.probe / "PLAN.json").read_text())
    model = json.loads((args.probe / "MODEL_INTAKE.json").read_text())
    visual = json.loads((args.visuals / "VISUALS.json").read_text())
    source = args.probe / "code/probe_detection_semantics_cpu.py"
    if (
        result["status"] != "COMPLETE_CPU_PROBE"
        or result["cuda_initialized"]
        or result["plan_sha256"] != digest(args.probe / "PLAN.json")
        or result["model_intake_sha256"] != digest(args.probe / "MODEL_INTAKE.json")
        or plan["source_sha256"] != digest(source)
        or visual["results_sha256"] != digest(args.probe / "RESULTS.json")
        or visual["image_sha256"] != digest(args.visuals / "bicycle_crops.png")
    ):
        raise ValueError("Probe/visual/source evidence changed")
    if ast.dump(ast.parse(source.read_text(encoding="utf-8"))) != ast.dump(
        ast.parse((ROOT / "scripts/probe_detection_semantics_cpu.py").read_text(encoding="utf-8"))
    ):
        raise ValueError("Current probe is not syntax-equivalent to executed source")
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for name in ("PLAN.json", "RESULTS.json", "MODEL_INTAKE.json", "SOURCE_FORMAT_RECEIPT.json"):
        path = args.probe / name
        inputs[str(path)] = digest(path)
        shutil.copy2(path, args.output / name)
    shutil.copy2(source, args.output / "executed_probe_v1.py")
    shutil.copy2(args.visuals / "VISUALS.json", args.output / "VISUALS.json")
    for name, expected in result["predictions"].items():
        path = args.probe / "predictions" / name
        if digest(path) != expected:
            raise ValueError("A retained prediction changed")
        destination = args.output / "predictions" / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(path, destination)
        inputs[str(path)] = expected
    rows = result["rows"]
    lines = [
        "# Comprobación semántica RGB en TRAIN",
        "",
        "Se probó Faster R-CNN ResNet50-FPN v2 con sus pesos COCO_V1 públicos de "
        "[torchvision 0.26](https://docs.pytorch.org/vision/0.26/models/generated/"
        "torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html). "
        "No se entrenó ni se evaluó en selección o test. El modelo recibió únicamente RGB.",
        "",
        f"Población fijada antes de inferir: {len(plan['positive_tokens'])} imágenes con "
        f"bicicleta y {len(plan['negative_tokens'])} sin anotación de bicicleta. "
        f"Umbrales: score ≥ {plan['confidence']} e IoU ≥ {plan['projected_cuboid_iou']} "
        "contra el rectángulo del cuboide 3D proyectado.",
        "",
        f"Coincidencias: **{result['matched_bicycle_targets']}/"
        f"{result['projectable_bicycle_targets']}** targets proyectables. "
        "Detecciones bicycle sobre los frames negativos: "
        f"**{result['bicycle_detections_on_negative_frames']}**. "
        "No son mAP eAP, recall general ni precisión de un detector 3D.",
        "",
        "| Token TRAIN | GT proyectable | Bicycle detectadas | Coincidencias | Segundos CPU |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['token']} | {row['projectable_bicycle_gt']} | "
            f"{row['bicycle_detections']} | {row['matches']} | "
            f"{row['inference_seconds']:.3f} |"
        )
    lines += [
        "",
        "La inspección de los recortes encontró árboles/vallas y regiones oscuras "
        "en varios ejemplos. En otro, la caja bicycle visible es menor que el cuboide "
        "proyectado con el ciclista. Son observaciones cualitativas de estos ejemplos, "
        "no etiquetas de visibilidad ni un censo del dataset.",
        "",
        "Decisión: no ampliar todavía esta receta a inferencia GPU completa. "
        "El resultado no justifica el coste y la comparación entre cajas visibles y "
        "cuboides amodales requiere cautela. No se cambian los umbrales del ensayo "
        "ni se interpreta el diagnóstico como descarte de todo preentrenamiento semántico.",
        "",
        "COCO no aporta clases nativas tricycle u other_vehicle. No se certificó "
        "admisión de estos pesos en el benchmark ni su redistribución; el trabajo "
        "es local. La licencia del código y las identidades de pesos/versiones están "
        "en MODEL_INTAKE.json. No hay promoción ni envío oficial.",
        "",
        f"SHA-256 de pesos: `{model['weights_sha256']}`. "
        f"Recortes diagnósticos locales: `{(args.visuals / 'bicycle_crops.png').as_posix()}`.",
        "",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        args.output / "REPORT_RECEIPT.json",
        {
            "inputs": inputs,
            "visual_metadata_sha256": digest(args.visuals / "VISUALS.json"),
            "report_sha256": digest(args.output / "REPORT.md"),
            "source_sha256": digest(Path(__file__)),
            "decision": "DO_NOT_EXPAND_THIS_PROBE_YET",
            "mAP_measured": False,
        },
    )


if __name__ == "__main__":
    main()
