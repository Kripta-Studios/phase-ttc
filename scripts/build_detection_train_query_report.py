"""Regenerate the TRAIN diagnostic report with class-by-depth comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from diagnose_detection_train_queries import render_report  # noqa: E402

from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402


def main() -> None:
    """Keep the original report intact and bind this expanded interpretation to its data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = json.loads((args.diagnostics / "COMPLETE.json").read_text())
    if receipt["status"] != "VERIFIED" or receipt["frames"] != 4096:
        raise ValueError("Complete TRAIN diagnostic required")
    for name, expected in receipt["files"].items():
        if digest(args.diagnostics / name) != expected:
            raise ValueError("Diagnostic results changed")
    intake_path = args.diagnostics / "INTAKE.json"
    if digest(intake_path) != receipt["intake_sha256"]:
        raise ValueError("Diagnostic intake changed")
    intake = json.loads(intake_path.read_text())
    source = ROOT / "scripts/diagnose_detection_train_queries.py"
    if digest(source) != intake["sources"][source.relative_to(ROOT).as_posix()]:
        raise ValueError("Base report source changed")
    data = json.loads((args.diagnostics / "COMPARISON.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    census = {"sequence_instance_id_counts": data["train_all_sequence_instance_id_counts"]}
    render_report(args.output, data["train"], data["selection"], census)
    path = args.output / "REPORT.md"
    lines = [
        path.read_text(encoding="utf-8"),
        "## Comparación dentro de tramos de profundidad óptica",
        "",
        "El matching se hizo sobre el frame completo y después se agruparon los GT. "
        "Compartir tramo de profundidad no iguala identidad, tamaño, visibilidad o "
        "escena. No se atribuye causalmente la diferencia al sobreajuste.",
        "",
        "| Clase | Profundidad | GT TRAIN | Correcta / agnóstica TRAIN, 2 m | "
        "GT selección | Correcta / agnóstica selección, 2 m |",
        "|---|---|---:|---:|---:|---:|",
    ]
    strata = {}
    for cls in ("bicycle", "pedestrian", "other_vehicle"):
        for depth in ("[0,20)", "[20,40)", "[40,60)", "[60,100)", "[100,150)", ">=150"):
            key = f"class_depth/{cls}/{depth}"
            cells = []
            strata[key] = {}
            for role in ("train", "selection"):
                group = data[role]["groups"].get(key)
                strata[key][role] = group
                if group is None:
                    cells.extend(["0", "n/a"])
                else:
                    correct = group["recall"]["same_class_maximum_2m"]
                    agnostic = group["recall"]["agnostic_maximum_2m"]
                    cells.extend([str(group["gt"]), f"{correct:.6f} / {agnostic:.6f}"])
            lines.append(f"| {cls} | {depth} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Decisión: priorizar la comparación de inicialización externa ya registrada "
        "y terminar el ensayo de rango con su presupuesto fijado. No repetir R2-rare "
        "ni los rankers de logits como si fueran intervenciones nuevas. La buena "
        "clasificación TRAIN de parte de los objetos raros contradice un fallo "
        "absoluto para aprender esas etiquetas; no garantiza que otra cabeza entrenada "
        "sobre las mismas identidades generalice. El problema de other_vehicle incluye "
        "poca diversidad admitida originalmente y baja cobertura geométrica raw.",
        "",
        "No se promociona un modelo ni se añaden semillas o ensayos GPU a partir "
        "de este diagnóstico. Los resultados pendientes decidirán la siguiente "
        "intervención sobre representación, datos o arquitectura.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    write_json(args.output / "CLASS_DEPTH.json", strata)
    write_json(
        args.output / "DECISION.json",
        {
            "decision": "PRIORITIZE_REGISTERED_PRETRAINED_COMPARISON",
            "new_gpu_experiment_started": False,
            "range300_and_kitti_continue_as_registered": True,
            "repeat_rare_sampling": False,
            "causal_generalization_gap_claim": False,
            "promotion": False,
            "goal_complete": False,
        },
    )
    write_json(
        args.output / "MANIFEST.json",
        {
            "diagnostic_complete_sha256": digest(args.diagnostics / "COMPLETE.json"),
            "comparison_sha256": digest(args.diagnostics / "COMPARISON.json"),
            "generator_sha256": digest(Path(__file__)),
            "base_report_source_sha256": digest(source),
            "files": {
                name: digest(args.output / name)
                for name in ("REPORT.md", "CLASS_DEPTH.json", "DECISION.json")
            },
        },
    )


if __name__ == "__main__":
    main()
