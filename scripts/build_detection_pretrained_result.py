"""Close the registered KITTI initialization screen against matched eAP controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def digest(path: Path) -> str:
    """Hash artifacts without buffering whole checkpoints."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summary(metrics: dict) -> dict:
    """Keep report quantities; original PR arrays remain in byte-bound evaluation JSON."""
    return {
        "mAP": metrics["mAP"],
        "samples": metrics["samples"],
        "gt_boxes": metrics["gt_boxes"],
        "profile_sha256": metrics["profile_sha256"],
        "classes": {
            name: {
                **{key: value[key] for key in ("AP", "gt_count", "pred_count")},
                "thresholds": {
                    threshold: {
                        **{key: curve[key] for key in ("AP", "TP", "FP", "recall")},
                        "maximum_precision_used_by_primary_AP": max(curve["precision"][11:]),
                    }
                    for threshold, curve in value["thresholds"].items()
                },
            }
            for name, value in metrics["classes"].items()
        },
    }


def main() -> None:
    """Require completion and verify the selected checkpoint before any next-GPU review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-root", type=Path, required=True)
    parser.add_argument("--range-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs: dict[str, str] = {}

    def read(path: Path) -> Any:
        inputs[path.as_posix()] = digest(path)
        return json.loads(path.read_text(encoding="utf-8"))

    root = args.pretrained_root / "comparison_campaign_v1"
    queue_path = root / "QUEUE.json"
    queue, state = read(queue_path), read(root / "QUEUE.status.json")
    plan_path = args.pretrained_root / "comparison_config_v1/PLAN.json"
    plan = read(plan_path)
    cfg_path = args.pretrained_root / "comparison_config_v1/screen.json"
    registered = read(cfg_path)
    if (
        state["status"] != "COMPLETE"
        or (root / "QUEUE.lock").exists()
        or (root / "screen/WRITER.lock").exists()
        or state["queue_sha256"] != inputs[queue_path.as_posix()]
        or queue["plan_sha256"] != inputs[plan_path.as_posix()]
        or plan["screen_config_sha256"] != inputs[cfg_path.as_posix()]
    ):
        raise ValueError("KITTI queue must complete with its registered contract")
    for stage, relative in (
        ("smoke", "smoke/STATUS.json"),
        ("smoke_gate", "SMOKE_GATE.json"),
        ("screen", "screen/STATUS.json"),
    ):
        read(root / relative)
        if (
            state["completed"][stage]["returncode"]
            or state["completed"][stage]["receipt_sha256"] != inputs[(root / relative).as_posix()]
        ):
            raise ValueError("Completion receipt changed")
    panel = read(args.range_root / "PANEL_COMPARISON.json")
    population = read(Path(registered["population"]))
    tokens = set(population["roles"]["selection"])
    if (
        len(tokens) != 512
        or len(population["roles"]["train"]) != 4096
        or panel["population_sha256"] != registered["population_sha256"]
        or registered["population_sha256"] != inputs[Path(registered["population"]).as_posix()]
    ):
        raise ValueError("Panel populations differ")
    if not panel["external_raw_gt_exact_report_parity"] or panel["test_labels_read"]:
        raise ValueError("Aligned external reference parity is missing")
    roots = {
        "dfull300": args.range_root / "long_range_campaign_v1/screen",
        "kitti": root / "screen",
    }
    runs: dict[str, Any] = {}
    contracts: dict[str, dict] = {}
    settings = None
    ignore = {
        "schema",
        "experiment_id",
        "initialization",
        "pretrained_weights",
        "pretrained_weights_sha256",
        "conversion_receipt",
        "conversion_receipt_sha256",
    }
    for name, folder in roots.items():
        status = read(folder / "STATUS.json")
        contract = read(folder / "CONTRACT.json")
        cfg = contract["config"]
        contracts[name] = contract
        if (
            status["status"] != "COMPLETE"
            or status["update"] != 2048
            or cfg["updates"] != 2048
            or cfg["depth_max_m"] != 300
            or contract["precision"] != "fp32"
            or contract["augmentation"] != "none"
            or contract["score"] != "class_sigmoid_no_ranker"
        ):
            raise ValueError("Incomplete or incompatible arm")
        shared = {key: value for key, value in cfg.items() if key not in ignore}
        if settings is None:
            settings = shared
        elif shared != settings:
            raise ValueError("Additional training recipes differ beyond initialization")
        if name == "kitti" and cfg != registered:
            raise ValueError("Executed KITTI config differs from registration")
        for relative, expected in contract["sources"].items():
            if digest(folder / "code" / relative) != expected:
                raise ValueError(f"Archived executed source changed: {relative}")
        evaluations = {}
        for update in cfg["evaluation_updates"]:
            result = read(folder / f"evaluation/{update:06d}.json")
            metric = result["metrics"]
            if (
                result["label_free_forward"] is not True
                or result["gt_population"] != "raw_all"
                or set(result["predictions"]) != tokens
                or metric["samples"] != 512
                or metric["profile_sha256"] != panel["profile_sha256"]
                or metric["gt_boxes"] != panel["metrics"]["frozen_ranked"]["gt_boxes"]
            ):
                raise ValueError("Evaluation populations or profile differ")
            evaluations[update] = {"metrics": summary(metric), "seconds": result["seconds"]}
        selected = max(evaluations, key=lambda step: evaluations[step]["metrics"]["mAP"])
        best = evaluations[selected]["metrics"]["mAP"]
        if status["best_aligned_mAP"] != best:
            raise ValueError("Selected metric does not match training status")
        runs[name] = {
            "selected_update": selected,
            "best": best,
            "evaluations": evaluations,
            "status": status,
        }
    checkpoint = root / "screen/checkpoints/selected.pt"
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    selected = runs["kitti"]
    if (
        saved["contract"] != contracts["kitti"]
        or saved["update"] != selected["selected_update"]
        or saved["best"] != selected["best"]
        or torch.cuda.is_initialized()
    ):
        raise ValueError("Selected checkpoint lineage differs or CUDA was initialized")
    if not all(torch.isfinite(value).all() for value in saved["model"].values()):
        raise ValueError("Nonfinite selected checkpoint")
    incumbent = summary(panel["metrics"]["frozen_ranked"])
    delta_incumbent = selected["best"] - incumbent["mAP"]
    delta_control = selected["best"] - runs["dfull300"]["best"]
    gate = delta_incumbent >= 0.005 and delta_control >= 0.005
    result = {
        "schema": "phase_completed_pretrained_screen_v1",
        "runs": runs,
        "incumbent_panel": incumbent,
        "delta_incumbent": delta_incumbent,
        "delta_matched_control": delta_control,
        "registered_promotion_rule": plan["promotion"],
        "panel_gate_passed": gate,
        "decision": "REVIEW_FULL_CONFIRMATION"
        if gate
        else "NO_FULL_CONFIRMATION_PANEL_GATE_NOT_MET",
        "selected_checkpoint": checkpoint.as_posix(),
        "selected_checkpoint_sha256": digest(checkpoint),
        "checkpoint_cpu_load_and_finite_verified": True,
        "full_selection_measured": False,
        "promotion": False,
        "goal_complete": False,
    }
    lines = [
        "# Resultado del screening de inicialización KITTI",
        "",
        "Una semilla; mismo presupuesto adicional de 4.096 exposiciones eAP, "
        "4.096 imágenes TRAIN y 512 de selección. No se iguala el historial "
        "anterior: Dfull ya vio 281.116 exposiciones eAP y el origen KITTI "
        "se entrenó para Car. Este resultado no descarta toda transferencia "
        "externa ni prueba convergencia con otra receta.",
        "",
        "| Update | Dfull 300 m | KITTI |",
        "|---:|---:|---:|",
    ]
    for step in (0, 512, 1024, 1536, 2048):
        lines.append(
            f"| {step} | {runs['dfull300']['evaluations'][step]['metrics']['mAP']:.8f} | "
            f"{runs['kitti']['evaluations'][step]['metrics']['mAP']:.8f} |"
        )
    lines += [
        "",
        f"Incumbente con ranking: **{incumbent['mAP']:.8f}**. Mejor KITTI: "
        f"**{selected['best']:.8f}**, update {selected['selected_update']}. "
        f"Delta frente al incumbente {delta_incumbent:+.8f}; frente al control "
        f"{delta_control:+.8f}.",
        "",
        "| Clase | GT panel | AP incumbente | AP control | AP KITTI |",
        "|---|---:|---:|---:|---:|",
    ]
    control_metric = runs["dfull300"]["evaluations"][runs["dfull300"]["selected_update"]]["metrics"]
    selected_metric = selected["evaluations"][selected["selected_update"]]["metrics"]
    for cls, row in incumbent["classes"].items():
        lines.append(
            f"| {cls} | {row['gt_count']} | {row['AP']:.8f} | "
            f"{control_metric['classes'][cls]['AP']:.8f} | "
            f"{selected_metric['classes'][cls]['AP']:.8f} |"
        )
    lines += [
        "",
        "| Clase KITTI seleccionado | Predicciones | TP a 2 m | FP a 2 m | "
        "Recall 2 m | Recall 4 m |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cls, row in selected_metric["classes"].items():
        two, four = row["thresholds"]["2.0"], row["thresholds"]["4.0"]
        recall_two = "n/a" if two["recall"] is None else f"{two['recall']:.4f}"
        recall_four = "n/a" if four["recall"] is None else f"{four['recall']:.4f}"
        lines.append(
            f"| {cls} | {row['pred_count']} | {two['TP']} | {two['FP']} | "
            f"{recall_two} | {recall_four} |"
        )
    lines += [
        "",
        "AP cero no implica necesariamente cero aciertos. El perfil alineado "
        "usa puntos de recall superiores a 0,1, resta 0,1 a la precisión "
        "interpolada y recorta resultados negativos a cero. El JSON conserva "
        "TP, FP, recall y la precisión máxima utilizada por AP en cada umbral. "
        "Así se distingue ausencia de propuestas útiles de una curva con "
        "precisión insuficiente. No se cambia la métrica ni el ranking del ensayo.",
        "",
        "Las clases sin GT conservan AP=0 en la macro. El panel ya está "
        "expuesto; no constituye test independiente.",
        "",
        f"Decisión: **{result['decision']}**. "
        "El criterio previo exige +0,005 frente a ambas referencias para "
        "considerar la confirmación completa. No se ejecuta selección completa "
        "ni se promociona este checkpoint mediante este informe.",
        "",
        "Se verificaron fuentes ejecutadas, recibos de finalización, "
        "receta, población, perfil, carga CPU, finitud y selección del checkpoint.",
        "",
        f"Checkpoint conservado: `{checkpoint.as_posix()}`; SHA-256 "
        f"`{result['selected_checkpoint_sha256']}`.",
        "",
        "Próxima revisión: valorar el smoke técnico de MonoCoP multiclase "
        "ya preparado, con otro contrato y presupuesto local de 600 s. "
        "Este informe no inicia ni admite automáticamente esa ejecución.",
        "",
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "input_sha256": inputs,
        "generator_sha256": digest(Path(__file__)),
        "files": {name: digest(args.output / name) for name in ("RESULTS.json", "REPORT.md")},
    }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
