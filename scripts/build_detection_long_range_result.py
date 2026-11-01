"""Close the 300m screen from original JSON and a verified CPU-loaded checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def digest(path: Path) -> str:
    """Hash large checkpoints without loading their bytes twice into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Require completed runs and distinguish the panel from full promotion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs: dict[str, str] = {}

    def read(relative: str) -> Any:
        path = args.campaign / relative
        raw = path.read_bytes()
        inputs[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    root300 = "long_range_campaign_v1"
    queue = read(f"{root300}/QUEUE.json")
    state = read(f"{root300}/QUEUE.status.json")
    plan = read("long_range_config_v1/PLAN.json")
    registered_cfg = read("long_range_config_v1/range300.json")
    if (
        state["status"] != "COMPLETE"
        or state["queue_sha256"] != inputs[f"{root300}/QUEUE.json"]
        or queue["plan_sha256"] != inputs["long_range_config_v1/PLAN.json"]
        or plan["screen_config_sha256"] != inputs["long_range_config_v1/range300.json"]
        or (args.campaign / root300 / "QUEUE.lock").exists()
    ):
        raise ValueError("300m queue must be complete and unlocked")
    panel = read("PANEL_COMPARISON.json")
    population = read("screen_config_v1/population.json")
    if not panel["external_raw_gt_exact_report_parity"] or panel["test_labels_read"]:
        raise ValueError("External reference panel parity missing")
    tokens = set(population["roles"]["selection"])
    if len(tokens) != 512 or len(population["roles"]["train"]) != 4096:
        raise ValueError("Unexpected screening population")
    if panel["population_sha256"] != inputs["screen_config_v1/population.json"]:
        raise ValueError("Reference panel uses another population")
    arms = {60: "screen_range60_v1", 150: "screen_range150_v1", 300: f"{root300}/screen"}
    runs: dict[str, Any] = {}
    common_config = None
    initial_predictions = None
    contract300 = None
    for distance, folder in arms.items():
        status = read(f"{folder}/STATUS.json")
        contract = read(f"{folder}/CONTRACT.json")
        cfg = contract["config"]
        if (
            status["status"] != "COMPLETE"
            or status["update"] != cfg["updates"]
            or cfg["updates"] != 2048
            or cfg["depth_max_m"] != distance
            or contract["precision"] != "fp32"
            or contract["augmentation"] != "none"
            or contract["score"] != "class_sigmoid_no_ranker"
            or (args.campaign / folder / "WRITER.lock").exists()
        ):
            raise ValueError(f"Incomplete arm: {distance}")
        if cfg["population_sha256"] != inputs["screen_config_v1/population.json"]:
            raise ValueError("Training population differs")
        shared = {
            key: value
            for key, value in cfg.items()
            if key not in ("schema", "experiment_id", "depth_max_m")
        }
        if common_config is None:
            common_config = shared
        elif shared != common_config:
            raise ValueError("Paired training settings differ")
        for relative, expected in contract["sources"].items():
            path = args.campaign / folder / "code" / relative
            if digest(path) != expected:
                raise ValueError(f"Archived executed source differs: {path}")
        evaluations: dict[int, Any] = {}
        for update in cfg["evaluation_updates"]:
            data = read(f"{folder}/evaluation/{update:06d}.json")
            metrics = data["metrics"]
            if (
                not data["label_free_forward"]
                or data["gt_population"] != "raw_all"
                or set(data["predictions"]) != tokens
                or metrics["profile_sha256"] != panel["profile_sha256"]
                or metrics["samples"] != 512
                or metrics["gt_boxes"] != panel["metrics"]["frozen_detector"]["gt_boxes"]
            ):
                raise ValueError("Evaluation contract or tokens differ")
            if update == 0:
                if initial_predictions is None:
                    initial_predictions = data["predictions"]
                elif data["predictions"] != initial_predictions:
                    raise ValueError("Initial prediction parity failed")
                if metrics != panel["metrics"]["frozen_detector"]:
                    raise ValueError("Initial external metric parity failed")
            evaluations[update] = {"metrics": metrics, "seconds": data["seconds"]}
        best_update = max(evaluations, key=lambda step: evaluations[step]["metrics"]["mAP"])
        best = evaluations[best_update]["metrics"]["mAP"]
        if best != status["best_aligned_mAP"]:
            raise ValueError("Selected metric differs from training status")
        runs[str(distance)] = {
            "status": status,
            "selected_update": best_update,
            "evaluations": evaluations,
            "best": best,
        }
        if distance == 300:
            contract300 = contract
            if cfg != registered_cfg:
                raise ValueError("Executed 300m configuration differs from registration")
            completed = state["completed"]["screen"]
            if (
                completed["returncode"]
                or completed["receipt_sha256"] != inputs[f"{folder}/STATUS.json"]
            ):
                raise ValueError("Queue completion receipt differs")
    checkpoint = args.campaign / root300 / "screen/checkpoints/selected.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    selected = runs["300"]
    if (
        payload["contract"] != contract300
        or payload["update"] != selected["selected_update"]
        or payload["best"] != selected["best"]
        or torch.cuda.is_initialized()
    ):
        raise ValueError("Selected checkpoint metadata differs or CUDA was initialized")
    del payload
    checkpoint_sha = digest(checkpoint)
    incumbent = panel["metrics"]["frozen_ranked"]
    delta_incumbent = selected["best"] - incumbent["mAP"]
    delta_control = selected["best"] - runs["150"]["best"]
    panel_advantage = delta_incumbent > 0 and delta_control > 0
    result = {
        "schema": "phase_ttc_completed_long_range_screen_v1",
        "runs": runs,
        "incumbent_panel": incumbent,
        "delta_incumbent": delta_incumbent,
        "delta_paired_150m_control": delta_control,
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": checkpoint_sha,
        "checkpoint_cpu_load_verified": True,
        "initial_predictions_and_metrics_exact_parity": True,
        "registered_full_promotion_rule": plan["promotion"],
        "decision": "REVIEW_FULL_CONFIRMATION_ALLOCATION"
        if panel_advantage
        else "NO_FULL_CONFIRMATION_NO_PANEL_ADVANTAGE",
        "decision_basis": "Resource allocation after screen, not a full-selection promotion test",
        "full_selection_300m_measured": False,
        "promotion": False,
        "goal_complete": False,
    }
    lines = [
        "# Resultado del screening 60/150/300 m",
        "",
        "Una semilla, 4.096 imágenes TRAIN y 512 de selección ya expuesta. "
        "Misma receta y población; cambia el rango. No es una prueba independiente.",
        "",
        "| Actualización | mAP 60 m | mAP 150 m | mAP 300 m |",
        "|---:|---:|---:|---:|",
    ]
    for step in (0, 512, 1024, 1536, 2048):
        values = [runs[str(r)]["evaluations"][step]["metrics"]["mAP"] for r in arms]
        lines.append(f"| {step} | " + " | ".join(f"{v:.8f}" for v in values) + " |")
    lines += [
        "",
        f"Incumbente con ranking en el mismo panel: **{incumbent['mAP']:.8f}**. "
        f"Mejor 300 m: update {selected['selected_update']}, mAP **{selected['best']:.8f}**. "
        f"Delta incumbente {delta_incumbent:+.8f}; delta control 150 m {delta_control:+.8f}.",
        "",
        "| Clase | GT panel | AP incumbente | AP 150 m seleccionado | AP 300 m seleccionado |",
        "|---|---:|---:|---:|---:|",
    ]
    for cls, row in incumbent["classes"].items():
        ap150 = runs["150"]["evaluations"][runs["150"]["selected_update"]]["metrics"]["classes"][
            cls
        ]["AP"]
        ap300 = selected["evaluations"][selected["selected_update"]]["metrics"]["classes"][cls][
            "AP"
        ]
        lines.append(f"| {cls} | {row['gt_count']} | {row['AP']:.8f} | {ap150:.8f} | {ap300:.8f} |")
    lines += [
        "",
        "Las clases sin GT mantienen AP=0 en la macro. Este panel no confirma su calidad.",
        "",
        "Decisión: "
        + (
            "revisar si se asigna una confirmación completa; no hay promoción automática."
            if panel_advantage
            else "conservar el incumbente y no asignar confirmación completa a este 300 m, "
            "al no superar en el panel al incumbente y al control pareado."
        ),
        "Esta es una decisión de recursos posterior al screening. El plan exige para "
        "promoción una ganancia completa de al menos 0,005 frente al incumbente y un "
        "control válido, con revisión de caídas por clase >0,01; esa prueba completa "
        "no se ha ejecutado para 300 m. No se descarta toda receta de mayor rango.",
        "",
        "El ensayo KITTI registrado continúa por separado. Igualar nuevas exposiciones "
        "no iguala el historial previo de aprendizaje de los dos modelos.",
        "",
        f"Checkpoint conservado: `{checkpoint.as_posix()}`. SHA-256 `{checkpoint_sha}`.",
        "Se verificaron carga CPU, update seleccionado, contrato, fuentes archivadas, "
        "paridad inicial de predicciones y métricas, y recibo de finalización de la cola.",
        "",
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "RESULTS.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "inputs": inputs,
        "generator_sha256": digest(Path(__file__)),
        "files": {name: digest(args.output / name) for name in ("RESULTS.json", "REPORT.md")},
    }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("runs", "incumbent_panel")}))


if __name__ == "__main__":
    main()
