"""Close a completed MonoCoP screen from raw evaluations and its selected state."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_detection_pretrained_result import summary

from e_jepa_ttc.detection.monocop_campaign import MonoCoPScreenConfig
from e_jepa_ttc.detection.range_campaign import digest, write_json


def main() -> None:
    """Bind all source data; report the registered gate without promoting a candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Require a new report destination")
    inputs: dict[str, str] = {}

    def read(path: Path) -> Any:
        inputs[path.as_posix()] = digest(path)
        return json.loads(path.read_text())

    plan = read(args.plan)
    cfg = MonoCoPScreenConfig.load(args.plan.parent / "screen.json")
    read(args.plan.parent / "screen.json")
    status, contract = read(args.run / "STATUS.json"), read(args.run / "CONTRACT.json")
    if (
        status["status"] != "COMPLETE"
        or status["update"] != cfg.updates
        or (args.run / "WRITER.lock").exists()
        or (args.plan.parent / "SCREEN.lock").exists()
        or contract["plan_sha256"] != inputs[args.plan.as_posix()]
        or contract["config_sha256"] != cfg.sha256
        or plan["config_sha256"] != cfg.sha256
        or plan["config_file_sha256"] != inputs[(args.plan.parent / "screen.json").as_posix()]
        or contract["model_forward_targets"] is not None
        or contract["score"] != "class_sigmoid_no_ranker"
    ):
        raise ValueError("Incomplete or incompatible screen")
    for name, expected in contract["sources"].items():
        if digest(args.run / "code" / name) != expected:
            raise ValueError(f"Executed source archive changed: {name}")
    controls_path = Path(plan["controls"]["report"])
    controls = read(controls_path)
    if inputs[controls_path.as_posix()] != plan["controls"]["report_sha256"]:
        raise ValueError("Control report changed")
    population = read(Path(cfg.population))
    if inputs[Path(cfg.population).as_posix()] != cfg.population_sha256:
        raise ValueError("Population changed")
    incumbent = controls["incumbent_panel"]
    tokens = set(population["roles"]["selection"])
    evaluations = {}
    for update in cfg.evaluation_updates:
        result = read(args.run / f"evaluation/{update:06d}.json")
        metric = result["metrics"]
        if (
            result["label_free_forward"] is not True
            or result["gt_population"] != "raw_all"
            or set(result["predictions"]) != tokens
            or metric["samples"] != 512
            or metric["profile_sha256"] != incumbent["profile_sha256"]
            or {k: v["gt_count"] for k, v in metric["classes"].items()}
            != {k: v["gt_count"] for k, v in incumbent["classes"].items()}
        ):
            raise ValueError("Evaluation population or metric profile differs")
        evaluations[update] = {"metrics": summary(metric), "seconds": result["seconds"]}
    selected_update = max(evaluations, key=lambda step: evaluations[step]["metrics"]["mAP"])
    selected_metric = evaluations[selected_update]["metrics"]
    best = selected_metric["mAP"]
    if best != status["best_aligned_mAP"] or selected_update != status["selected_update"]:
        raise ValueError("Selected update differs from earliest maximum")
    checkpoint = args.run / "checkpoints/selected.pt"
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if saved["contract"] != contract or saved["update"] != selected_update or saved["best"] != best:
        raise ValueError("Selected checkpoint lineage differs")
    if torch.cuda.is_initialized() or not all(
        torch.isfinite(v).all() for v in saved["model"].values()
    ):
        raise ValueError("Expected finite checkpoint verified on CPU only")
    delta_inc = best - incumbent["mAP"]
    delta_control = best - controls["runs"]["dfull300"]["best"]
    gate = delta_inc >= 0.005 and delta_control >= 0.005
    history_path = args.run / "metrics.jsonl"
    inputs[history_path.as_posix()] = digest(history_path)
    history = [json.loads(line) for line in history_path.read_text().splitlines()]
    if [row["update"] for row in history] != list(range(1, cfg.updates + 1)):
        raise ValueError("Training history is incomplete")
    timing_fields = [key for key in history[0] if key.endswith("_seconds")]
    means = {key: sum(row[key] for row in history) / len(history) for key in timing_fields}
    recovery = None
    if args.recovery is not None:
        recovery = read(args.recovery / "RECOVERY.json")
        for name, expected in recovery["files"].items():
            if digest(args.recovery / name) != expected:
                raise ValueError(f"Recovered predecessor changed: {name}")
        previous = [
            json.loads(line) for line in (args.recovery / "metrics.jsonl").read_text().splitlines()
        ]
        resumed_at = recovery["restored_update"]
        if history[:resumed_at] != previous[:resumed_at]:
            raise ValueError("Restored history differs from its predecessor")
        replayed = previous[resumed_at:]
        if any(
            a["sample_tokens"] != b["sample_tokens"]
            for a, b in zip(replayed, history[resumed_at:], strict=False)
        ):
            raise ValueError("Resumed data ordering differs")
        recovery["replayed_losses_bitwise_equal"] = all(
            a["loss"] == b["loss"] for a, b in zip(replayed, history[resumed_at:], strict=False)
        )
        recovery["logical_exposures"] = cfg.updates * contract["technical_config"]["batch_size"]
        recovery["minimum_executed_exposures_including_lost_work"] = (
            recovery["logical_exposures"] + recovery["replayed_logged_exposures"]
        )
    result = {
        "schema": "phase_completed_monocop_screen_v1",
        "status": status,
        "evaluations": evaluations,
        "selected_update": selected_update,
        "best": best,
        "incumbent_panel": incumbent,
        "control_df300": controls["runs"]["dfull300"],
        "delta_incumbent": delta_inc,
        "delta_control": delta_control,
        "registered_rule": plan["promotion"],
        "panel_gate_passed": gate,
        "decision": "REVIEW_FULL_CONFIRMATION"
        if gate
        else "NO_FULL_CONFIRMATION_PANEL_GATE_NOT_MET",
        "class_drops_gt_0_01_vs_incumbent": [
            k
            for k, v in selected_metric["classes"].items()
            if v["AP"] < incumbent["classes"][k]["AP"] - 0.01
        ],
        "selected_checkpoint": checkpoint.as_posix(),
        "selected_checkpoint_sha256": digest(checkpoint),
        "checkpoint_cpu_load_and_finite_verified": True,
        "mean_stage_seconds": means,
        "full_selection_measured": False,
        "promotion": False,
        "goal_complete": False,
        "host_restart_recovery": recovery,
    }
    lines = [
        "# Resultado del primer screening MonoCoP",
        "",
        "Una semilla; 4.096 imágenes TRAIN, 4.096 exposiciones adicionales y 512 de selección. "
        "Cambian arquitectura y procedencia de pesos respecto a MonoDGP; no se iguala su "
        "entrenamiento previo ni se atribuye causalidad a un componente aislado. "
        "La selección pública ya fue utilizada en desarrollo. No es score oficial.",
        "",
        "| Update | MonoCoP | Dfull 300 m | KITTI MonoDGP |",
        "|---:|---:|---:|---:|",
    ]
    for step, row in evaluations.items():
        lines.append(
            f"| {step} | {row['metrics']['mAP']:.8f} | "
            f"{controls['runs']['dfull300']['evaluations'][str(step)]['metrics']['mAP']:.8f} | "
            f"{controls['runs']['kitti']['evaluations'][str(step)]['metrics']['mAP']:.8f} |"
        )
    lines += [
        "",
        f"Mejor MonoCoP: **{best:.8f}**, update {selected_update}. "
        f"Incumbente en el mismo panel: **{incumbent['mAP']:.8f}**. "
        f"Delta incumbente {delta_inc:+.8f}; delta control {delta_control:+.8f}.",
        "",
        "| Clase | GT | AP incumbente | AP MonoCoP | TP 2 m | FP 2 m | Recall 2 m |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in selected_metric["classes"].items():
        curve = row["thresholds"]["2.0"]
        recall = "n/a" if curve["recall"] is None else f"{curve['recall']:.4f}"
        lines.append(
            f"| {name} | {row['gt_count']} | {incumbent['classes'][name]['AP']:.8f} | "
            f"{row['AP']:.8f} | {curve['TP']} | {curve['FP']} | {recall} |"
        )
    lines += [
        "",
        f"Decisión registrada: **{result['decision']}**. "
        "No se ha medido selección completa ni se promociona este candidato.",
        "",
        f"Tiempo comunicado por el productor: {status['elapsed_seconds']:.2f} s; "
        f"peak CUDA asignado: {status['peak_cuda_bytes'] / 2**30:.3f} GiB.",
        "",
        "| Etapa | Media por actualización (s) |",
        "|---|---:|",
    ]
    lines += [f"| {name} | {value:.6f} |" for name, value in means.items()]
    lines += [
        "",
        "Los tiempos por actualización no incluyen validación ni checkpoints; "
        "el tiempo de ejecución sí los incluye. Un resultado negativo con este presupuesto "
        "no prueba convergencia ni descarta recetas más largas. No se añaden épocas "
        "o semillas automáticamente.",
        "",
        f"Checkpoint SHA-256: `{result['selected_checkpoint_sha256']}`.",
    ]
    if recovery is not None:
        lines += [
            "",
            "Hubo un reinicio del host. Se preservó el STATUS dañado y se verificaron "
            "en CPU el modelo y el optimizador de ambos checkpoints recuperados. "
            f"Reanudación desde {recovery['restored_update']}; "
            f"{recovery['replayed_logged_updates']} updates registrados se perdieron "
            "y se repitieron con los mismos tokens. "
            f"Losses repetidas bit a bit: {recovery['replayed_losses_bitwise_equal']}.",
            "",
            f"Exposiciones lógicas: {recovery['logical_exposures']}; "
            "ejecutadas como mínimo incluyendo trabajo perdido: "
            f"{recovery['minimum_executed_exposures_including_lost_work']}. "
            f"Se reservaron {recovery['reserved_consumed_seconds']} s del presupuesto "
            "original para todo el trabajo anterior al reinicio. Por ello, elapsed_seconds "
            "incluye una reserva conservadora; no es una medición exacta del tiempo "
            "de cómputo. No se amplió el límite de 7.200 s ni el de 2.048 updates lógicos.",
        ]
    args.output.mkdir()
    write_json(args.output / "RESULTS.json", result)
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name in ("build_monocop_screen_result.py", "build_detection_pretrained_result.py"):
        shutil.copyfile(ROOT / "scripts" / name, args.output / name)
    write_json(
        args.output / "MANIFEST.json",
        {
            "inputs": inputs,
            "files": {p.name: digest(p) for p in args.output.iterdir() if p.is_file()},
        },
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "best": best,
                "selected_update": selected_update,
                "results_sha256": digest(args.output / "RESULTS.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
