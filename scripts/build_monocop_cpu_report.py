"""Render the verified MonoCoP CPU admission without treating it as detection quality."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    """Hash an evidence artifact."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Verify executed sources and regenerate the technical result table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.run / "RESULTS.json"
    result = json.loads(result_path.read_text())
    source_path = args.source / "PORT.json"
    source = json.loads(source_path.read_text())
    if (
        result["status"] != "PASS_CPU_PORT_ONLY"
        or result["cuda_initialized"]
        or result["optimizer_steps"] != 0
        or result["eap_quality_measured"]
        or result["gpu_training_admitted"]
        or result["source_receipt_sha256"] != digest(source_path)
        or result["runner_sha256"] != digest(args.run / "executed_runner.py")
        or source["generator_sha256"] != digest(args.run / "executed_prepare.py")
        or source["patch_sha256"] != digest(args.source / "PORT.patch")
    ):
        raise ValueError("CPU evidence or source binding differs")
    for name, expected in source["files"].items():
        if digest(args.source / name) != expected:
            raise ValueError(f"Executed model source changed: {name}")
    checkpoint = args.run / "random_fixture_model.pt"
    if digest(checkpoint) != result["stages"]["checkpoint_replay"]["result"]["sha256"]:
        raise ValueError("Random fixture checkpoint changed")
    lines = [
        "# MonoCoP: admisión técnica CPU",
        "",
        f"Estado **{result['status']}**; {result['parameters']:,} parámetros, "
        f"{result['trainable_parameters']:,} entrenables. Dos hilos CPU; "
        f"tiempo total {result['elapsed_seconds']:.3f} s. CUDA no se inicializó.",
        "",
        "Inicialización aleatoria, cero pasos de optimizador. No es una evaluación "
        "de detección ni una admisión para entrenar en GPU.",
        "",
        "| Etapa | Tiempo CPU, s |",
        "|---|---:|",
    ]
    for name, row in result["stages"].items():
        lines.append(f"| {name} | {row['seconds']:.3f} |")
    attention = result["stages"]["attention_reference"]["result"]
    focal = result["stages"]["focal_contract"]["result"]
    real = result["stages"]["real_train_frame_backward"]["result"]
    lines += [
        "",
        f"Atención fp64 frente a la referencia PyTorch del autor: error forward máximo "
        f"{attention['forward_max_abs']:.3e}; errores de primeros gradientes "
        f"{attention['gradient_max_abs_value_locations_weights']}. "
        "La extensión CUDA compilada no se comparó.",
        "",
        "Se ejecutó el forward/backward completo con ocho clases sintéticas y con "
        "targets vacíos: 550 consultas y ocho logits por consulta durante entrenamiento. "
        "En evaluación se comprobaron 50 consultas, `targets=None` y replay exacto "
        "tras guardar, perturbar y restaurar los pesos.",
        "",
        f"Sensibilidad geométrica: error máximo frente al cambio esperado por fy "
        f"{focal['fy_max_abs_error']:.3e}; cambiar solamente fx conservó las salidas "
        "exactamente. El fixture de 96×160 tiene focales distintas. La pirámide "
        "se ajustó al tamaño receptor después del fallo v1; conserva valores y "
        "gradientes del aumento ×2 para niveles compatibles de una entrada 640×384.",
        "",
        f"Frame TRAIN real: `{real['token']}`, {real['admitted_targets']} targets "
        f"admitidos y profundidad máxima {real['maximum_target_depth_m']:.6f} m. "
        "La etapa incluye lectura, preparación, forward/backward a 640×384, "
        "otra inferencia sin etiquetas y decodificación; su tiempo no es una "
        "medida aislada de latencia ni una estimación de velocidad GPU.",
        "",
        f"Con pesos aleatorios se emitieron {real['decoded_boxes']} cajas válidas "
        f"de {real['decode_counts']['queries']} consultas. Las cajas con tamaño o "
        "profundidad no positivos fueron rechazadas por el decoder. El port aún "
        "necesita pesos admitidos, aprendizaje y evaluación de ocho clases.",
        "",
        "Los bins comparten el rango de 300 m entre predictor y pérdida. Se comprobaron "
        "las fronteras de rango, profundidades lejanas y entradas no finitas. "
        "Se preserva el intento v1 fallido, su código y su traceback.",
        "",
        "Siguiente paso técnico: inspección y transferencia trazable del checkpoint "
        "multiclase oficial. La eventual comparación GPU requiere un presupuesto "
        "y coste medidos, y la decisión posterior al ensayo KITTI en curso.",
        "",
        "Referencias de API: "
        "[grid_sample de PyTorch 2.11]"
        "(https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.grid_sample.html). "
        "Los resultados numéricos de esta página proceden del JSON del ensayo local.",
        "",
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    report = args.output / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "result_sha256": digest(result_path),
        "source_receipt_sha256": digest(source_path),
        "random_checkpoint_sha256": digest(checkpoint),
        "generator_sha256": digest(Path(__file__)),
        "report_sha256": digest(report),
        "eap_quality_measured": False,
        "promotion": False,
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
