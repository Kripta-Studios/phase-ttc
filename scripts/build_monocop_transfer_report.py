"""Regenerate the bounded CPU transfer result, including the API-correction replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    """Hash an evidence file without retaining checkpoint bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Verify both receipts, executed bytes and weights before rendering their results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-run", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs: dict[str, str] = {}

    def read(path: Path) -> dict:
        inputs[str(path)] = digest(path)
        return json.loads(path.read_text(encoding="utf-8"))

    conversion = read(args.weights / "CONVERSION.json")
    if conversion["model_state_sha256"] != digest(args.weights / "model_state.pt"):
        raise ValueError("Pure upstream weights changed")
    if conversion["source_sha256"] != digest(args.weights / "executed_converter.py"):
        raise ValueError("Executed conversion source changed")
    first, result = [read(path / "RESULTS.json") for path in (args.first_run, args.run)]
    for directory, value in ((args.first_run, first), (args.run, result)):
        if (
            value["status"] != "PASS_CPU_TRANSFER_ONLY"
            or value["cuda_initialized"]
            or value["optimizer_steps"] != 0
            or value["quality_evaluated"]
            or value["gpu_training_admitted"]
            or not value["model_parameters_and_buffers_unchanged_by_checks"]
            or value["conversion_sha256"] != inputs[str(args.weights / "CONVERSION.json")]
        ):
            raise ValueError("Transfer result was not a successful CPU-only check")
        for name, expected in value["source_sha256"].items():
            if digest(directory / ("executed_" + Path(name.replace("\\", "/")).name)) != expected:
                raise ValueError(f"Executed source differs: {name}")
        if (
            digest(directory / "initialized_model.pt")
            != value["stages"]["checkpoint_replay"]["result"]["sha256"]
        ):
            raise ValueError("Initialized checkpoint changed")
    if (
        first["configuration"] != result["configuration"]
        or first["stages"].keys() != result["stages"].keys()
    ):
        raise ValueError("API-correction replay changed configuration or stages")
    for name in first["stages"]:
        if first["stages"][name]["result"] != result["stages"][name]["result"]:
            raise ValueError(f"API-correction replay changed numerical results: {name}")
    real = result["stages"]["real_train_frame_backward"]["result"]
    transfer = result["stages"]["transfer_contract"]["result"]
    checkpoint_sha = result["stages"]["checkpoint_replay"]["result"]["sha256"]
    lines = [
        "# MonoCoP multiclase: transferencia CPU verificada",
        "",
        f"Estado **{result['status']}**. Pesos públicos fijados, cero pasos de "
        "optimizador y ninguna métrica eAP medida. No hay promoción ni otro "
        "entrenamiento GPU admitido por este resultado.",
        "",
        f"Checkpoint original: `{conversion['upstream_checkpoint_sha256']}`; "
        f"{conversion['tensor_count']} tensores. Pesos puros: "
        f"`{conversion['model_state_sha256']}`. La conversión no reutiliza "
        "el optimizador y conserva exactamente todos los tensores.",
        "",
        "Se mapearon Pedestrian→pedestrian, Car→car y Cyclist→bike. Las otras "
        "cinco filas se inicializaron por separado. Cyclist/bike es una hipótesis "
        "de transferencia, no una equivalencia certificada de las anotaciones. "
        "Los bins de 300 m se conservaron y su clasificador se reinicializó; "
        "el embedding conserva las filas físicas 0–60 y repite la última por encima.",
        "",
        f"Se copiaron {len(transfer['copied_tensor_keys'])} tensores completos, "
        f"se adaptaron {len(transfer['adapted_tensor_keys'])} y se verificaron "
        f"{len(transfer['shared_tensor_aliases_verified'])} grupos de aliases. "
        "Repetir la transferencia conserva pesos exactos aunque cambie el RNG "
        "exterior, y no modifica el estado aleatorio CPU del llamante.",
        "",
        "| Etapa | CPU v1, s | CPU v2, s |",
        "|---|---:|---:|",
    ]
    for name, stage in result["stages"].items():
        lines.append(
            f"| {name} | {first['stages'][name]['seconds']:.3f} | {stage['seconds']:.3f} |"
        )
    lines += [
        "",
        "v1 pasó las pruebas de ejecución, pero Pyright rechazó el acceso al "
        "generador CPU por un símbolo no exportado. v2 usa Generator/get_state/"
        "set_rng_state públicos. Todos los resultados numéricos de las etapas "
        "y el hash del checkpoint inicializado son **idénticos**. No hubo "
        "reentrenamiento ni ajuste según métricas de selección.",
        "",
        "Pasaron forward/backward con ocho clases y targets vacíos, replay de "
        "inferencia sin etiquetas y un frame TRAIN real a 640×384. Los parámetros "
        "y buffers quedaron intactos tras las comprobaciones.",
        "",
        f"Frame `{real['token']}`: {real['admitted_targets']} targets y profundidad "
        f"máxima {real['maximum_target_depth_m']:.6f} m. El decoder emitió "
        f"{real['decoded_boxes']}/{real['decode_counts']['queries']} cajas con "
        "geometría válida. Esto no mide recall, AP ni utilidad de esas cajas. "
        "El tiempo de esta etapa incluye lectura, preparación, backward y "
        "otra inferencia; no es latencia aislada ni una estimación de coste GPU.",
        "",
        f"Checkpoint inicializado: `{checkpoint_sha}`. Es un punto de partida "
        "reproducible, no un candidato entrenado en eAP.",
        "",
        "Se conservan v1, v2, fuentes ejecutadas y los recibos de procedencia. "
        "El checkpoint upstream no incluye hashes de su receta original; "
        "la asociación a las tres clases se apoya en la configuración y el "
        "log publicados. No se certifican aquí licencia de pesos ni admisión "
        "de datos externos en una evaluación oficial.",
        "",
        "Siguiente decisión: terminar KITTI y revisar su resultado. Después, "
        "si procede, registrar un smoke GPU acotado para aprendizaje, paridad "
        "y coste; el screening de calidad requiere su propio control y presupuesto.",
        "",
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    report = args.output / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "input_sha256": inputs,
        "report_sha256": digest(report),
        "generator_sha256": digest(Path(__file__)),
        "api_correction_all_stage_results_exact": True,
        "initialized_checkpoint_sha256": checkpoint_sha,
        "quality_evaluated": False,
        "promotion": False,
    }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
