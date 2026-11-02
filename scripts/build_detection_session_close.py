"""Package the best verified detector and close this bounded research round."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    """Hash an artifact incrementally."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Generate all reported values from retained JSON and verify copied model files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--monocop-report", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--range-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Require a new delivery directory")
    root = Path(__file__).resolve().parents[1]
    inputs: dict[str, str] = {}

    def read(path: Path) -> dict:
        inputs[path.as_posix()] = digest(path)
        return json.loads(path.read_text())

    release = read(args.release / "MANIFEST.json")
    aligned = read(args.aligned / "ranked.json")
    unranked = read(args.aligned / "baseline.json")
    official = read(
        root / "docs/detection_sota_20261029/evidence/official_benchmark_20261101/scores.json"
    )
    screen = read(args.monocop_report / "RESULTS.json")
    ideal = read(args.geometry / "train_panel_v2/RESULTS.json")
    predicted = read(args.geometry / "predicted_train_v1/RESULTS.json")
    receipt = read(args.geometry / "predicted_train_v1/COMPLETE.json")
    census = read(args.range_root / "train_long_range_v2/CENSUS.json")
    if (
        screen["status"]["status"] != "COMPLETE"
        or screen["panel_gate_passed"]
        or screen["promotion"]
        or aligned["samples"] != 23958
        or receipt["results_sha256"]
        != inputs[(args.geometry / "predicted_train_v1/RESULTS.json").as_posix()]
        or not receipt["baseline_decode_exact"]
        or receipt["frames"] != 4096
        or ideal["rows_sha256"] != digest(args.geometry / "train_panel_v2/ROWS.jsonl.gz")
    ):
        raise ValueError("Incomplete closure or a positive candidate needs separate review")
    for name, expected in release["files"].items():
        if digest(args.release / name) != expected:
            raise ValueError(f"Best-model package changed: {name}")
    args.output.mkdir(parents=True)
    model = args.output / "modelo"
    for name, expected in {
        **release["files"],
        "MANIFEST.json": digest(args.release / "MANIFEST.json"),
    }.items():
        target = model / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.release / name, target)
        if digest(target) != expected:
            raise ValueError(f"Delivery copy differs: {name}")
    evidence_sources = {
        "monocop/REPORT.md": args.monocop_report / "REPORT.md",
        "monocop/RESULTS.json": args.monocop_report / "RESULTS.json",
        "monocop/MANIFEST.json": args.monocop_report / "MANIFEST.json",
        "geometria_ideal/REPORT.md": args.geometry / "train_report_v2/REPORT.md",
        "geometria_ideal/RESULTS.json": args.geometry / "train_panel_v2/RESULTS.json",
        "geometria_predicha/RESULTS.json": args.geometry / "predicted_train_v1/RESULTS.json",
        "geometria_predicha/COMPLETE.json": args.geometry / "predicted_train_v1/COMPLETE.json",
        "geometria_predicha/PLAN.json": args.geometry / "predicted_train_v1/PLAN.json",
        "pytest.xml": args.geometry / "validation_v2/pytest.xml",
        "seleccion_completa.json": args.aligned / "ranked.json",
        "score_oficial.json": root
        / "docs/detection_sota_20261029/evidence/official_benchmark_20261101/scores.json",
    }
    for name, source in evidence_sources.items():
        target = args.output / "evidencia" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != digest(source):
            raise ValueError("Evidence copy differs")
    checkpoint_sha = release["files"][release["checkpoint"]]
    lines = [
        "# Cierre de la ronda eAP — 2 de noviembre de 2026",
        "",
        "Se cierra esta ronda a petición del usuario. El mejor candidato verificado sigue "
        "siendo Dfull RGB con ranker TRAIN. Las variantes nuevas no lo superan. "
        "No se ha conseguido un modelo excelente ni una mAP cercana a 1,0.",
        "",
        "| Resultado | Población | mAP |",
        "|---|---|---:|",
        f"| Dfull + ranker | Selección completa, {aligned['samples']} imágenes | "
        f"{aligned['mAP']:.8f} |",
        f"| Dfull sin ranker | Misma selección completa | {unranked['mAP']:.8f} |",
        "| Dfull + ranker, envío oficial anterior | Test oficial; protocolo privado "
        f"no certificado | {official['AP']:.8f} |",
        "",
        "No hubo un nuevo envío al benchmark ni gasto cloud en esta ronda local. "
        "No comparar directamente test oficial, selección completa y panel reducido.",
        "",
        "| Clase | GT en selección completa | AP del candidato conservado |",
        "|---|---:|---:|",
    ]
    for name, row in aligned["classes"].items():
        lines.append(f"| {name} | {row['gt_count']} | {row['AP']:.8f} |")
    lines += [
        "",
        "La taxonomía tiene ocho clases, pero la calidad no es suficiente en todas ellas. "
        "Una AP cero no implica necesariamente cero verdaderos positivos. Es un "
        "candidato de investigación de una semilla; no se certifica seguridad, producción, "
        "generalización independiente ni calidad TTC a partir de estas métricas de detección.",
        "",
        "## Último entrenamiento cerrado",
        "",
        "MonoCoP: 2.048 actualizaciones lógicas, 4.096 imágenes TRAIN y panel de 512. "
        "Misma aritmética raw-GT de ocho clases y mismo panel para las comparaciones siguientes.",
        "",
        "| Modelo en panel | Mejor mAP |",
        "|---|---:|",
        f"| Incumbente Dfull + ranker | {screen['incumbent_panel']['mAP']:.8f} |",
        f"| Control Dfull 300 m | {screen['control_df300']['best']:.8f} |",
        f"| MonoCoP | {screen['best']:.8f} |",
        "",
        f"Mejor MonoCoP en update {screen['selected_update']}. "
        f"Decisión: `{screen['decision']}`. No se promociona ni se ejecuta confirmación "
        "completa de una variante que no pasó el gate del panel.",
        "",
        "El reinicio del host interrumpió el run original. Se recuperó el estado íntegro "
        "de la actualización 1.792 y se conservó la evidencia dañada. Se repitieron 66 "
        "updates registrados pero no guardados; son 132 exposiciones extra de cómputo "
        "perdido, sin ampliar los 2.048 updates lógicos ni el presupuesto original. "
        "Los detalles y la reserva conservadora de tiempo están en evidencia/monocop/REPORT.md.",
        "",
        "## Geometría: hallazgo y prueba con predicciones",
        "",
        f"Con targets perfectos de {ideal['admitted_targets']} objetos TRAIN, el prior "
        "altura×fy/altura_en_píxeles tiene MAE "
        f"{ideal['groups']['all']['height_prior']['mae_m']:.6f} m. "
        "La inversión de los vértices recupera la profundidad dentro del redondeo numérico. "
        "Es consistencia de la proyección pinhole, no una mejora demostrada del detector.",
        "",
        "Prueba posterior: 4.096 imágenes TRAIN, 204.800 propuestas guardadas; "
        "decodificación original reproducida exactamente, sin inferencia ni entrenamiento nuevos. "
        "Mismas clases, scores y propuestas. Solo se sustituye la profundidad; "
        "las variantes eliminan el residual aprendido.",
        "",
        "| Profundidad usada en TRAIN | mAP TRAIN |",
        "|---|---:|",
    ]
    for name in ("baseline", "height_only", "cuboid_edges_only"):
        lines.append(f"| {name} | {predicted[name]['mAP']:.8f} |")
    lines += [
        "",
        "La geometría de vértices mejora respecto al prior de altura aislado, pero "
        "empeora respecto a la salida aprendida completa. No se aplica al candidato. "
        "Estas cifras TRAIN no estiman generalización ni son comparables con selección.",
        "",
        "## Objetos lejanos y alcance",
        "",
        f"El censo de {census['frames']} frames TRAIN llega a "
        f"{census['raw_center_depth_extrema_m'][1]:.6f} m de profundidad óptica. "
        "No hay anotaciones TRAIN a 400 m. Ampliar el límite no crea supervisión a esa "
        "distancia. El screening a 300 m tampoco mejoró el resultado global del incumbente.",
        "",
        "## Pesos y reproducción",
        "",
        f"Modelo conservado: [modelo/README.md](modelo/README.md). "
        f"Pesos: [selected.pt](modelo/{release['checkpoint']}). "
        "Ranker: [ranker.json](modelo/ranker.json). Los 112 archivos del manifest "
        "original fueron verificados por SHA-256 antes y después de copiar.",
        "",
        f"Checkpoint SHA-256: `{checkpoint_sha}`.",
        "",
        "Desde `modelo`, con el entorno Python 3.11/PyTorch CUDA documentado:",
        "",
        "```text",
        "python infer.py verify",
        "python infer.py infer --data-root DATA_ROOT --tokens tokens.json --output ../replay",
        "```",
        "",
        "No se incluyen datasets. El informe se regenera con "
        "`scripts/build_detection_session_close.py`; el JSON de cierre conserva rutas "
        "y hashes de entrada. Los diagnósticos conservan los intentos negativos, "
        "el fallo técnico inicial y las pruebas. La ronda se entrega sin iniciar otra campaña.",
        "",
        "Lo pendiente para una mejora sustancial es aprender mejores clases y centros 3D "
        "con evidencia de validación; no queda demostrado que baste aumentar rango, "
        "cambiar la inicialización o sustituir el prior geométrico. Una futura receta "
        "con geometría consistente necesitaría reentrenamiento y un control comparable. "
        "No se ha ejecutado como parte de este cierre.",
        "",
    ]
    (args.output / "CIERRE.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "round_closed": True,
        "excellent_model_demonstrated": False,
        "new_candidate_promoted": False,
        "input_sha256": inputs,
        "checkpoint_sha256": checkpoint_sha,
        "model_files_verified": len(release["files"]),
        "generator_sha256": digest(Path(__file__)),
    }
    (args.output / "CIERRE.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    shutil.copyfile(Path(__file__), args.output / Path(__file__).name)
    all_files = {
        p.relative_to(args.output).as_posix(): digest(p)
        for p in args.output.rglob("*")
        if p.is_file()
    }
    (args.output / "MANIFEST_ENTREGA.json").write_text(
        json.dumps(all_files, indent=2), encoding="utf-8"
    )
    print(json.dumps({"delivery": args.output.as_posix(), "verified_files": len(all_files)}))


if __name__ == "__main__":
    main()
