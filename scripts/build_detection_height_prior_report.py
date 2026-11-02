"""Render a TRAIN-only geometric-prior diagnostic from its saved results."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    """Hash a file without loading its entire contents into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def render(result: dict) -> str:
    """Keep ideal target geometry distinct from model accuracy."""
    lines = [
        "# Sesgo del prior de profundidad por altura en TRAIN",
        "",
        f"Panel registrado: {result['frames']:,} imágenes TRAIN; "
        f"{result['raw_targets']:,} anotaciones originales y "
        f"{result['admitted_targets']:,} targets admitidos a 300 m. "
        "No se leen etiquetas de selección ni test; no hay inferencia ni entrenamiento.",
        "",
        "Se compara z = altura × fy / altura de la caja proyectada con la profundidad "
        "óptica del centro. Altura, yaw y bordes son targets perfectos: este diagnóstico "
        "mide el sesgo de esa fórmula, no el error final de un detector con residual aprendido.",
        "",
        "La alternativa invierte exactamente la proyección pinhole de los ocho vértices. "
        "Cada borde no recortado produce una profundidad; se toma su mediana. "
        "Ambas fórmulas se comparan también sobre el mismo subconjunto válido. "
        "Los errores pequeños de la inversión incluyen el redondeo de targets a float32. "
        "Esto comprueba consistencia interna, no calibración física del sensor.",
        "",
        "| Grupo | Targets | Sesgo altura (m) | MAE altura (m) | P90 abs. (m) | "
        "Targets con borde válido | MAE altura en ese subconjunto | MAE inversión (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted(result["groups"].items()):
        a = row["height_prior"]
        b = row["height_prior_on_edge_valid_subset"]
        c = row["cuboid_edges_on_same_subset"]
        if b["count"] != c["count"]:
            raise ValueError("Compared subsets differ")
        matched = f"{b['mae_m']:.6f} | {c['mae_m']:.6f}" if b["count"] else "n/a | n/a"
        lines.append(
            f"| {name} | {a['count']} | {a['mean_signed_m']:.6f} | {a['mae_m']:.6f} | "
            f"{a['p90_abs_m']:.6f} | {b['count']} | {matched} |"
        )
    lines += [
        "",
        "Sensibilidad: cambio de profundidad al aumentar un píxel la altura proyectada, "
        "manteniendo tamaño y focal. No es una cota de precisión: un regresor puede "
        "predecir coordenadas subpíxel.",
        "",
        "| Distancia óptica | Mediana de sensibilidad (m/píxel) |",
        "|---|---:|",
    ]
    for name, row in sorted(result["groups"].items()):
        if name.startswith("depth/"):
            lines.append(f"| {name[6:]} | {row['median_one_pixel_sensitivity_m']:.6f} |")
    lines += [
        "",
        "No se ha cambiado ni promocionado ningún modelo. Sustituir directamente el prior "
        "de un checkpoint entrenado cambiaría su predicción y podría sumar una corrección "
        "que su residual ya aprendió. Antes de entrenar otra variante, hay que medir la "
        "inversión con tamaños, orientación y bordes predichos, separar bordes recortados "
        "y declarar cómo se tratará ese residual.",
        "",
        f"Tiempo del diagnóstico: {result['elapsed_seconds']:.3f} s. "
        f"CUDA inicializada: {result['cuda_initialized']}.",
        "",
        f"Filas comprimidas SHA-256: `{result['rows_sha256']}`. "
        "Se conservan fuera de Git; contienen información derivada de las anotaciones.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Validate the receipt and preserve executed sources with the generated report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result_path = args.audit / "RESULTS.json"
    result = json.loads(result_path.read_text())
    status = json.loads((args.audit / "STATUS.json").read_text())
    if (
        args.output.exists()
        or status["status"] != "COMPLETE"
        or result["status"] != "VERIFIED_TRAIN_DIAGNOSTIC_ONLY"
        or result["frames"] != 4096
        or result["groups"]["all"]["height_prior"]["count"] != result["admitted_targets"]
        or result["cuda_initialized"]
        or result["model_inference"]
        or result["training"]
        or result["test_labels_read"]
        or digest(args.audit / "ROWS.jsonl.gz") != result["rows_sha256"]
    ):
        raise ValueError("Incomplete, incompatible or changed audit")
    for name, expected in result["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Executed diagnostic source changed: {name}")
    for name, expected in result["input_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Diagnostic input changed: {name}")
    args.output.mkdir(parents=True)
    for name in result["source_sha256"]:
        target = args.output / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    shutil.copyfile(result_path, args.output / "RESULTS.json")
    shutil.copyfile(Path(__file__), args.output / Path(__file__).name)
    (args.output / "REPORT.md").write_text(render(result), encoding="utf-8")
    files = {
        p.relative_to(args.output).as_posix(): digest(p)
        for p in args.output.rglob("*")
        if p.is_file()
    }
    manifest = {"audit": args.audit.as_posix(), "files": files, "promotion": False}
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.output / "REPORT.md"), "files": len(files)}))


if __name__ == "__main__":
    main()
