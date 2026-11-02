"""Diagnose retained TRAIN proposals without repeating detector inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.detection.benchmark_evaluation import CLASSES, PROFILE_SHA256  # noqa: E402
from e_jepa_ttc.detection.eap_rgb_data import finite_matrix  # noqa: E402
from e_jepa_ttc.detection.range_campaign import digest, write_json  # noqa: E402
from e_jepa_ttc.detection.raw_diagnostics import RawDiagnostics  # noqa: E402


def main() -> None:
    """Reconcile full rare-class TRAIN coverage and reuse the exact selection diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--inference", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--census", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Set CUDA_VISIBLE_DEVICES=-1 for this CPU-only diagnostic")
    torch.set_num_threads(1)
    inputs: dict[str, str] = {}

    def read(path: Path, expected: str | None = None) -> dict:
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Changed input: {path}")
        inputs[path.as_posix()] = actual
        return json.loads(path.read_text(encoding="utf-8"))

    complete = read(args.inference / "COMPLETE.json")
    intake = read(args.inference / "INTAKE.json", complete["intake_sha256"])
    panel = read(args.panel / "PANEL.json", intake["panel_sha256"])
    gt = read(args.panel / "gt_train.json", panel["gt_train_sha256"])
    reference = read(args.reference / "COMPLETE.json")
    selection = read(
        args.reference / "ERROR_DIAGNOSTICS.json", reference["files"]["ERROR_DIAGNOSTICS.json"]
    )
    census = read(args.census)
    if (
        complete["role"] != "train"
        or complete["frames"] != 4096
        or gt["role"] != "train"
        or intake["role"] != "train"
        or intake["checkpoint_sha256"] != reference["checkpoint_sha256"]
        or reference["status"] != "VERIFIED"
        or reference["frames"] != 23958
        or reference["profile_sha256"] != PROFILE_SHA256
        or panel["test_labels_read"]
        or census["label_sha256"] != panel["label_sha256"]
    ):
        raise ValueError("TRAIN ownership or retained selection reference differs")
    for relative, expected in reference["source_sha256"].items():
        relative = relative.replace("\\", "/")
        if Path(relative).name in (
            "raw_diagnostics.py",
            "benchmark_evaluation.py",
            "geometry_contract.py",
        ):
            if digest(ROOT / relative) != expected:
                raise ValueError("Selection and TRAIN diagnostic implementations differ")
    tokens = panel["roles"]["train"]
    if set(tokens) != set(gt["ground_truth"]) or len(tokens) != 4096:
        raise ValueError("TRAIN panel population differs")
    metadata_path = args.dataset / "data/train.parquet"
    if digest(metadata_path) != panel["source_media_sha256"]:
        raise ValueError("Canonical sensor metadata changed")
    inputs[metadata_path.as_posix()] = panel["source_media_sha256"]
    table = pd.read_parquet(
        metadata_path,
        columns=["sample_token", "sequence_id", "K_event", "T_event_ego"],
        filters=[("sample_token", "in", tokens)],
    )
    frames = {}
    for token, sequence, intrinsic, transform in table.itertuples(index=False, name=None):
        media = panel["media"][token]
        if token in frames or media["role"] != "train" or media["sequence_id"] != sequence:
            raise ValueError("Metadata duplication or ownership mismatch")
        frames[token] = {
            "intrinsic": finite_matrix(intrinsic, (3, 3)),
            "event_from_ego": finite_matrix(transform, (4, 4)),
            "image_wh": media["image_wh"],
        }
    if set(frames) != set(tokens):
        raise ValueError("Canonical metadata does not cover the entire TRAIN panel")
    paths = [
        Path(__file__),
        *[
            ROOT / f"src/e_jepa_ttc/detection/{name}.py"
            for name in (
                "raw_diagnostics",
                "benchmark_evaluation",
                "geometry_contract",
                "range_campaign",
                "eap_rgb_data",
            )
        ],
    ]
    sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    contract = {
        "schema": "phase_train_query_diagnostic_v1",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sources": sources,
        "inputs": dict(inputs),
        "frames": 4096,
        "population": "TRAIN panel enriched with every raw bicycle/other_vehicle frame",
        "comparison_scope": "Different TRAIN and selection populations; not a generalization gap",
        "maximum_seconds": 600,
        "new_inference": False,
        "new_training": False,
        "test_labels_read": False,
        "promotion": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "INTAKE.json", contract)
    for relative in sources:
        target = args.output / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        if digest(target) != sources[relative]:
            raise ValueError("Diagnostic source copy differs")
    lock = args.output / "WRITER.lock"
    lock.write_text(str(os.getpid()))
    started = time.perf_counter()
    diagnostic = RawDiagnostics()
    seen: set[str] = set()
    counts: Counter = Counter()
    try:
        for name, expected in sorted(complete["shards"].items()):
            if time.perf_counter() - started > contract["maximum_seconds"]:
                raise TimeoutError("TRAIN diagnostic budget exhausted")
            folder = args.inference / "shards" / name
            receipt = read(folder / "COMPLETE.json", expected)
            predictions = read(folder / "predictions.json", receipt["files"]["predictions.json"])
            if (
                receipt["intake_sha256"] != complete["intake_sha256"]
                or predictions["checkpoint_sha256"] != intake["checkpoint_sha256"]
                or predictions["targets_argument"] is not None
                or predictions["annotations_returned"] != 0
                or set(predictions["predictions"]) != set(receipt["tokens"])
                or seen.intersection(receipt["tokens"])
            ):
                raise ValueError("Retained inference lineage or token population changed")
            for token in receipt["tokens"]:
                boxes = gt["ground_truth"][token]
                # Admission was not replayed here; do not label raw objects as admitted.
                raw = [{**b, "ignore_reason": "TRAIN_admission_not_replayed"} for b in boxes]
                diagnostic.add_frame(token, predictions["predictions"][token], raw, frames[token])
                counts.update(b["class_name"] for b in boxes)
            seen.update(receipt["tokens"])
            write_json(
                args.output / "STATUS.json",
                {
                    "status": "DIAGNOSING",
                    "frames": len(seen),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
        if seen != set(tokens) or dict(counts) != panel["raw_train_class_counts"]:
            raise ValueError("TRAIN frame or raw class counts do not reconcile")
        if torch.cuda.is_initialized():
            raise ValueError("CPU diagnostic initialized CUDA")
        for path, expected in inputs.items():
            if digest(Path(path)) != expected:
                raise ValueError("Diagnostic input changed during the run")
        if sources != {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}:
            raise ValueError("Diagnostic code changed")
        result = diagnostic.summarize()
        result.pop("examples")  # Retain aggregate evidence without copying raw labels to Git.
        write_json(args.output / "TRAIN_DIAGNOSTICS.json", result)
        write_json(
            args.output / "COMPARISON.json",
            {
                "train": result,
                "selection": {
                    key: selection[key] for key in ("groups", "confusion_agnostic_maximum_4m")
                },
                "train_all_sequence_instance_id_counts": census["sequence_instance_id_counts"],
                "population_caveat": contract["comparison_scope"],
                "promotion": False,
            },
        )
        render_report(args.output, result, selection, census)
        write_json(
            args.output / "COMPLETE.json",
            {
                "status": "VERIFIED",
                "frames": len(seen),
                "cuda_initialized": False,
                "intake_sha256": digest(args.output / "INTAKE.json"),
                "source_shards": complete["shards"],
                "files": {
                    n: digest(args.output / n)
                    for n in ("TRAIN_DIAGNOSTICS.json", "COMPARISON.json", "REPORT.md")
                },
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        write_json(args.output / "STATUS.json", {"status": "COMPLETE", "frames": len(seen)})
    except Exception as error:
        failure = {"status": "FAILED", "error": str(error)}
        write_json(args.output / "FAILURE.json", failure)
        write_json(args.output / "STATUS.json", failure)
        raise
    finally:
        lock.unlink()


def render_report(output: Path, train: dict, selection: dict, census: dict) -> None:
    """Report fixed-pool recall separately from AP and independent generalization."""
    lines = [
        "# Diagnóstico de propuestas TRAIN y clases raras",
        "",
        "Mismo checkpoint congelado, sin inferencia nueva. TRAIN: 4.096 imágenes, "
        "incluidas todas las imágenes TRAIN anotadas con bicycle u other_vehicle. "
        "Selección: 23.958 imágenes. Son poblaciones diferentes; sus diferencias "
        "no estiman por sí solas una brecha de generalización. Las vistas de un mismo "
        "objeto están correlacionadas.",
        "",
        "Recall a 2 m con asignación de cardinalidad máxima en el frame completo. "
        "El matching ignora el orden de scores. Después se agrupan los GT por clase. "
        "No son AP ni recall operativo a un umbral de confianza.",
        "",
        "| Clase | GT TRAIN | TRAIN clase correcta | TRAIN sin exigir clase | "
        "GT selección | Selección clase correcta | Selección sin exigir clase |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cls in CLASSES:
        a, b = train["groups"][f"class/{cls}"], selection["groups"][f"class/{cls}"]
        lines.append(
            f"| {cls} | {a['gt']} | {a['recall']['same_class_maximum_2m']:.6f} | "
            f"{a['recall']['agnostic_maximum_2m']:.6f} | {b['gt']} | "
            f"{b['recall']['same_class_maximum_2m']:.6f} | "
            f"{b['recall']['agnostic_maximum_2m']:.6f} |"
        )
    lines += [
        "",
        "Confusión sobre asociaciones agnósticas máximas a 4 m; "
        "un emparejamiento cercano no certifica identidad física del objeto.",
        "",
        "| GT | Clase predicha | Asociaciones TRAIN | Asociaciones selección |",
        "|---|---|---:|---:|",
    ]
    for cls in ("bicycle", "other_vehicle", "pedestrian"):
        for predicted in CLASSES:
            key = f"{cls}->{predicted}"
            a = train["confusion_agnostic_maximum_4m"].get(key, 0)
            b = selection["confusion_agnostic_maximum_4m"].get(key, 0)
            if a or b:
                lines.append(f"| {cls} | {predicted} | {a} | {b} |")
    lines += [
        "",
        "IDs (secuencia, instance_id) del censo TRAIN completo, no solo este panel. "
        "No equivalen a escenas independientes.",
        "",
        "| Clase | IDs raw | IDs admitidos 60 m | IDs admitidos 150 m |",
        "|---|---:|---:|---:|",
    ]
    identities = census["sequence_instance_id_counts"]
    for cls in ("bicycle", "other_vehicle", "pedestrian"):
        lines.append(
            f"| {cls} | {identities['raw'][cls]} | {identities['admitted60'][cls]} | "
            f"{identities['admitted150'][cls]} |"
        )
    lines += [
        "",
        "El diagnóstico no separa por sí solo capacidad, ambigüedad visual, "
        "ruido de anotación o desbalance. Las distancias usan GT únicamente para medir "
        "errores; no producen nuevas predicciones. No se ha entrenado una receta nueva "
        "ni se promueve un modelo. Se conservan los resultados negativos de R2-rare, "
        "de los rankers y del ensayo semántico CPU.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
