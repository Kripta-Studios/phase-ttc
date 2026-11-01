"""Write a compact continuation handoff from live campaign JSON, without loading Torch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Snapshot actual states and metrics; do not infer completion from queued plans."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range-root", required=True, type=Path)
    parser.add_argument("--query-root", required=True, type=Path)
    parser.add_argument("--pretrained-root", required=True, type=Path)
    parser.add_argument("--aligned-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs: dict[str, str] = {}

    def read(path: Path) -> Any:
        raw = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    now = datetime.now(UTC).isoformat()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    incumbent = read(args.aligned_root / "ranked.json")
    census = read(args.range_root / "train_long_range_v2/CENSUS.json")
    lines = [
        "# Continuidad PHASE-TTC eAP",
        "",
        f"Snapshot UTC: {now}. No acredita estados posteriores.",
        f"Worktree de trabajo: `{ROOT.as_posix()}`.",
        f"Commit local: `{commit}`. Rama `dev/eap-diagnostics-20261101`.",
        "",
        "El objetivo sigue abierto. No se ha demostrado un modelo excelente ni "
        "una mAP próxima a 1,0. No declarar éxito por agotar horas o terminar una cola.",
        f"Incumbente en selección completa alineada: mAP **{incumbent['mAP']:.8f}**.",
        "No comparar directamente esta cifra con el panel de 512 imágenes.",
        "",
        "## Rango",
        "",
        f"Censo: {census['frames']} frames TRAIN, {len(census['sequences'])} secuencias; "
        f"máximo óptico {census['raw_center_depth_extrema_m'][1]:.6f} m. "
        "No hay etiquetas TRAIN a 400 m.",
        "",
        "| Run | Estado | Update comprometido | Mejor mAP panel |",
        "|---|---|---:|---:|",
    ]
    for name in ("screen_range60_v1", "screen_range150_v1"):
        status = read(args.range_root / name / "STATUS.json")
        lines.append(
            f"| {name} | {status['status']} | {status['update']} | "
            f"{status['best_aligned_mAP']:.8f} |"
        )
    lines.extend(
        [
            "",
            "Evaluaciones terminadas del brazo de 150 m:",
            "",
            "| Update | mAP panel |",
            "|---:|---:|",
        ]
    )
    for path in sorted((args.range_root / "screen_range150_v1/evaluation").glob("*.json")):
        value = read(path)
        lines.append(f"| {int(path.stem)} | {value['metrics']['mAP']:.8f} |")
    lines.extend(["", "## Colas posteriores", ""])
    for name, queue in (
        ("Ranking por clase", args.query_root / "campaign/QUEUE.json"),
        ("Rango de 300 m", args.range_root / "long_range_campaign_v1/QUEUE.json"),
    ):
        state = read(queue.with_suffix(".status.json"))
        contract = read(queue)
        lines.extend(
            [
                f"- {name}: **{state['status']}**, etapa activa: "
                f"`{state.get('active_job', 'ninguna')}`.",
                f"  Contrato: `{queue.as_posix()}`, SHA-256 `{inputs[str(queue)]}`.",
            ]
        )
        if not contract.get("sources"):
            raise ValueError("Missing guarded source inventory")
    lines.extend(
        [
            "",
            "La cola de 300 m espera el fin de la inferencia TRAIN del ranker. "
            "No añadir otro proceso GPU que compita con estas colas.",
            "No editar las fuentes enumeradas en los contratos activos. "
            "Un cambio exige otro productor/contrato; nunca relajar hashes.",
            "",
            "## Inicialización externa",
            "",
        ]
    )
    audit = read(args.pretrained_root / "cpu_transfer_v2/TRANSFER_AUDIT.json")
    conversion = read(args.pretrained_root / "monodgp_kitti_plain_v1/CONVERSION.json")
    weights_path = args.pretrained_root / "monodgp_kitti_plain_v1/model_state.pt"
    lines.extend(
        [
            f"Transferencia KITTI: **{audit['status']}**, "
            f"CUDA inicializada: {audit['cuda_initialized']}. "
            "No se ha medido mAP del modelo transferido.",
            f"Pesos puros: `{weights_path.as_posix()}`, "
            f"SHA-256 `{conversion['model_state_sha256']}`.",
            "Falta smoke GPU y comparación de aprendizaje; conservar el fallo CPU v1.",
            "",
            "## Siguiente trabajo",
            "",
            "1. Verificar procesos, locks y estados reales antes de intervenir.",
            "2. Cerrar 150 m con sus evaluaciones previstas y registrar el resultado, "
            "incluidas las caídas. Confirmar en selección completa solo un candidato justificado.",
            "3. Verificar la paridad GPU y los resultados de ranking por clase; "
            "seguir el smoke/gate de 300 m y su comparación controlada.",
            "4. Si las mejoras son insuficientes, usar la admisión técnica del checkpoint "
            "KITTI para preparar el siguiente ensayo, sin atribuirle rendimiento no medido.",
            "5. Mantener ocho clases, raw GT, particiones, recibos y artefactos negativos. "
            "No tocar etiquetas privadas ni usar score test para hiperparámetros.",
            "6. Todo gasto cloud nuevo requiere un límite explícito. "
            "El trabajo autorizado y en cola es local.",
            "",
            "Documentación en el worktree: LONG_RANGE_TRAIN_AUDIT_20261101.md, "
            "LONG_RANGE_CAMPAIGN_20261101.md y PRETRAINED_MONODGP_INTAKE_20261101.md "
            "bajo docs/detection_sota_20261029/.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "timestamp_utc": now,
                "git_commit": commit,
                "input_sha256": inputs,
                "report_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "goal_complete": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
