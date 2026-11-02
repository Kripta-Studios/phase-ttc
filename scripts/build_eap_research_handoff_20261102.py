"""Regenerate the two Spanish research handoff documents from archived evidence."""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/detection_sota_20261029"
EVIDENCE = DOCS / "evidence"
STATE_NAME = "ESTADO_RESULTADOS_Y_ARTEFACTOS_EAP_20261102.md"
PLAN_NAME = "PLAN_INVESTIGACION_Y_PROMPT_CODEX_EAP_20261102.md"
SOURCE_COMMIT = "543fa399"


def sha(path: Path) -> str:
    """Hash the exact archived bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a table whose numeric values originate in JSON, not prose literals."""
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def main() -> None:
    """Build documents and a receipt without importing Torch or reading datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DOCS)
    args = parser.parse_args()
    inputs: dict[str, str] = {}

    def read(relative: str) -> dict:
        path = EVIDENCE / relative
        if not path.exists():
            path = Path(str(path) + ".gz")
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
        raw = path.read_bytes()
        return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)

    aligned = read("benchmark_local_alignment_20261101/ranked.json")
    baseline = read("benchmark_local_alignment_20261101/baseline.json")
    scalar = read("benchmark_local_alignment_20261101/scalar_uncertainty.json")
    official = read("official_benchmark_20261101/scores.json")
    release = read("release_candidate_20261101/release_manifest.json")
    census = read("long_range_20261101/CENSUS.json")
    ranges = read("range300_screen_20261101/RESULTS.json")
    range150 = read("range150_confirmation_20261101/RESULTS.json")["metrics"]
    ensemble = read("range_ensemble_20261101/RESULTS.json")
    kitti = read("kitti_screen_20261101/RESULTS.json")
    mono = read("monocop_screen_final_20261102/RESULTS.json")
    recovery = read("monocop_screen_final_20261102/RECOVERY.json")
    delivery = read("monocop_screen_final_20261102/DELIVERY.json")
    ideal = read("height_prior_geometry_20261102/RESULTS.json")
    predicted = read("predicted_geometry_20261102/RESULTS.json")
    pred_receipt = read("predicted_geometry_20261102/COMPLETE.json")
    semantic = read("semantic_probe_20261101/RESULTS.json")
    raw_oracles = {
        name: read(f"raw_diagnostics_20261101/{name}.json")
        for name in ("oracle_class", "oracle_depth", "oracle_rank", "oracle_size")
    }
    if (
        len(aligned["classes"]) != 8
        or aligned["samples"] != 23958
        or mono["status"]["status"] != "COMPLETE"
        or mono["promotion"]
        or mono["panel_gate_passed"]
        or ranges["full_selection_300m_measured"]
        or pred_receipt["frames"] != 4096
        or not pred_receipt["baseline_decode_exact"]
        or delivery["new_candidate_promoted"]
    ):
        raise ValueError("Evidence does not match this closed research round")
    classes = table(
        ["Clase", "GT selección completa", "AP conservada"],
        [[name, row["gt_count"], f"{row['AP']:.8f}"] for name, row in aligned["classes"].items()],
    )
    full_rows = [
        ["Dfull sin ranker", f"{baseline['mAP']:.8f}", "Control geométrico original"],
        ["Dfull + ranker TRAIN", f"{aligned['mAP']:.8f}", "Mejor candidato conservado"],
        ["Score escalar de incertidumbre", f"{scalar['mAP']:.8f}", "No supera al conservado"],
    ]
    for folder, stem, label in (
        ("multiclass_20261101", "v1", "Ocho hipótesis por geometría"),
        ("multiclass_20261101", "top50_v1", "Hipótesis alternativas, top 50"),
        ("query_quality_20261101", "evaluation_linear_16_to_8", "Ranking por clase lineal"),
        ("query_quality_20261101", "evaluation_mlp_16_to_64_gelu_to_8", "Ranking por clase MLP"),
    ):
        metric = read(f"{folder}/{stem}_RESULTS.json")["metrics"]
        if (
            metric["samples"] != aligned["samples"]
            or metric["profile_sha256"] != aligned["profile_sha256"]
        ):
            raise ValueError("Full-selection populations or profiles differ")
        full_rows.append([label, f"{metric['mAP']:.8f}", "No promocionado"])
    full_rows += [
        ["Fine-tuning a 150 m", f"{range150['mAP']:.8f}", "Confirmado; no supera al conservado"],
        [
            "Combinación con frontera fija 60 m",
            f"{ensemble['metrics']['gated_60m']['mAP']:.8f}",
            "No promocionada",
        ],
        [
            "Unión sin frontera",
            f"{ensemble['metrics']['ungated_union']['mAP']:.8f}",
            "No promocionada",
        ],
    ]
    full_table = table(["Variante", "mAP selección completa", "Decisión"], full_rows)
    range_table = table(
        ["Update", "60 m", "150 m", "300 m"],
        [
            [
                step,
                *[
                    f"{ranges['runs'][r]['evaluations'][str(step)]['metrics']['mAP']:.8f}"
                    for r in ("60", "150", "300")
                ],
            ]
            for step in (0, 512, 1024, 1536, 2048)
        ],
    )
    model_table = table(
        ["Update", "MonoDGP desde KITTI", "MonoCoP"],
        [
            [
                step,
                f"{kitti['runs']['kitti']['evaluations'][str(step)]['metrics']['mAP']:.8f}",
                f"{mono['evaluations'][str(step)]['metrics']['mAP']:.8f}",
            ]
            for step in (0, 512, 1024, 1536, 2048)
        ],
    )
    oracle_table = table(
        ["Sustitución diagnóstica con GT", "mAP en selección completa"],
        [[name, f"{metric['mAP']:.8f}"] for name, metric in raw_oracles.items()],
    )
    panel_oracle_rows = []
    for label, folder in (
        ("MonoCoP inicial", "monocop_screen_baseline_diagnostic_20261101"),
        ("MonoCoP update 512", "monocop_screen_update512_diagnostic_20261101"),
        ("Dfull 300 m update 512", "control300_screen_update512_diagnostic_20261101"),
    ):
        panel_oracle_rows.append(
            [
                label,
                *[
                    f"{read(f'{folder}/{name}.json')['mAP']:.8f}"
                    for name in ("candidate", "oracle_class", "oracle_depth", "oracle_rank")
                ],
            ]
        )
    panel_oracle_table = table(
        ["Panel diagnóstico", "Real", "GT clase", "GT profundidad", "GT ranking"], panel_oracle_rows
    )
    census_table = table(
        ["Clase", "GT raw TRAIN", "Admitidos 60 m", "Admitidos 150 m", "Admitidos hasta 400 m"],
        [
            [
                name,
                *[
                    census["counts"][key].get(name, 0)
                    for key in ("raw", "admitted60", "admitted150", "admitted400")
                ],
            ]
            for name in aligned["classes"]
        ],
    )
    geometry_table = table(
        ["Grupo", "Targets", "MAE prior altura (m)", "MAE inversión (m)"],
        [
            [
                key,
                row["height_prior"]["count"],
                f"{row['height_prior']['mae_m']:.6f}",
                f"{row['cuboid_edges_on_same_subset']['mae_m']:.6f}",
            ]
            for key, row in ideal["groups"].items()
            if key == "all" or key.startswith("class/")
        ],
    )
    predicted_table = table(
        ["Profundidad", "mAP TRAIN", "Interpretación"],
        [
            [name, f"{predicted[name]['mAP']:.8f}", meaning]
            for name, meaning in (
                ("baseline", "Salida completa aprendida"),
                ("height_only", "Prior de altura sin residual"),
                ("cuboid_edges_only", "Inversión de vértices sin residual"),
            )
        ],
    )
    test_rows = []
    for name, relative in (
        (
            "Smoke MonoCoP y replay JSON",
            "monocop_gpu_execution_20261101/monocop_gpu_smoke_validation_v3/pytest.xml",
        ),
        (
            "Screen MonoCoP, pruebas CPU",
            "monocop_screen_preparation_20261101/monocop_screen_validation_v1/pytest.xml",
        ),
        (
            "Paridad de caché CUDA",
            "monocop_screen_preparation_20261101/monocop_screen_validation_v1/cuda_cache.xml",
        ),
        ("Geometría y sustitución de profundidad", "height_prior_geometry_20261102/pytest.xml"),
    ):
        path = EVIDENCE / relative
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
        suites = ElementTree.parse(path).getroot().findall(".//testsuite")
        totals = {
            key: sum(int(s.get(key, "0")) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        test_rows.append(
            [
                name,
                totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"],
                totals["skipped"],
                totals["failures"] + totals["errors"],
            ]
        )
    tests_table = table(
        ["Evidencia focalizada", "Pasadas", "Omitidas", "Fallos/errores"], test_rows
    )
    checkpoint_hashes = table(
        ["Artefacto", "SHA-256"],
        [
            ["Dfull conservado", release["files"][release["checkpoint"]]],
            ["Ranker conservado", release["files"][release["ranker"]]],
            [
                "150 m, confirmación completa",
                read("range150_confirmation_20261101/INTAKE.json")["checkpoint_sha256"],
            ],
            ["300 m, mejor panel", ranges["selected_checkpoint_sha256"]],
            ["KITTI MonoDGP, mejor panel", kitti["selected_checkpoint_sha256"]],
            ["MonoCoP, mejor panel", mono["selected_checkpoint_sha256"]],
            ["Manifest entrega local", delivery["delivery_manifest_sha256"]],
            ["Filas auditoría geométrica ideal", ideal["rows_sha256"]],
        ],
    )
    state = f"""# Estado, resultados y artefactos eAP: cierre de 1–2 de noviembre de 2026

## 1. Resultado y alcance

El mejor candidato conservado sigue siendo **Dfull RGB + ranker entrenado en TRAIN**:
**{aligned["mAP"]:.8f} mAP** en {aligned["samples"]} imágenes de selección completa alineada,
y **{official["AP"]:.8f} AP** en el envío oficial anterior.
Esta ronda no produjo un sustituto mejor ni un modelo cercano a 1,0 mAP.

La investigación quedó pausada tras la petición «continua y vete acabando».
Terminamos el entrenamiento en curso, cerramos el diagnóstico geométrico y verificamos
la entrega. La petición posterior de documentación y publicación no inicia otra campaña.
El cierre técnico de referencia es el commit {SOURCE_COMMIT}, rama
dev/eap-diagnostics-20261101, repositorio Kripta-Studios/phase-ttc.

Este documento cubre la continuación del handoff publicado en 3b6cfe2f.
Conservamos los documentos del 1 de noviembre como antecedentes; este texto actualiza
sus experimentos pendientes. PHASE-TTC es el nombre público; mantenemos los IDs
históricos y el namespace e_jepa_ttc. La ronda trata **detección 3D RGB en eAP**:
no demuestra una mejora JEPA, de fusión con eventos ni de TTC.

Las tablas se generan desde los JSON archivados con
[build_eap_research_handoff_20261102.py](../../scripts/build_eap_research_handoff_20261102.py).
El [recibo de fuentes](HANDOFF_EAP_20261102.sources.json) conserva sus hashes.
Los enlaces relativos se resuelven dentro del repositorio; las copias de Descargas
conservan el mismo texto para adjuntarlo a otra sesión.

## 2. Trabajo realizado

1. Verificamos el paquete, checkpoint, ranker, fuentes congeladas y perfil de evaluación.
   Reutilizamos las predicciones ya guardadas y comprobamos paridad con el evaluador alineado.
2. Auditamos geometría, focales, resize, tamaños, yaw, targets, filtros de admisión y rango.
   Inspeccionamos proyecciones y el contrato del mapa de profundidad.
3. Diagnosticamos las propuestas por clase, distancia, tamaño proyectado, secuencia,
   duplicados, ranking y sustituciones oracle. Separamos el error de clase del de profundidad.
4. Comparamos hipótesis alternativas de clase, dos cabezas de ranking TRAIN,
   fine-tuning con rangos de 60/150/300 m y combinaciones de dos modelos.
5. Ejecutamos el censo TRAIN hasta 400 m y una prueba semántica RGB pequeña.
   Analizamos clases raras con las propuestas TRAIN ya conservadas.
6. Transferimos pesos KITTI a MonoDGP; adaptamos y verificamos un port MonoCoP,
   transferimos sus pesos multiclase, ejecutamos smoke técnico y screening de calidad.
7. Recuperamos el entrenamiento tras el reinicio del host sin ampliar su presupuesto.
8. Derivamos y probamos una inversión geométrica por vértices. Medimos primero el caso
   de targets perfectos y después su uso con predicciones reales guardadas.
9. Conservamos los resultados negativos, fallos técnicos, checkpoints y recibos.
   Entregamos el mejor paquete existente, con copias y hashes verificados.

## 3. Protocolo y poblaciones

La separación pública de trabajo usa 24 secuencias TRAIN, ocho de selección y ocho
de calibración. El censo contiene {census["frames"]} frames TRAIN.
La selección completa contiene {aligned["samples"]} frames y {aligned["gt_boxes"]} GT.
El panel reducido de los screenings tiene {mono["incumbent_panel"]["samples"]} imágenes.
Los paneles TRAIN de rango y de queries tienen ambos 4.096 imágenes, pero **no son
la misma población**: el de queries incluye las imágenes con clases raras.

El perfil alineado conserva ocho clases en la macro, incluidos los casos sin GT,
y usa todos los GT raw. Evalúa distancia ego XY con umbrales estrictos
0,5/1/2/4 m. La aritmética interpola 101 puntos de recall y aplica el recorte
de precisión/recall 0,1, sin envolvente de precisión. Su SHA-256 es
{aligned["profile_sha256"]}.

La alineación local no certifica equivalencia con el scorer privado.
La selección ya estuvo expuesta al desarrollo; no es un test independiente.
No usamos el score privado para elegir hiperparámetros ni enviamos una nueva variante.
Una AP cero puede coexistir con verdaderos positivos. Las métricas de TRAIN,
panel, selección completa y test oficial deben citar su población.

## 4. Resultados sobre selección completa

Todas las filas de esta tabla usan la misma población raw de selección completa:

{full_table}

La confirmación de 150 m mejora al Dfull sin ranker en
{range150["mAP"] - baseline["mAP"]:+.8f}, pero queda
{range150["mAP"] - aligned["mAP"]:+.8f} respecto al modelo conservado.
Por eso no lo promocionamos. La combinación por rango tampoco mejora al mejor
modelo individual. La expansión de clase y las cabezas lineal/MLP conservan las
geometrías; aumentar las hipótesis no crea nuevos centros 3D.

Resultado por clase del modelo conservado:

{classes}

Fuentes: [alineación](evidence/benchmark_local_alignment_20261101/REPORT.md),
[hipótesis de clase](evidence/multiclass_20261101/REPORT.md),
[rankers](evidence/query_quality_20261101/REPORT.md),
[confirmación 150 m](evidence/range150_confirmation_20261101/REPORT.md) y
[combinación](evidence/range_ensemble_20261101/REPORT.md).
El score oficial anterior procede de [scores.json](evidence/official_benchmark_20261101/scores.json).

## 5. Diagnóstico del modelo original

Reutilizamos sus propuestas completas y sustituimos atributos con GT público:

{oracle_table}

Estas cifras **no son modelos entrenados, mejoras logradas ni cotas superiores
generales**. La sustitución de profundidad depende de una asociación por rayos;
la de clase/tamaño, de asociaciones geométricas; la de ranking no elimina duplicados.
El tamaño no cambia la AP basada en centros, aunque sí puede cambiar el error de escala.

Encontramos problemas de cobertura lejana, confusión de clases y precisión de centros.
La auditoría inicial no justificó intercambiar globalmente dimensiones ni añadir un
desplazamiento bottom-center. Aclaramos que el parámetro de incertidumbre de profundidad
corresponde a una escala Laplace en esa pérdida; no lo presentamos como desviación
estándar calibrada.

El diagnóstico TRAIN de clases raras reutiliza el mismo checkpoint y distingue
matching con clase correcta y matching agnóstico. Las diferencias TRAIN/selección
no aíslan sobreajuste porque cambian identidades y distribución.
Véanse [diagnóstico raw](evidence/raw_diagnostics_20261101/REPORT.md),
[propuestas TRAIN](evidence/train_query_diagnostics_20261101/report/REPORT.md) y
[semántica de escala](DEPTH_SCALE_SEMANTICS_20261101.md).

La prueba semántica CPU encontró {semantic["matched_bicycle_targets"]}/{semantic["projectable_bicycle_targets"]}
coincidencias bicycle en 16 frames TRAIN enriquecidos, con los umbrales declarados.
No es mAP ni recall general; no justificó ampliar esa receta.
Las cajas visibles COCO y las proyecciones de cuboides no son etiquetas equivalentes.

## 6. Rango y la pregunta sobre 400 m

El máximo observado en TRAIN es **{census["raw_center_depth_extrema_m"][1]:.6f} m**
de profundidad óptica; el máximo radial ego XY es
{census["raw_horizontal_range_extrema_m"][1]:.6f} m.
No encontramos anotaciones TRAIN a 400 m.

{census_table}

El último límite es un censo de admisión; **no significa que existan targets a 400 m**.
El rango de 300 m cubre los targets observados, sujeto a visibilidad, proyección,
tamaño mínimo y capacidad de queries. Incluir objetos lejanos válidos requiere
controlar el balance por clase/rango y el descenso de precisión con la distancia.
El límite configurado por sí solo no añade información ni diversidad.

Screenings de rango, una semilla y receta pareada:

{range_table}

El incumbente logra {ranges["incumbent_panel"]["mAP"]:.8f} en ese mismo panel.
El mejor 300 m alcanza {ranges["runs"]["300"]["best"]:.8f}, por debajo del incumbente
y del mejor control 150 m. Cerramos 300 m sin evaluación completa; no inventamos
una cifra de selección completa para ese modelo. El ensayo de 150 m sí tuvo
confirmación completa, mostrada en el apartado 4.

Fuentes: [censo](LONG_RANGE_TRAIN_AUDIT_20261101.md),
[contrato de campaña](LONG_RANGE_CAMPAIGN_20261101.md) y
[cierre 300 m](evidence/range300_screen_20261101/REPORT.md).

## 7. Inicialización externa y MonoCoP

En MonoDGP adaptamos la inicialización KITTI y comprobamos transferencia de filas,
targets vacíos, profundidad, guardado/carga y replay antes del screening.
Su mejor resultado quedó por debajo del control y no pasó a confirmación completa.

Para MonoCoP fijamos fuente y pesos, auditamos el port CPU e implementamos atención
portable, focal vertical, rango, bins de profundidad y taxonomía eAP.
Mapeamos las filas Car/Pedestrian/Cyclist; inicializamos las otras cinco clases.
Mantuvimos el forward sin targets; las etiquetas entran en la pérdida.
Registramos la procedencia y la incertidumbre de admisión/licencia de los pesos,
sin afirmar que el entrenamiento externo sea equivalente al del incumbente.

El smoke GPU de 32 updates solo verifica funcionamiento y aprendizaje local.
El screening posterior empezó desde el checkpoint de transferencia anterior al
smoke, con 2.048 updates, batch 2, 640×384, FP32, TF32 desactivado, seed 7,
7.200 s y límite de 9 GiB asignados. Cambian arquitectura e inicialización;
la comparación no aísla causalmente uno de esos factores.

{model_table}

En el mismo panel, el incumbente obtiene {mono["incumbent_panel"]["mAP"]:.8f}.
MonoCoP termina en {mono["best"]:.8f}, update {mono["selected_update"]};
su diferencia es {mono["delta_incumbent"]:+.8f}.
El gate exigía +0,005 sobre el incumbente y el control Dfull 300 m antes de confirmar.
No lo superó. El resultado con este presupuesto no prueba convergencia ni descarta
todas las recetas más largas.

Los diagnósticos intermedios reutilizan predicciones guardadas y verifican la
paridad raw-GT externa:

{panel_oracle_table}

También aquí, las columnas GT son contrafactuales, no resultados desplegables.
Fuentes: [KITTI](evidence/kitti_screen_20261101/REPORT.md),
[port CPU](evidence/monocop_cpu_port_20261101/report/REPORT.md),
[transferencia](evidence/monocop_transfer_20261101/report/REPORT.md) y
[screen final](evidence/monocop_screen_final_20261102/REPORT.md).

## 8. Hallazgo geométrico y comprobación con predicciones

Derivamos una inversión del pinhole a partir de los ocho vértices rotados del cuboide,
el rayo del centro y las distancias a sus bordes proyectados. Para cada borde
no recortado, la condición de contacto con un vértice determina la profundidad.
Excluimos bordes recortados y conservamos la distinción entre profundidad óptica
y distancia radial.

El censo diagnóstico usa {ideal["frames"]} imágenes TRAIN,
{ideal["raw_targets"]} anotaciones originales y {ideal["admitted_targets"]} targets admitidos.
Con tamaño, yaw y bordes perfectos:

{geometry_table}

El prior altura×fy/altura_en_píxeles introduce un sesgo incluso con targets perfectos.
La inversión comprueba consistencia interna del pinhole; no certifica rectificación
física del sensor ni calidad de predicciones. Los regresores pueden emitir coordenadas
subpíxel, por lo que la sensibilidad a un píxel tampoco es una cota de precisión.

Después evaluamos {pred_receipt["frames"]} imágenes TRAIN y
{pred_receipt["counts"]["queries"]} propuestas **predichas**, sin nueva inferencia.
Reprodujimos la decodificación original exactamente y conservamos clase, score,
tamaño, yaw y pool de propuestas. Las variantes reemplazan profundidad y eliminan
el residual aprendido; la geometría inválida vuelve a la detección original.

{predicted_table}

La inversión supera al prior aislado, pero empeora la salida completa aprendida.
No la incorporamos al candidato. El ensayo no descarta reentrenar un residual
compatible con otro prior; ese reentrenamiento queda sin ejecutar.

Fuentes: [auditoría ideal](evidence/height_prior_geometry_20261102/REPORT.md),
[resultado predicho](evidence/predicted_geometry_20261102/RESULTS.json) y
[contrato predicho](evidence/predicted_geometry_20261102/PLAN.json).

## 9. Fallos técnicos y recuperación

Conservamos los intentos fallidos, sin sobrescribirlos como éxitos:

- El port CPU inicial de MonoCoP falló por la pirámide de features; lo corregimos
  en otra versión y repetimos sus comprobaciones.
- La transferencia y preparación tuvieron fallos de API/estructura de datos;
  los corregimos antes de los screenings admitidos.
- Una llamada a psutil con Path en Windows falló antes de GPU.
- Un sombreado de variable cambió el nombre del checkpoint del primer smoke.
  El siguiente intento conservó correctamente los checkpoints.
- El verificador de replay comparó tuplas en memoria con listas JSON. Corregimos
  la normalización estructural sin tolerancia numérica. Un proceso nuevo recuperó
  el gate técnico sin pasos extra de entrenamiento. El STATUS original sigue FAILED.
- El primer informe KITTI trató un recall ausente como número; preservamos el fallo
  y corregimos el formato, sin alterar métricas.
- La primera auditoría de altura pasó una proyección 3×4 a una función que exigía
  la intrínseca 3×3. Conservamos el intento y corrigimos el bloque que se le entrega.
- El primer intento de combinación se detuvo por un problema técnico. La versión
  compacta reutiliza índices y hashes de propuestas y verificó la reconstrucción.

El host se reinició durante MonoCoP. El STATUS original quedó lleno de bytes cero.
Comprobamos la ausencia del proceso antiguo y el reinicio; cargamos y verificamos
modelo/optimizador en CPU. Reanudamos desde update {recovery["restored_update"]};
el log llegaba a {recovery["last_logged_update"]}. Repetimos
{recovery["replayed_logged_updates"]} updates perdidos, equivalentes a
{recovery["replayed_logged_exposures"]} exposiciones.

El run final contiene {mono["host_restart_recovery"]["logical_exposures"]} exposiciones
lógicas y al menos {mono["host_restart_recovery"]["minimum_executed_exposures_including_lost_work"]}
ejecutadas contando el trabajo perdido. Reservamos {recovery["reserved_consumed_seconds"]} s
del presupuesto original para el tramo anterior al reinicio.
El elapsed comunicado ({mono["status"]["elapsed_seconds"]:.2f} s) incluye esa reserva:
no es tiempo de cómputo exacto. El pico CUDA asignado final fue
{mono["status"]["peak_cuda_bytes"] / 2**30:.3f} GiB, distinto de toda la VRAM del dispositivo.
Los tokens repetidos coincidieron; las losses no fueron bit a bit iguales.
No ampliamos el límite de updates ni el de tiempo.

Véase [RECOVERY.json](evidence/monocop_screen_final_20261102/RECOVERY.json).

## 10. Validación y alcance de las pruebas

{tests_table}

Son recibos focalizados; no deben sumarse como una suite independiente completa.
La prueba CUDA separada cubre el caso omitido en el comando CPU.
También conservamos comprobaciones de Ruff/Pyright, paridad de métricas,
contratos de datos, fuentes congeladas y carga de checkpoints.
No afirmamos que se haya ejecutado todo el CI histórico del repositorio en este cierre.

Verificamos {delivery["delivery_files_verified"]} archivos de entrega por SHA-256.
El manifest del modelo conserva {len(release["files"])} archivos y su comando verify pasó
antes y después de copiar. Los recibos de respaldo incluyen restauración CPU del
checkpoint final; los datos crudos y los pesos externos no se publican en Git.

## 11. Entrega, rutas y hashes

Worktree utilizado:

~~~text
C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/EVOCON_JEPA_Codex_Handoff/phase-ttc-eap-diagnostics
~~~

La entrega local está en:

~~~text
C:/Users/Álvaro Schwiedop/Downloads/PHASE_TTC_ENTREGA_20261102/
  CIERRE.md
  CIERRE.json
  MANIFEST_ENTREGA.json
  modelo/README.md
  modelo/detector/checkpoints/selected.pt
  modelo/ranker.json
  modelo/infer.py
  evidencia/
~~~

El checkpoint conservado es el Dfull previo; no es el nuevo MonoCoP.
Los originales de esta ronda están bajo:

~~~text
D:/PHASE_TTC_RANGE_20261101/
D:/PHASE_TTC_QUERY_QUALITY_20261101/
D:/PHASE_TTC_PRETRAINED_20261101/
D:/PHASE_TTC_COMPETITOR_20261101/monocop_screen_resume_v1/
D:/PHASE_TTC_COMPETITOR_20261101/monocop_screen_report_v1/
D:/PHASE_TTC_GEOMETRY_PRIOR_20261102/
~~~

Los backups en Descargas están bajo PHASE_TTC_BACKUP_20261101:
kitti_init_update2048, monocop_gpu_smoke_recovered, monocop_screen_update512,
monocop_screen_final_20261102 y geometry_diagnostics_20261102.
El último conserva la auditoría ideal y las predicciones geométricas derivadas.
Los recibos finales están referenciados en
[DELIVERY.json](evidence/monocop_screen_final_20261102/DELIVERY.json).

{checkpoint_hashes}

Git contiene código, planes, informes y evidencia resumida. Para volver a ejecutar
inferencia se necesitan el paquete local, el entorno documentado y el dataset.
Los hashes y rutas no equivalen a publicar esos pesos o datos en GitHub.

## 12. Reproducción

Regenerar estos dos documentos desde los JSON versionados:

~~~powershell
python scripts/build_eap_research_handoff_20261102.py
~~~

Verificar el paquete y ejecutar inferencia desde su carpeta modelo:

~~~powershell
python infer.py verify
python infer.py infer --data-root DATA_ROOT --tokens tokens.json --output ../replay
~~~

El entorno probado usa Python 3.11 y PyTorch CUDA; las versiones exactas están en
ENVIRONMENT.json y requirements-tested.txt del paquete. El lector requiere
metadata/calibración y RGB eAP, no etiquetas de objeto en el forward.

Regenerar el cierre MonoCoP, usando un destino nuevo:

~~~powershell
python scripts/build_monocop_screen_result.py --run D:/PHASE_TTC_COMPETITOR_20261101/monocop_screen_resume_v1 --plan D:/PHASE_TTC_COMPETITOR_20261101/monocop_screen_plan_v2/PLAN.json --recovery D:/PHASE_TTC_COMPETITOR_20261101/monocop_recovery_predecessor_v1 --output D:/PHASE_TTC_COMPETITOR_20261101/rebuilt_screen_report
~~~

Regenerar la auditoría geométrica requiere TRAIN y un destino nuevo:

~~~powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python scripts/audit_detection_height_prior.py --dataset E:/eAP_dataset --population D:/PHASE_TTC_RANGE_20261101/screen_config_v1/population.json --output D:/PHASE_TTC_GEOMETRY_PRIOR_20261102/rebuilt_train_audit
python scripts/diagnose_detection_predicted_geometry.py --panel D:/PHASE_TTC_QUERY_QUALITY_20261101/panel --inference D:/PHASE_TTC_QUERY_QUALITY_20261101/campaign/train_inference --dataset E:/eAP_dataset --output D:/PHASE_TTC_GEOMETRY_PRIOR_20261102/rebuilt_predicted_audit
~~~

Esos comandos explican la reproducción; no son una cola pendiente de ejecución.
Conservar los originales evita repetir inferencia o entrenamiento sin motivo.

## 13. Resultado defendible y límites

Entregamos un candidato reproducible, evidencia sobre sus errores y comparaciones
negativas que acotan las siguientes decisiones. No hemos demostrado excelencia
en ocho clases, SOTA, producción, calibración de seguridad ni una mAP próxima a 1.
La calidad de bicycle y other_vehicle sigue siendo insuficiente.
El detector RGB no sustituye los entregables TTC/JEPA de la carta original.

La selección estuvo expuesta a decisiones, el panel no contiene GT de other_vehicle
y los screenings principales usan una semilla. No se han completado una confirmación
independiente, tres semillas de un candidato superior ni una garantía de geometría
física más allá del modelo pinhole auditado. No hubo gasto cloud nuevo.

La continuidad y el prompt de reanudación están en
[{PLAN_NAME}]({PLAN_NAME}).
"""
    plan = f"""# Plan de investigación y mensaje para Codex: estado al 2 de noviembre de 2026

## 1. Situación que debe respetar la siguiente sesión

La ronda terminó a petición del usuario y la investigación quedó pausada.
Este documento prepara una posible reanudación; no autoriza a lanzar entrenamientos
solo por leerlo. La solicitud posterior pidió documentación, git add, commit y push.

El candidato conservado es Dfull RGB + ranker TRAIN:
{aligned["mAP"]:.8f} mAP en selección completa alineada y
{official["AP"]:.8f} AP oficial anterior. MonoCoP terminó en
{mono["best"]:.8f} sobre el panel, cuyo incumbente logra
{mono["incumbent_panel"]["mAP"]:.8f}. No hay un modelo cercano a 1,0.

Leer primero [{STATE_NAME}]({STATE_NAME}).
La implementación y evidencia del cierre llegan a {SOURCE_COMMIT}.
La rama de trabajo es dev/eap-diagnostics-20261101; el repositorio es
Kripta-Studios/phase-ttc. Verificar el HEAD remoto y local al reanudar.

## 2. Qué se consiguió y qué sigue sin resolver

Completamos el diagnóstico raw, la auditoría de rango, las comparaciones de ranking,
la confirmación 150 m, el control 300 m, la transferencia KITTI, el port y screening
MonoCoP, la recuperación tras reinicio y las dos auditorías geométricas.
Conservamos un paquete ejecutable, checkpoints, resultados por clase, fuentes y hashes.

Ninguna variante nueva superó al candidato conservado. El fallo dominante combina
profundidad/centros, cobertura por rango y clasificación de clases raras.
Los oracles orientan el diagnóstico; no representan mejoras ya conseguidas.

La inversión geométrica eliminó el sesgo con targets perfectos, pero su uso directo
con atributos predichos empeoró TRAIN: {predicted["baseline"]["mAP"]:.8f} →
{predicted["cuboid_edges_only"]["mAP"]:.8f}. No trasladar la conclusión del caso
ideal a inferencia ni añadir una corrección al residual de un checkpoint antiguo
sin comprobar qué compensación ya aprendió.

## 3. Experimentos cerrados: no repetir por rutina

| Intervención | Conclusión de esta ronda | Evidencia |
|---|---|---|
| Expansión de clases, top 50 y rankers lineal/MLP | Sin mejora macro suficiente | multiclass_20261101 y query_quality_20261101 |
| 150 m | Mejor que el control sin ranker, inferior al candidato conservado en selección completa | range150_confirmation_20261101 |
| 300 m | Inferior al incumbente y al mejor 150 m en el panel; sin confirmación completa | range300_screen_20261101 |
| Combinación por profundidad y unión | Sin mejora completa | range_ensemble_20261101 |
| Inicialización KITTI MonoDGP | Screening negativo bajo el presupuesto registrado | kitti_screen_20261101 |
| MonoCoP | Screening negativo; arquitectura y pesos cambiaron a la vez | monocop_screen_final_20261102 |
| Inversión geométrica directa | Empeora la salida aprendida en TRAIN | predicted_geometry_20261102 |
| Prueba semántica RGB | Demasiado débil y pequeña para justificar una campaña | semantic_probe_20261101 |

Las carpetas citadas están bajo docs/detection_sota_20261029/evidence.
No generalizar estos negativos a toda temporalidad, eventos, mayor resolución,
otras arquitecturas o recetas más largas: no las hemos descartado en general.
Tampoco ampliar épocas por la sola caída de loss TRAIN.

## 4. Primer trabajo si el usuario reanuda

1. Verificar el worktree, modificaciones ajenas, HEAD y recibos de entrega.
2. Confirmar que no existe un proceso escritor vivo. La presencia de un lock no
   acredita actividad: comprobar PID y fecha de creación.
3. Usar el run final monocop_screen_resume_v1. El STATUS de monocop_screen_v1
   quedó dañado en el reinicio y se preserva como evidencia, no como estado vigente.
4. Verificar los hashes de pesos y del perfil antes de cargar un productor.
5. Reutilizar los JSON y sidecars de predicciones. Repetir inferencia solo si un
   cambio concreto requiere salidas que aún no se guardaron.
6. Declarar una hipótesis falsable, población, control, receta, presupuesto y gate
   antes de reservar GPU. No relajar contratos congelados para hacer pasar un run.

El script de handoff de estado admite --monocop-run para señalar el run final.
Los snapshots antiguos contienen decisiones que entonces estaban pendientes.
No interpretar esas frases históricas como una cola todavía activa.

## 5. Hipótesis priorizadas, aún no ejecutadas

### A. Geometría consistente durante el aprendizaje

La inversión por vértices justifica estudiar otra parametrización, no reemplazar
la profundidad de un checkpoint como si fuera una corrección demostrada.
Antes de una campaña, separar con outputs TRAIN guardados el efecto del error en
tamaño, yaw, centro proyectado y bordes. Examinar recortes, oclusión y dispersión
entre las profundidades inferidas por cada borde.

Una variante entrenable debe declarar cómo reinicializa o conserva el residual,
cómo trata bordes inválidos y qué supervisión geométrica usa. El control debe compartir
datos, inicialización, rango, resolución y presupuesto. Cualquier tensor derivado de
GT pertenece a la pérdida o al diagnóstico, nunca al forward de inferencia.
Primero exigir pérdida finita, gradientes útiles y replay del checkpoint; después,
medir mAP alineada y errores a 0,5/1/2/4 m en el mismo panel.

### B. Clases raras y diversidad de datos

El diagnóstico muestra confusión semántica además de falta de cobertura.
La diferencia TRAIN/selección no distingue por sí sola memorización, distribución
de escenas y ruido. Antes de otro sampler, revisar IDs por secuencia, tamaño,
visibilidad y asociación. No contar ventanas correlacionadas como objetos nuevos.

Un balance nuevo necesita cambiar una causa documentada y mantener ocho clases en
la evaluación. No repetir R2-rare o las mismas cabezas de logits con otro nombre.
No elegir una variante porque mejora una clase rara si deteriora el resultado macro
sin una justificación y un criterio registrados.

### C. Resolución, representación o contexto temporal

La sensibilidad geométrica de cajas pequeñas permite plantear un ensayo de resolución,
si cabe en el presupuesto y conserva la calibración de resize.
Una comparación temporal o con eventos debe separar el valor del contexto del
objetivo JEPA y usar información disponible de forma causal en inferencia.
Los screenings históricos no constituyen un descarte general de esas vías.
No portar simultáneamente otro backbone, pérdida, resolución y sampler si se quiere
atribuir la mejora a un componente.

### D. Duración de entrenamiento

KITTI y MonoCoP alcanzaron su mejor punto al final de sus screenings, pero ambos
quedaron lejos del modelo conservado. Eso no demuestra convergencia ni que ampliar
el entrenamiento sea rentable. Una extensión futura necesita curva por clase,
cobertura suficiente, señal de validación y un presupuesto nuevo declarado.
El buen resultado del smoke sobre dos imágenes no estima calidad ni ETA de una época.

## 6. Objetivos y reglas de decisión

Una mAP de 1,0 es una aspiración, no una expectativa respaldada por estos datos.
Superar 0,10 en selección completa alineada puede servir como objetivo de investigación;
no es un resultado ni una promesa. El primer objetivo operativo sería superar al
mejor candidato con una ganancia defendible y sin depender solo de car.

Para una campaña comparable, predefinir antes de ejecutar:

- Control y población idénticos, score y protocolo raw-GT de ocho clases.
- Cribado con mejora de al menos +0,005 frente al incumbente y al control pertinente.
  Los controles y el umbral deben quedar en el plan; no copiarlos a otra población.
- Confirmación completa de todo candidato que supere el cribado.
- Revisión explícita de caídas de AP por clase mayores de 0,01, GT y TP/FP/recall.
- Incertidumbre por secuencia y confirmación independiente; tres semillas para
  resultados principales cuando una primera receta justifique ese coste.
- Checkpoint elegido por validación, inferencia sin etiquetas, guardado/carga,
  paridad del evaluador y paquete reproducible antes de promoverlo.

El panel actual no contiene GT de other_vehicle. Puede rechazar un candidato
globalmente débil, pero no certificar esa clase. Si se cambia el panel para darle
cobertura, medir de nuevo los controles en él y conservar los resultados anteriores.
No retocar el test ni el perfil después de ver el score oficial.

## 7. Recursos y límites

La ronda se ejecutó localmente en una RTX 5070 Ti Laptop de 12 GiB, con 32 GiB RAM.
El entorno probado usa Python 3.11.15 y PyTorch 2.11.0+cu128.
Verificar el hardware y el entorno actual antes de reutilizar esos datos.
Los presupuestos de 2.048 updates/7.200 s pertenecen al screening cerrado; no son
un saldo abierto para otro experimento.

No existe autorización de gasto cloud nueva en este cierre. No deducir presupuesto
de pods, IPs o acuerdos antiguos. Si el usuario pide cloud, solicitar el límite que
falte una sola vez y preparar antes el experimento concreto.

Medir preparación, forward, matching, backward, optimizador, evaluación y checkpoint.
Usar caché/prefetch solo con pruebas de paridad y consumo de RAM/VRAM.
Evitar procesos GPU competidores. Una observación que vence no acredita que el
entrenamiento haya muerto; consultar el mismo proceso antes de reiniciarlo.

## 8. Artefactos imprescindibles y recuperación

El paquete listo para usar está en Downloads/PHASE_TTC_ENTREGA_20261102.
El candidato conservado y el MonoCoP final son modelos distintos.
Los hashes y rutas exactas están en el documento de estado y en DELIVERY.json.
Los backups en C: conservan checkpoints intermedios/finales y diagnósticos; D:
conserva los runs originales. Git contiene la evidencia y el código, no los pesos
ni los datasets.

Mantener el directorio monocop_recovery_predecessor_v1 y su RECOVERY.json.
Las losses repetidas tras la recuperación no fueron bit a bit iguales; no afirmar
reanudación numérica exacta. El contrato, el estado del optimizador/sampler/RNG y
el orden de tokens sí se conservaron y verificaron.
No sobrescribir el STATUS dañado ni eliminar los fallos técnicos para simplificar
la presentación.

## 9. Reproducción y publicación

Desde la raíz del worktree:

~~~powershell
python scripts/build_eap_research_handoff_20261102.py
~~~

Ese comando usa evidencia versionada y no inicializa CUDA. Escribe ambos Markdown
y HANDOFF_EAP_20261102.sources.json. Los valores de tablas proceden de los JSON.
Copiar ambos Markdown a Descargas para una sesión futura.

La rama de publicación es dev/eap-diagnostics-20261101.
Verificar que el commit remoto coincide con el local después del push.
No inferir un merge a main, una release pública de pesos o una nueva presentación
al benchmark a partir de ese push.

## 10. Estado de las afirmaciones

Podemos defender el paquete, las cifras medidas, el diagnóstico de sesgo del prior,
la mejora de la inversión frente al prior aislado y los resultados negativos de
las variantes probadas. No podemos defender excelencia, SOTA, seguridad para
conducción, generalización independiente ni una mejora TTC/JEPA.

Si una próxima campaña no alcanza su gate, conservar el candidato anterior y
documentar qué se aprendió. Terminar una cola de GPU no prueba éxito científico.
Si el usuario pide cerrar, finalizar el trabajo pendiente y entregar resultados
con sus límites; no seguir entrenando para poder escribir una conclusión positiva.

## 11. Mensaje completo para una sesión futura

Usar este mensaje únicamente cuando el usuario decida reanudar:

~~~text
Continúa la investigación del detector PHASE-TTC para eAP 3D Detection a partir
del cierre de 2 de noviembre de 2026. Quiero mejoras defendibles en ocho clases.

Lee AGENTS.md y estos dos documentos completos:
docs/detection_sota_20261029/ESTADO_RESULTADOS_Y_ARTEFACTOS_EAP_20261102.md
docs/detection_sota_20261029/PLAN_INVESTIGACION_Y_PROMPT_CODEX_EAP_20261102.md

Repositorio: Kripta-Studios/phase-ttc.
Rama de continuidad: dev/eap-diagnostics-20261101.
El cierre de implementación llegó a 543fa399; hay un commit posterior que publica
estos documentos. Verifica HEAD, rama remota y recibos. El worktree utilizado fue:
C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/EVOCON_JEPA_Codex_Handoff/phase-ttc-eap-diagnostics
Conserva cambios ajenos y usa un worktree aislado si la rama actual es otra.

El modelo conservado es Dfull RGB + ranker TRAIN, no MonoCoP. Sus métricas exactas
están en los JSON y en el documento de estado. No se logró un modelo excelente,
no hay resultados cercanos a 1,0 y no hay etiquetas TRAIN a 400 m.
No confundas métricas TRAIN, panel reducido, selección completa y test oficial.

La ronda anterior terminó: 150 m, 300 m, rankers, combinaciones, KITTI y MonoCoP
no sustituyeron al incumbente. La inversión por vértices funciona con targets
perfectos, pero su sustitución directa con predicciones empeora TRAIN respecto
a la salida aprendida. No repitas esos screenings ni apliques la corrección
como si hubiera mejorado un modelo.

Verifica procesos e integridad antes de actuar. El run final MonoCoP es
D:/PHASE_TTC_COMPETITOR_20261101/monocop_screen_resume_v1.
El STATUS del run original quedó dañado tras un reinicio y se preservó.
Los checkpoints y diagnósticos tienen backups verificados en Descargas.
No interpretes un lock antiguo o un timeout como prueba de actividad o de muerte.

Reutiliza predicciones y sidecars ya guardados. Elige una hipótesis nueva según
los errores medidos: geometría aprendida con residual compatible, diversidad
de clases raras, resolución o contexto. Cambia una causa identificada y fija
control, población, receta, presupuesto y criterio de promoción antes de GPU.
Los screenings negativos no descartan toda temporalidad, eventos o recetas largas.

Mantén ocho clases, GT raw y splits por secuencia. No uses etiquetas privadas,
el score test para ajustar hiperparámetros ni GT en el forward de inferencia.
No modifiques fuentes congeladas ni relajes hashes. Un productor nuevo requiere
contrato nuevo y paridad con el evaluador alineado.

No hay gasto cloud autorizado por los acuerdos antiguos. Comprueba el límite
vigente si necesitas un servicio de pago. Mide el tiempo completo y los recursos.
No añadas semillas ni épocas sin una razón y un presupuesto concretos.

Trabaja solo por defecto y conserva pruebas negativas, varios checkpoints,
recibos, configuración, semillas y tablas regenerables desde JSON.
Avanza hasta el siguiente resultado defendible dentro de los recursos autorizados,
con validación y entrega reproducible. No prometas SOTA, producción ni mAP 1,0.
Si el usuario vuelve a pedir cerrar, entrega el mejor candidato verificado y
explica lo que sigue sin resolver.
~~~
"""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in ((STATE_NAME, state), (PLAN_NAME, plan)):
        (args.output_dir / name).write_text(content, encoding="utf-8")
    receipt = {
        "schema": "phase_eap_research_handoff_20261102",
        "implementation_close_commit": SOURCE_COMMIT,
        "inputs": inputs,
        "generator": Path(__file__).relative_to(ROOT).as_posix(),
        "generator_sha256": sha(Path(__file__)),
        "outputs": {name: sha(args.output_dir / name) for name in (STATE_NAME, PLAN_NAME)},
        "new_training": False,
        "new_metrics_computed": False,
    }
    (args.output_dir / "HANDOFF_EAP_20261102.sources.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "documents": receipt["outputs"],
                "sources": len(inputs),
            }
        )
    )


if __name__ == "__main__":
    main()
