# Calidad por clase de las queries congeladas

La expansión a ocho clases conserva geometría útil que el argmax descarta,
pero no supera al incumbent ranked en selección pública. El informe se regenera
con `scripts/build_detection_multiclass_report.py`; ambas recetas y sus resultados
se conservan en `evidence/multiclass_20261101/`. La expansión a 400 propuestas y
el control de 50 salidas se registraron antes de evaluar sus métricas finales.
No se promociona ninguna de estas dos variantes.

## Experimento siguiente, todavía sin resultados de validación

`QUERY_QUALITY_PLAN_20261101.json` fija dos cabezas sobre las 16 features ya
expuestas por el detector: una lineal y una MLP con 64 unidades y GELU. El
detector, sus cajas y el ranker histórico permanecen congelados. Cada cabeza
produce ocho puntuaciones de calidad relativa; no son probabilidades calibradas.

El panel contiene 4.096 imágenes de TRAIN. Incluye las 799 imágenes de TRAIN
con bicicletas u otros vehículos y completa el presupuesto mediante muestreo
sin reemplazo con semilla 7. Conserva todas las anotaciones raw de cada imagen.
La población difiere de la usada por el ranker histórico; una mejora no aislaría
el efecto de la arquitectura frente al cambio de población.

Los targets de calidad se calculan por clase a partir de la distancia XY al GT
más cercano: fracción de umbrales 0,5/1/2/4 m que satisface estrictamente.
Las queries duplicadas pueden recibir targets positivos. La evaluación mantiene
el matching uno a uno y penaliza los duplicados; esta limitación queda explícita.
Las cabezas se ajustan durante ocho épocas fijas sobre TRAIN, sin seleccionar
épocas ni normalización usando selección.

## Admisión y ejecución

La inferencia usa los módulos congelados verificados y un lector RGB indexado
nuevo, sin anotaciones. Antes de procesar TRAIN, exige igualdad exacta de las
predicciones y tensores de las queries en 16 imágenes ya evaluadas. Una diferencia
detiene la cola. No se relajan hashes ni se modifica la release histórica.

La cola espera a que la campaña emparejada de rango termine y libere su lock.
Después ejecuta paridad, inferencia TRAIN, ajuste de las dos cabezas y evaluación
completa de ambas en las 23.958 imágenes públicas de selección. Usa una GPU a
la vez; las cabezas y la evaluación de las propuestas guardadas usan CPU.

Rutas locales de esta ejecución:

- Panel y GT TRAIN: `D:/PHASE_TTC_QUERY_QUALITY_20261101/panel/`.
- Contrato/estado de la cola: `D:/PHASE_TTC_QUERY_QUALITY_20261101/campaign/QUEUE*.json`.
- Artefactos por etapa: `parity/`, `train_inference/`, `heads/`, `evaluation_<arm>/`.
- Recetas anteriores: `D:/PHASE_TTC_MULTICLASS_20261101/{v1,top50_v1}/`.
- Código exacto de cada replay anterior: su subdirectorio `code/` y `CODE_ARCHIVE.json`.

Las versiones archivadas de los replays anteriores son necesarias para reanudar
sus contratos exactos: cambios posteriores de tipos, cardinalidad o recibos
producen hashes distintos aunque una rama matemática conserve el resultado.

Para repetir la campaña en directorios nuevos:

```powershell
$python = '../phase-ttc-main-integration/.venv/Scripts/python.exe'
& $python scripts/prepare_detection_query_panel.py `
  --release D:/PHASE_TTC_RELEASE_20261101/dfull-rgb-ranked-seed7 `
  --dataset E:/eAP_dataset `
  --shards D:/PHASE_TTC_LOCAL_CONTINUATION_20261101/ranking_local_v1/shards `
  --plan docs/detection_sota_20261029/QUERY_QUALITY_PLAN_20261101.json `
  --output D:/PHASE_TTC_QUERY_reproduction/panel
& $python scripts/run_detection_query_queue.py `
  --panel D:/PHASE_TTC_QUERY_reproduction/panel/PANEL.json `
  --plan docs/detection_sota_20261029/QUERY_QUALITY_PLAN_20261101.json `
  --range-queue D:/PHASE_TTC_RANGE_20261101/screen_config_v1/QUEUE.json `
  --shards D:/PHASE_TTC_LOCAL_CONTINUATION_20261101/ranking_local_v1/shards `
  --aligned D:/PHASE_TTC_BENCHMARK_ALIGNED_20261101/verified `
  --output D:/PHASE_TTC_QUERY_reproduction/campaign
```

Cada etapa rechaza inputs alterados. Las etapas de inferencia/evaluación guardan
shards reanudables. El ajuste de cabezas es corto y exige un destino nuevo; un
fallo parcial requiere conservar ese intento y admitir explícitamente otra
ejecución. La cola verifica los recibos de etapas ya terminadas al reanudarse.

No hay gasto cloud nuevo ni acceso a etiquetas privadas. La selección está
expuesta y las dos cabezas se compararán de forma declarada. La promoción exige
el margen fijado frente al incumbent, revisión por clase y productor ejecutable;
no equivale por sí sola a un modelo excelente ni a SOTA.
