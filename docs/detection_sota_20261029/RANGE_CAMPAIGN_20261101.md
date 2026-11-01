# Ensayo de rango: productor nuevo con validación raw

## Estado

Se implementó y admitió el screening pareado de 60/150 m que motivó
`NEXT_INTERVENTION_DECISION_20261101.md`. Está en ejecución local. No hay
promoción ni afirmación de excelencia. El candidato RGB ranked sellado sigue
siendo la referencia válida; se conserva sin modificar.

## Comparación fijada antes de observar las curvas

- Semilla 7, mismas 4.096 imágenes TRAIN y 512 imágenes de selección, muestreo
  uniforme sin reemplazo, split histórico por secuencia.
- 2.048 updates, batch 2, 4.096 exposiciones por brazo. Validaciones en
  0/512/1024/1536/2048, selección por mAP raw de ocho clases.
- RGB 640×384, 50 queries, 11 grupos durante entrenamiento, FP32, sin augmentation.
- AdamW, LR 5e-5, backbone ×0,1, weight decay 1e-4, warmup/coseno 5%, suelo 5%.
- Límite de 7.200 segundos por brazo, incluyendo preparación y validaciones
  desde la construcción del modelo. Una parada por tiempo no equivale a una
  comparación con igual número de exposiciones ni permite promoción automática.
- Checkpoint Dfull como inicialización de pesos; optimizador, scheduler,
  muestreador y RNG nuevos. Reinicialización idéntica de la convolución de bins
  en ambos brazos. Embeddings de posición física 0–60 m conservados; filas
  nuevas de 61–150 m inicializadas repitiendo la última fila física conocida.
- La admisión de targets, el predictor y la pérdida DDN comparten el rango.
  La forma de los 80 bins no cambia, pero sí su significado en el brazo 150 m.
  El estudio no es una ablation exclusiva del filtro de targets.
- Scoring sigmoid de clase en ambos brazos, sin reajustar el ranker histórico.

Las configuraciones, admisión, hashes y auditoría de smoke están en
`evidence/range_campaign_20261101/`. La población completa se conserva en
`D:/PHASE_TTC_RANGE_20261101/screen_config_v1/population.json` y se regenera con
el script de preparación. Los JSON son la fuente de las cifras.

## Validación y límites

El nuevo checkpoint tiene esquema `phase_range_checkpoint_v1`, contrato de
fuentes/configuración y guardado atómico. La reanudación restaura el estado
completo y rechaza contratos distintos. El test CPU verifica el siguiente
update exacto. Los smoke reales CUDA verificaron forward/loss/backward,
checkpoint y continuación 8→16 updates. Sampler, calendario y RNG al update 16
coincidieron con la ejecución continua; los pesos CUDA presentaron una pequeña
diferencia numérica registrada en `SMOKE_AUDIT.json`. No se promete igualdad
bit a bit en GPU. El primer fallo por `depth_min` YAML de tipo texto permanece
registrado; la conversión explícita a float lo corrigió antes del screening.

La extracción de GT conserva todas las anotaciones raw, incluidas las que el
entrenamiento no admite. El panel de 16 frames reprodujo exactamente el informe
completo calculado desde los GT externos guardados. La comparación de bins usa
las funciones reales extraídas por AST: 10.006 profundidades, modos UD/LID/SID,
paridad exacta a 60 m y admisión de puntos lejanos a 150 m.

La comprobación se amplió a las 512 imágenes: los informes completos de los
updates 0 y 512 coinciden exactamente con su reevaluación usando GT externos
guardados. El informe inicial de 60 m también coincide exactamente con el
detector congelado. En este MonoDGP, `weighted_depth` se calcula, pero el
embedding posicional interpolado no se suma a las features y la profundidad
final es el prior geométrico de altura más un residual. El clasificador de bins
participa en la pérdida auxiliar; cambiarlo no equivale a cambiar directamente
el centro 3D en inferencia. No se activó ninguna rama comentada del modelo.

El perfil local está ligado a su SHA-256 y a la configuración observada del
benchmark. La equivalencia con el scorer privado sigue sin certificarse.
Las ocho secuencias de selección ya fueron utilizadas durante desarrollo;
esta campaña no crea un test independiente. No se leen etiquetas de test.
El panel uniforme de 512 frames no contiene `other_vehicle`; la selección
completa sí contiene 153 anotaciones. Se conserva AP=0 para la clase sin soporte
en el panel y no se interpreta esa ausencia como una validación de ocho clases.

La API pública de la competición 17291 se verificó solicitando JSON: confirma
ocho clases y umbrales 0,5/1/2/4 m; el leaderboard devolvió cero entradas
visibles. No proporciona una cota competitiva. Los términos permiten investigar
en el benchmark y remiten a la licencia de Hugging Face; el README y el listado
raíz consultados no declaran una licencia explícita. No se admitieron nuevos
datos externos ni se descargaron artefactos privados. El recibo
`BENCHMARK_PUBLIC_STATUS.json` conserva enlaces y hashes.

## Cola y rendimiento

`run_detection_range_queue.py` ejecuta un único proceso GPU, conserva logs y
recibos, omite trabajos ya completos con el mismo contrato y reanuda segmentos
solo desde estados compatibles. Un lock existente exige verificar el PID antes
de recuperarlo. La cola no extiende presupuestos ni promociona modelos.

El screening usa caché compacta con verificación exacta de imágenes normalizadas.
La lectura inicial de TAR se midió como un cuello adicional: el panel abarca
78 TAR TRAIN y 29 de selección. Un perfil mostró interpretación de cabeceras
TAR en el hilo de preparación mientras el hilo principal calculaba la loss.
No se alteró el productor durante la ejecución. El evaluador completo nuevo
reutiliza el lector indexado existente y verificará su paridad antes de usarse
para una promoción. Su validación GPU está pendiente mientras entrena la cola.

El censo TRAIN se pausó brevemente para medir competencia de CPU y se reanudó
con prioridad inferior. La pausa no demostró una ganancia sostenida. Este cambio
de planificación del sistema no modifica datos, pesos ni hiperparámetros.

## Decisión posterior al screening

Confirmar los checkpoints seleccionados concretos en las 23.958 imágenes de
selección. Exigir al menos +0,005 mAP frente al control, revisar caídas >0,01 AP
por clase, cobertura lejana y soporte por instancia. Para reemplazar el paquete
sellado se requiere superar también su resultado comparable, no basta vencer
a un control degradado por la reinicialización de bins. Conservar resultados
negativos y no lanzar semillas adicionales sin una mejora que confirmar.

El censo de clases/instancias y su unión con targets admitidos se auditan por
separado. `sequence + instance_id` es un ámbito de identidad, no certificación
de adquisiciones independientes. Una colisión de clase/centro invalida la unión
del censo hasta resolverla.

## Reproducción

Desde este worktree, con el entorno existente:

```powershell
$python = '../phase-ttc-main-integration/.venv/Scripts/python.exe'
& $python scripts/prepare_detection_range_campaign.py `
  --release D:/PHASE_TTC_RELEASE_20261101/dfull-rgb-ranked-seed7 `
  --dataset E:/eAP_dataset --output D:/PHASE_TTC_RANGE_reproduction/config `
  --train-count 4096 --selection-count 512 --updates 2048 `
  --evaluation-interval 512 --maximum-seconds 7200
& $python scripts/admit_detection_range_panel.py `
  --config D:/PHASE_TTC_RANGE_reproduction/config/range150.json `
  --output D:/PHASE_TTC_RANGE_reproduction/config/ADMISSION.json
& $python scripts/train_detection_range.py `
  --config D:/PHASE_TTC_RANGE_reproduction/config/range60.json `
  --output D:/PHASE_TTC_RANGE_reproduction/range60
```

Ejecutar el brazo 150 con su configuración y otro directorio. La reanudación
requiere `--resume <segmento>/checkpoints/last.pt`, el `selected.pt` adyacente y
un directorio de salida nuevo. El evaluador acepta un checkpoint concreto:

```powershell
& $python scripts/evaluate_detection_range.py `
  --checkpoint D:/PHASE_TTC_RANGE_reproduction/range60/checkpoints/selected.pt `
  --output D:/PHASE_TTC_RANGE_reproduction/evaluation60
```

El evaluador usa inferencia sin anotaciones y guarda shards verificables antes
de resumir la métrica. `--panel-only` sirve para la comprobación de paridad.
El script se debe validar antes de confiar en sus resultados; escribirlo no
equivale a haber completado esa comprobación.
