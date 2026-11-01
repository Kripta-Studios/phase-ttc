# Comparación de inicialización MonoDGP

Los dos rankers por clase terminaron por debajo del incumbente. Sus resultados
por clase se regeneran en [el informe](evidence/query_quality_20261101/REPORT.md).
La investigación continúa con los targets de rango corregidos y una comparación
acotada de la inicialización externa admitida en
[el recibo técnico](PRETRAINED_MONODGP_INTAKE_20261101.md).

El nuevo productor conserva las 4.096 imágenes TRAIN, las 512 de selección,
el rango de 300 m, la resolución 640×384, seed 7, batch 2, FP32, LR 0,00005,
AdamW, scheduler y 2.048 actualizaciones del control Dfull a 300 m. Solo cambia
la procedencia de los pesos, con las adaptaciones obligatorias de cabezas:
Car KITTI corresponde a car eAP; las otras siete clases, velocidad y clasificador
de profundidad se inicializan según la política explícita. No se importa estado
del optimizador upstream. Los módulos históricos siguen sin modificarse.

La comparación mide el efecto de esa inicialización con igual presupuesto
adicional. No iguala el entrenamiento anterior: Dfull consumió 281.116 exposiciones
eAP y el checkpoint KITTI procede de supervisión Car-only. Una derrota en este
screening no descarta toda transferencia externa ni una receta más larga.

Antes de entrenar, el runner verifica la conversión segura y los hashes de pesos,
configuración y fuentes. Espera el fin de la confirmación 150 m y de la cola 300 m.
Ejecuta después 32 actualizaciones sobre dos imágenes TRAIN repetidas, una con
un target por encima de 200 m. Exige pérdida media menor en las últimas ocho
actualizaciones que en las primeras ocho, valores finitos, restauración de pesos
y reproducción exacta de predicciones y métricas del último checkpoint en las
16 imágenes de selección del smoke. Este sobreajuste no cuenta como calidad.

Solo al pasar esos controles inicia el ensayo de 2.048 actualizaciones, con
validación alineada en 0/512/1.024/1.536/2.048 y límite de 7.200 segundos comprobado
entre actualizaciones. Se conservan seleccionado y último. No se añaden épocas
o semillas automáticamente. El máximo de 300 m cubre el TRAIN observado hasta
unos 250 m; no demuestra capacidad a 400 m.

La regla registrada exige +0,005 mAP de ocho clases frente al control y al
incumbente comparable en el panel antes de confirmar el checkpoint en selección
completa. La confirmación debe mantener esa ganancia y revisar caídas por clase
superiores a 0,01. No es test independiente, no hay promoción automática y no se
usan etiquetas privadas. Las métricas del screening se conservan en los JSON
de evaluación; no constituyen confirmación completa de esta variante.

Artefactos locales:

- `D:/PHASE_TTC_PRETRAINED_20261101/comparison_config_v1/PLAN.json`
- `D:/PHASE_TTC_PRETRAINED_20261101/comparison_campaign_v1/QUEUE.json`
- `D:/PHASE_TTC_PRETRAINED_20261101/comparison_campaign_v1/QUEUE.status.json`

La cola admite un único proceso GPU. Los tests verifican mapeo de clases,
reproducibilidad de los nuevos pesos, conservación de bins físicos y rechazo de
contratos o artefactos alterados. El ensayo está autorizado únicamente en recursos
locales; no abre trabajos cloud ni realiza envíos oficiales.

## Inicio verificado tras cerrar 300 m

El control 300 m completó su presupuesto y no mejoró a las referencias del panel;
véase el [cierre de rango](evidence/range300_screen_20261101/REPORT.md).
La cola KITTI pasó las 32 actualizaciones del smoke, el replay exacto del último
checkpoint en 16 imágenes y el gate de aprendizaje. Se verificó una reducción
de la pérdida media, valores finitos y presencia efectiva del target lejano.
Los valores originales y los hashes están en
[SMOKE_GATE.json](evidence/pretrained_gpu_smoke_20261101/SMOKE_GATE.json).

El screening completó las 2.048 actualizaciones con su presupuesto original.
`build_detection_continuation_status.py` incorpora las evaluaciones terminadas
en el documento de continuidad de Descargas. Distingue las 16 imágenes del smoke
de las 512 del panel principal. Los recibos del smoke no acreditan calidad de
detección ni autorizan épocas o semillas adicionales.

## Cierre negativo

El [informe regenerado](evidence/kitti_screen_20261101/REPORT.md) conserva todas
las evaluaciones y la comparación con las referencias del mismo panel. No se
cumplió la regla de avance a selección completa. No se añaden épocas ni semillas.
El checkpoint seleccionado, su estado de entrenamiento y la evidencia tienen
una copia local adicional verificada por SHA-256 y carga CPU.

La AP cero de una clase no implica ausencia de aciertos. El informe incluye
TP, FP, recall y precisión usada por la fórmula alineada: algunas clases
producen aciertos, pero acompañados de demasiados falsos positivos. Estos datos
motivan estudiar la arquitectura y la inicialización multiclase de MonoCoP;
no demuestran por sí solos que el cambio vaya a mejorar eAP.
