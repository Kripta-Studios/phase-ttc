# Extensión a todos los targets observados

El censo TRAIN verificado encontró una profundidad óptica máxima de 249,893 m.
La extensión usa un límite exclusivo de 300 m para incluir todas las anotaciones
que pasan los filtros geométricos actuales. No hay ejemplos TRAIN a 400 m.
Las cifras y tamaños proyectados se regeneran en
[la auditoría de rango](LONG_RANGE_TRAIN_AUDIT_20261101.md).

Se añade un productor separado, sin modificar los bytes de los ensayos 60/150 m
en ejecución. El diff del bucle de entrenamiento solo cambia el tipo de contrato,
el identificador del productor y su inventario de fuentes. La validación del
contrato histórico sigue rechazando rangos distintos de 60/150 m.

La comparación conserva el checkpoint Dfull, la política de reinicialización de
la clasificación de profundidad, las 4.096 imágenes TRAIN, las 512 de selección,
seed 7, batch 2, FP32, LR, scheduler, 2.048 actualizaciones y evaluaciones en
0/512/1.024/1.536/2.048. El límite es 7.200 segundos de ejecución medidos por el
productor, comprobado entre actualizaciones; una evaluación en curso puede
rebasar ese límite. No se amplía automáticamente.

Esta extensión se registró después de observar curvas parciales de 60/150 m.
Es exploratoria y no aporta confirmación independiente. Una mejora del panel
no permite promover el modelo: se exige la selección completa, al menos +0,005
mAP sobre el incumbente y un control válido, con revisión de caídas por clase
mayores de 0,01.

Antes del screening se ejecutan ocho actualizaciones sobre 16 imágenes TRAIN,
incluido un frame con un target a 249,153 m, y 16 imágenes del mismo panel de
selección. El gate exige pérdidas, gradientes y pesos finitos; presencia de un
target admitido por encima de 200 m en un frame efectivamente entrenado;
binning LID válido hasta 299 m; restauración de los pesos guardados; y paridad
exacta de las predicciones iniciales frente al ensayo anterior para esos tokens.
Esta prueba comprueba ejecución, no calidad del modelo.

La cola espera a que el ensayo de rango y la inferencia TRAIN del experimento
de ranking hayan terminado y sus recibos estén verificados. Así no ejecuta dos
procesos GPU simultáneos. Las evaluaciones CPU de ranking pueden continuar.
Si una etapa falla o agota su presupuesto, conserva el fallo y no promueve ni
prolonga el entrenamiento. Las etapas completas pueden verificarse al reiniciar;
la recuperación de una etapa GPU incompleta requiere un segmento nuevo del
productor y revisión de su lock/PID.

Registro y estado real:

- `D:/PHASE_TTC_RANGE_20261101/long_range_config_v1/PLAN.json`
- `D:/PHASE_TTC_RANGE_20261101/long_range_campaign_v1/QUEUE.json`
- `D:/PHASE_TTC_RANGE_20261101/long_range_campaign_v1/QUEUE.status.json`

El registro inicial y la cola se conservan en
`evidence/long_range_20261101/`. No implican que el smoke o el screening
hayan concluido. No se autoriza gasto cloud ni se leen etiquetas privadas.
