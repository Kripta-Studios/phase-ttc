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

## Orden actualizado tras cerrar 150 m

El plan original de 60/150 m exige confirmar los checkpoints seleccionados
en selección completa. Se prioriza esa confirmación antes del ensayo 300 m.
El marcador `long_range_campaign_v1/STOP` y el recibo
`DEFERRED_FOR_CONFIRMATION.json` postergan la cola sin editar sus fuentes:
el runner registra `FAILED: Stop requested before next stage` al liberar
su espera. Es una parada operativa solicitada, no un fallo científico del modelo.

`run_detection_range_confirmation.py` espera esa parada y el fin de la
inferencia TRAIN. Comprueba primero paridad exacta de predicciones y métricas
en las mismas 512 imágenes; solo entonces evalúa las 23.958 de selección.
Al completar, verifica que el marcador STOP es exactamente el suyo, lo retira
y reinicia la cola original 300 m. Ante cualquier discrepancia conserva la parada.
Contrato y progreso: `D:/PHASE_TTC_RANGE_20261101/range150_confirmation_v1/`.

La paridad del evaluador ya pasó en las 512 imágenes: coinciden todas las
predicciones y el diccionario completo de métricas. El checkpoint seleccionado
de 150 m corresponde al update 1.024, con mAP 0,06586902 en ese panel; el
incumbente con ranking obtiene 0,06533347 y el control 60 m, 0,06035564.
Esto justifica confirmar según el plan original, pero no demuestra mejora en
selección completa ni cumple por sí solo la regla más exigente del nuevo 300 m.

El control 60 m seleccionó el update cero. `verify_detection_range_control_reuse.py`
verifica igualdad de todos los pesos salvo dos tensores de clasificación de
profundidad auxiliar, que no alimentan las detecciones decodificadas. Verifica
también las fuentes, el decoder y la igualdad exacta de las 512 predicciones con
los shards históricos. Se reutiliza por ello su evaluación completa ya guardada,
mAP 0,05701726 en 23.958 imágenes. Esta equivalencia solo cubre las detecciones:
los logits auxiliares y el entrenamiento sí difieren. Recibo:
`evidence/range_campaign_20261101/CONTROL_REUSE.json`.
