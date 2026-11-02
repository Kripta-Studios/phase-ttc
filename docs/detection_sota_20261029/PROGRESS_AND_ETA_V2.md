> **Instantánea histórica anterior al cierre de R0/R2; no es el estado actual.**
> Se conserva como registro de seguimiento. Para resultados posteriores, véanse
> [la decisión R0/R2](R0_R2_SEED7_DECISION.md) y
> [el estado del 2 de noviembre](ESTADO_RESULTADOS_Y_ARTEFACTOS_EAP_20261102.md).

# Estado y ETA de la continuación S1–S6

Estado: **S1 en recuperación; ETA temporal total todavía no defendible**.

## Estado retenido

- El camino científico durable llega a la actualización **3840** de **8192**; quedan **4352 actualizaciones**.
- El candidato seleccionado sigue siendo el de la actualización **3072**, con AP101 de desarrollo **0.07669187**. No es una puntuación oficial.
- La cola posterior (228 actualizaciones, 912 exposiciones) está contabilizada pero descartada: no pertenece al estado del optimizador retenido y no reduce el trabajo restante.
- Dos admisiones limpias de 16 actualizaciones (`old_A` y `old_B`) terminaron. Son comprobaciones de ejecución, no una comparación causal de velocidad entre hosts.
- La suite estricta fue interrumpida externamente por LogonUI. El siguiente paso autorizado es solo recuperar la paridad; aún no hay una tasa definitiva del host de recuperación.
- El perfil FP32 por lotes conserva paridad exacta con batch 1, pero batch 2 y 4 incumplen la tolerancia preregistrada. La recomendación retenida es `REJECT_BATCHED_EVALUATION`.

## ETA

No se publica una cifra en horas. La ETA total permanece `UNKNOWN_PENDING_PARITY_RECOVERY_RATE`: usar tiempos del host anterior o la velocidad bruta de batch 2/4 mezclaría entornos o una ruta numéricamente no equivalente.

La próxima estimación condicional de S1 solo se calculará tras: (1) paridad del recovery en batch 1, (2) enlace al checkpoint durable de update 3840 y (3) una ventana pasiva preregistrada de continuación limpia. S2–S6 seguirán sin horas hasta conocer los resultados que deciden sus ramas.

## Lo que falta

| Fase | Estado | Trabajo pendiente |
|---|---|---|
| S1 | `IN_PROGRESS_RECOVERY_GATED` | reanudar en update 3840, llegar a 8192 y completar evaluaciones y gates de confirmación |
| S2 | `PREREGISTERED_NOT_FIT` | ajustar y comparar una intervención de ranking de pool fijo tras congelar el baseline S1 |
| S3 | `ENGINEERING_READY_EXPERIMENT_NOT_RUN` | ejecutar historia RGB causal y controles de capacidad/historia frente a S1 |
| S4 | `FEASIBILITY_READY_EXPERIMENT_NOT_RUN` | ejecutar la matriz RGB/eventos y controles causales negativos después de S3 |
| S5 | `PARTIAL_ENGINEERING_NO_GENERALIZATION_CLAIM` | completar evaluación multiobjeto con entradas predichas y el gate TTC independiente |
| S6 | `NOT_STARTED` | confirmación multisemilla, calibración, comparación final congelada y revisión |

## Reproducción

```powershell
python scripts/build_detection_progress_status.py --evidence-root <CAMPAIGN_EVIDENCE_ROOT>
```

El JSON generado enlaza cada fuente pequeña por SHA-256. El proceso no abre checkpoints, no recorre datasets y no ejecuta benchmarks.
