> **Instantánea histórica anterior al cierre de R0/R2; no es el estado actual.**
> Se conserva como registro de seguimiento. Para resultados posteriores, véanse
> [la decisión R0/R2](R0_R2_SEED7_DECISION.md) y
> [el estado del 2 de noviembre](ESTADO_RESULTADOS_Y_ARTEFACTOS_EAP_20261102.md).

# Estado y ETA de la continuación S1–S6

Estado: **S1 en recuperación; ETA temporal total todavía no defendible**.

## Estado retenido

- R0 con seed 7 llega de forma durable a la actualización **3840** de **8192**; le quedan **4352 actualizaciones**. Esta cifra no representa todo S1.
- El candidato seleccionado sigue siendo el de la actualización **3072**, con AP101 de desarrollo **0.07669187**. No es una puntuación oficial.
- La cola posterior (228 actualizaciones, 912 exposiciones) está contabilizada pero descartada: no pertenece al estado del optimizador retenido y no reduce el trabajo restante.
- Ambas admisiones limpias de 16 actualizaciones (`old_A` y `old_B`) terminaron. Son comprobaciones de ejecución, no una comparación causal de velocidad entre hosts. Después, el harness rechazó la condición de recursos al detectar un proceso gráfico; no hubo un stop externo de esos runs.
- La recuperación mantiene batch 4 para entrenamiento y batch 1 para evaluación. En el perfil de evaluación, batch 1 conserva paridad exacta, mientras batch 2 y 4 incumplen la tolerancia preregistrada; por eso no se adoptan para evaluación.

## ETA

No se publica una cifra en horas. La ETA total permanece `UNKNOWN_PENDING_PARITY_RECOVERY_RATE`: usar tiempos del host anterior o la velocidad bruta de batch 2/4 mezclaría entornos o una ruta numéricamente no equivalente.

La próxima estimación condicional de R0/seed 7 solo se calculará tras: (1) recovery de entrenamiento B4 con evaluación B1, (2) enlace al checkpoint durable de update 3840 y (3) exactamente 100 updates reanudados. Esa ventana informará por separado las medias de preparación, cómputo y evaluación. Todavía no existen esas observaciones. S1 completo incluye además R1/R2, recetas posteriores y confirmación con tres semillas; S2–S6 seguirán sin horas hasta conocer los resultados que deciden sus ramas.

## Lo que falta

| Fase | Estado | Trabajo pendiente |
|---|---|---|
| S1 | `IN_PROGRESS_RECOVERY_GATED` | terminar R0 seed 7; ejecutar R1/R2 y las recetas posteriores admitidas; confirmar el resultado principal con tres semillas |
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
