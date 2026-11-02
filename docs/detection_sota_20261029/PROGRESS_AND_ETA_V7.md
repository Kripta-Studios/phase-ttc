> **Instantánea histórica anterior al cierre de R0/R2; no es el estado actual.**
> Se conserva como registro de seguimiento. Para resultados posteriores, véanse
> [la decisión R0/R2](R0_R2_SEED7_DECISION.md) y
> [el estado del 2 de noviembre](ESTADO_RESULTADOS_Y_ARTEFACTOS_EAP_20261102.md).

# Estado y ETA de la continuación S1–S6

Estado: **R0/seed 7 continúa con el runtime original; ETA temporal total todavía no defendible**.

## Estado retenido

- R0 con seed 7 llega de forma durable a la actualización **3840** de **8192**; le quedan **4352 actualizaciones**. Esta cifra no representa todo S1.
- El candidato seleccionado sigue siendo el de la actualización **3072**, con AP101 de desarrollo **0.07669187**. No es una puntuación oficial.
- La cola posterior (228 actualizaciones, 912 exposiciones) está contabilizada pero descartada: no pertenece al estado del optimizador retenido y no reduce el trabajo restante.
- Ambas admisiones limpias de 16 actualizaciones (`old_A` y `old_B`) terminaron. Son comprobaciones de ejecución, no una comparación causal de velocidad entre hosts. Después, el harness rechazó la condición de recursos al detectar un proceso gráfico; no hubo un stop externo de esos runs.
- La recuperación mantiene batch 4 para entrenamiento y batch 1 para evaluación. En el perfil de evaluación, batch 1 conserva paridad exacta, mientras batch 2 y 4 incumplen la tolerancia preregistrada; por eso no se adoptan para evaluación.
- El runtime optimizado fue rechazado por el gate de paridad numérica en `loss_depth_map` (error relativo en el techo 2). No se adopta.
- El fallback original fue lanzado desde el estado durable de update 3840. Lanzado no significa aprobado: el run sigue incompleto y no existe un receipt final PASS.

## ETA

No se publica una ETA total en horas. El estado es `UNKNOWN_PENDING_EVALUATION_AND_REMAINING_EXPERIMENTS`: usar tiempos del host anterior o la velocidad bruta de batch 2/4 mezclaría entornos o una ruta numéricamente no equivalente.

La ventana 3841–3940 está completa, pero coincide con el SHA-256 H5: 268.107 s agregados, con medias 2.326535 s de preparación y 0.354531 s de cómputo por update. Se conserva como observación contaminada por I/O y no se extrapola.

El SHA terminó con exit 0 a las 23:15:26 locales (24/24 H5; 321,074 s). La ventana limpia quedó fijada en **3901–4000** antes de existir.

La ventana limpia 3901–4000 ya está completa: 61.743 s agregados; media de preparación 0.316047 s/update y media de cómputo 0.301384 s/update. Aplicar solo esa media a las 4192 actualizaciones posteriores da 0.719 h de componentes de entrenamiento. No es una ETA total: excluye evaluación y overhead de pared no medido.

La selección completa de 23.958 frames puede dominar el coste y todavía no se ha medido en este runtime. Por ello la ETA total continúa desconocida. S1 incluye además R1/R2, recetas posteriores y confirmación con tres semillas; S2–S6 siguen sin horas.

## Lo que falta

| Fase | Estado | Trabajo pendiente |
|---|---|---|
| S1 | `ORIGINAL_FALLBACK_RUNNING_UNVERIFIED` | terminar R0 seed 7; ejecutar R1/R2 y las recetas posteriores admitidas; confirmar el resultado principal con tres semillas |
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
