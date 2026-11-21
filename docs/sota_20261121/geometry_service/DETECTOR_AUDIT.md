# Auditoría de disponibilidad del detector + tracker

Fecha: 2026-11-21. Esta auditoría responde únicamente si existe un productor verificable que pueda añadirse al ensayo de servicio de cajas suministradas.

## Material localizado

- Integración causal: `operational/detection_phase_service/ttc_panel_adapter.py`, SHA-256 `025e298c…af8b`.
- Replay detector → tracker → PHASE: `scripts/replay_detection_phase_service.py`, SHA-256 `d7aa888c…66af`.
- Checkout del detector: `phase-ttc-detection-local@24b8f36f6e4aa17c965f9526d8cf486b512a8046`, con `output/` sin seguimiento.
- Checkpoint piloto: `output/portable_experiments/pilot256_seed7_velocity_v1/checkpoint.pt`, 498.531.791 bytes, SHA-256 `dc0ce304…e338`.
- Recibo de inferencia: `post_pilot_inference_execution.json`, SHA-256 `4c624476…01e3`.
- Demo integrada: `integrated_prediction_pilot_v1.json`, SHA-256 `63968673…3cd2`.

El adaptador verifica hashes del candidato, manifiesto eAP, eventos y miembros RGB; impone intervalos causales contiguos y conserva IDs generacionales del tracker. Por tanto existe una implementación auditable del enlace.

## Por qué no es una celda comparable

El checkpoint fue un piloto expuesto a desarrollo sobre dos secuencias de entrenamiento y dos de selección. Su puente temporal usa siete frames y cuboides 3D monoculares proyectados al plano de eventos, no las mismas cajas suministradas ni la misma población de ofertas del servicio geométrico. El recibo reporta medianas de forward del detector de 46,73–47,89 ms en ejecuciones separadas; excluye el servicio completo y no es un reloj deadline-a-salida a 50 Hz. La demo integrada mide 0,717 s de runtime PHASE CPU preparado y excluye detector, preparación y carga.

No existe en el material localizado una configuración `detected_phase_panel_adapter_config_v1` que selle este checkpoint y los 100 instantes físicos del ensayo, ni un replay continuo de 3.000 ofertas idénticas. Ejecutarlo como si fuera la tercera política de cajas mezclaría población, modalidad, cadencia, historia y reloj.

## Decisión

Se conserva como implementación en protocolo distinto y coste descriptivo del productor, no como comparador emparejado. Para incorporarlo harían falta: congelar un checkpoint no ajustado a la cohorte, materializar predicciones para exactamente las mismas ofertas, fijar la latencia de disponibilidad por frame, ejecutar detector+tracker+TTC dentro del mismo reloj y publicar cobertura/drops por oferta. No se inventa un coste total a partir de los tiempos parciales existentes.
