# Protocolo ejecutado: servicio CPU de geometría suministrada

Fecha: 2026-11-21. Este protocolo materializa la parte CPU-first de P5 del plan de continuación. No se presenta como prerregistro anterior a los resultados.

## Pregunta

¿Cuántas ofertas libera a tiempo el brazo congelado `GEO_LEARNED` cuando recibe cajas suministradas, con disponibilidad inmediata o retrasada 50 ms, bajo llegadas programadas a 50 Hz y colas acotadas?

## Contrato

- Máquina única y CPU; `CUDA_VISIBLE_DEVICES` vacío, dos hilos Torch/OMP/MKL.
- Checkpoint: `artifacts/sota_20261118/matched_comparator/work/final/GEO_LEARNED__ALL__S7.state.pt`.
- Entradas reales, sin targets: `data/train.parquet` de GarlTTC y `train_inputs.npz` del estudio geométrico congelado.
- Instantes físicos agrupados por `(sequence_id, timestamp_us)`; los cuatro objetos de una oferta proceden del mismo instante y tienen tracks distintos.
- 3.000 ofertas por celda, 50 Hz y 60 s, batch 1 por objeto.
- Población fija de 100 instantes físicos únicos por capacidad de objetos, ciclada 30 veces;
  la repetición mide capacidad de servicio y no aumenta la independencia estadística.
- Objetos por oferta: 1 y 4.
- Capacidad de cola: 2 y 64; la oferta nueva se descarta explícitamente si la cola está llena.
- Disponibilidad de caja: 0 y 50 ms. El retraso desplaza sólo `available_us`; no cambia el tiempo de captura, la caja ni el target (que no se lee).
- Primera llamada fría: cabeza cargada, 250 ms de reposo, una inferencia; queda fuera de la distribución caliente.
- Régimen caliente: cinco llamadas excluidas de calentamiento y luego llegadas absolutas programadas. Las ocho celdas se ejecutan en orden pseudoaleatorio fijado por la semilla 20261121.
- Paridad: tolerancia absoluta `1e-6` por objeto. A 0 ms, referencia desde rasgos guardados; a 50 ms, referencia causal precalculada antes del reloj con los mismos pesos.
- Métricas: ofertas, encoladas, drops, respuestas completas válidas, porcentaje estricto antes de 20 ms, p50/p95/p99/máximo desde deadline programado hasta última salida del conjunto de objetos.
- Alcance: replay host-paced de cajas precargadas. No certifica llegada de sensor físico ni incluye adquisición de cajas.
- Detector/tracker: sólo se incluiría con artefacto congelado, recibo causal de entrada/salida y reloj de disponibilidad verificable. No se identificó tal paquete para esta ejecución, por lo que no se atribuye un coste inventado.
- Telemetría: frecuencia CPU y RSS antes/después. La temperatura queda nula con explicación si Windows/psutil no expone sensor.

El criterio histórico estricto es: todas las ofertas elegibles válidas y terminadas en menos de
20 ms. Cada celda dura 60 s. Una suspensión del host, desmontaje de medios o salto masivo de
lateness invalida la campaña completa para atribución y exige repetirla en host quieto.

