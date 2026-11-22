# WP5 · El modelo ganador dentro de 20 ms

Antes de empezar, leer el [traspaso](README.md).

## Pregunta

¿Puede la variante RGB + geometría (23,47 MiD) responder dentro del plazo de 20 ms sin perder precisión?

## Qué hay medido

| Medida | Resultado | Fuente |
|---|---|---|
| Solo cajas, cómputo aislado en CPU | 0,39 ms de mediana, 1,63 ms de máximo | [`variant_latency`](../../sota_20261118/variant_latency/REPORT.md) |
| Solo cajas, servicio a 50 Hz en CPU | 23.992 de 24.000 ofertas en plazo | [`geometry_service`](../geometry_service/REPORT.md) |
| RGB + geometría, cómputo aislado en GPU, historia en caché | 28,8 ms de mediana, 36,2 ms p95 | `variant_latency` |
| RGB + geometría, recalculando la historia | 114,7 ms de mediana | `variant_latency` |
| RGB + geometría, servicio con GPU compartida | 1.886 de 24.000 con salida válida, ninguna en plazo | [`geometry_service/rgb_gpu`](../geometry_service/rgb_gpu/REPORT.md) |
| Geometría robusta + escala de contenido, CPU | 69,7 ms de mediana; 52,4 ms son la estimación de escala | `variant_latency` |

Desglose de RGB + geometría con caché: recorte y normalización 3,9 ms; productores `R_A5` + `R_C2F` + `PAIR_R` 21,5 ms; refinador `R_CTX` 2,1 ms; geometría 0,2 ms; cabeza 1,3 ms.

Dos cosas sin resolver en esas medidas:
- Los productores tardaban unos 11 ms en las pruebas previas y 21,5 ms en la medición. Se sospecha del estado térmico y de los relojes de la GPU; no se aisló.
- El ensayo de servicio con GPU se hizo con carga ajena, así que no mide el modelo.

## Tareas

1. **Repetir el servicio RGB + geometría con la GPU dedicada.** Mismo protocolo que [`geometry_service/PROTOCOL.md`](../geometry_service/PROTOCOL.md): 24.000 ofertas a 50 Hz, uno y cuatro objetos, registro de respuestas y descartes. Comprobar antes que no hay otros procesos en la GPU y registrar temperatura y frecuencia.
2. **Aislar la discrepancia 11 ms / 21,5 ms.** Medir los productores con la GPU fría y caliente, y con pausas realistas entre consultas (10 Hz y 50 Hz) en lugar de consultas consecutivas.
3. **Optimizar la rama RGB, de menos a más invasivo**, midiendo después de cada paso:
   - Media precisión y formato de memoria `channels_last`.
   - Compilación del grafo (`torch.compile` o exportación a TensorRT/ONNX Runtime).
   - Recorte y normalización en GPU.
   - Fusionar los tres productores en una sola pasada si comparten tronco.
   - Destilar la rama a un productor más pequeño. Esto ya es un modelo nuevo: requiere ajuste por pliegue y pasa por la regla de precisión de abajo.
4. **Optimizar la escala de contenido** (52 ms): vectorizar la estimación, reducir el número de escalas candidatas o pasarla a GPU. Verificar que los campos resultantes coinciden con los actuales dentro de una tolerancia fijada de antemano.
5. **Paridad de precisión.** Tras cada optimización, recalcular las predicciones en una muestra fija de TRAIN40 y comparar con las de referencia.

## Método

- Seguir las convenciones del artículo: lote 1, FP32 con TF32 desactivado como referencia, 2 hilos de PyTorch, calentamiento excluido, cuantiles lineales, lectura de disco fuera del reloj y declarada.
- Mínimo 600 consultas reales de 12 secuencias con semilla fija para el cómputo aislado; 24.000 ofertas para el servicio.
- Informar mediana, p95, p99, máximo, primera llamada y número de consultas por encima de 20 ms.
- Describir el hardware y qué etapas se midieron en vivo.

## Regla de decisión

- **Cumple el plazo** solo si el máximo de las consultas medidas queda por debajo de 20 ms en cómputo aislado y la tasa en plazo del servicio dedicado supera el 99,9 %.
- **Conserva la precisión** si la diferencia de Overall MiD frente a la referencia, en pliegues, tiene un intervalo contenido en ±0,3 MiD. Las optimizaciones puramente numéricas deben además reproducir el TTC dentro de 1 ms en la muestra de paridad.
- No escribir «tiempo real» en ningún sitio: no hay cámara física, detector ni carga sostenida medidos.

## Entregables

`docs/sota_20261121/latency_optimization/PROTOCOL.md`, `REPORT.md`, `RESULTS.json`, trazas del servicio comprimidas con su SHA-256, código en `operational/latency_optimization/` con pruebas. Si alguna variante optimizada cumple ambas reglas, indicar exactamente qué pesos y qué camino de inferencia la definen.

## Coordinación

Necesita la GPU en exclusiva. No ejecutar a la vez que el entrenamiento de [WP1](WP1_COMPARACION_EXTERIOR.md) o [WP7](WP7_DETECTOR.md).
