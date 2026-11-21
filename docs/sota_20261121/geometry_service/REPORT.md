# Ensayo de servicio CPU de geometría suministrada

Ensayo host-paced a 50 Hz, batch 1 por objeto, sobre cajas reales de TRAIN40 y la cabeza
congelada `GEO_LEARNED__ALL__S7`. No mide sensor físico, detector ni tracker. Las cajas con
retraso de 50 ms se implementan desplazando sólo su reloj de disponibilidad; sus tiempos de
captura y bytes permanecen iguales. La primera llamada tras reposo se informa fuera de la
distribución caliente. Todas las ofertas, drops y respuestas fallidas permanecen en los JSON
por celda.

| celda | ofertas | válidas | drops | <20 ms % | p50 | p95 | p99 | máx | fría | paridad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| supplied_050ms__objects_01__queue_64 | 3000 | 3000 | 0 | 3000 (100.00%) | 1.450 | 2.052 | 2.316 | 2.910 | 38.009 | sí |
| supplied_000ms__objects_04__queue_02 | 3000 | 3000 | 0 | 3000 (100.00%) | 2.712 | 5.169 | 6.830 | 8.428 | 3.360 | sí |
| supplied_000ms__objects_04__queue_64 | 3000 | 3000 | 0 | 3000 (100.00%) | 2.761 | 3.591 | 4.200 | 5.102 | 2.120 | sí |
| supplied_050ms__objects_04__queue_64 | 3000 | 3000 | 0 | 3000 (100.00%) | 2.611 | 3.437 | 3.899 | 6.576 | 2.644 | sí |
| supplied_050ms__objects_04__queue_02 | 3000 | 2995 | 5 | 2992 (99.73%) | 2.899 | 4.042 | 6.262 | 154.827 | 2.158 | sí |
| supplied_000ms__objects_01__queue_02 | 3000 | 3000 | 0 | 3000 (100.00%) | 1.439 | 2.565 | 3.986 | 8.023 | 1.452 | sí |
| supplied_000ms__objects_01__queue_64 | 3000 | 3000 | 0 | 3000 (100.00%) | 1.407 | 2.539 | 5.876 | 7.756 | 1.689 | sí |
| supplied_050ms__objects_01__queue_02 | 3000 | 3000 | 0 | 3000 (100.00%) | 1.389 | 2.192 | 3.467 | 6.497 | 1.364 | sí |

La paridad compara cada salida de servicio con una referencia calculada antes del reloj usando
los mismos pesos congelados; para retraso cero también se contrasta con los rasgos guardados.
El criterio histórico estricto exige que todas las ofertas elegibles sean válidas y terminen
antes de 20 ms. El gate estricto es **FAIL**: 7 de 8 celdas cumplen; en agregado hay 23.992
de 24.000 ofertas antes de 20 ms, 23.995 respuestas completas válidas y 5 drops. Por tanto,
no se afirma capacidad garantizada a 50 Hz para toda la matriz. Cada celda ejecutó 3.000
ofertas durante 60 s y cicla una población fija de `100` instantes físicos únicos. El coste
del detector/tracker no se informa porque no
había un artefacto de servicio causal sellado y verificable; asignarle un coste habría mezclado
otro protocolo.

El manifiesto identifica los archivos contenidos en `long_cpu_clean_traces.zip`, incluida la versión del informe guardada al terminar el replay. Este informe de lectura incorpora la comprobación posterior del plazo y de las ofertas descartadas.
