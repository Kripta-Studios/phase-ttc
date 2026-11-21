# Estado del replay RGB + geometría

Estado final al 2026-11-21: **COMPLETADO BAJO CARGA, EVIDENCIA DE EXCLUSIVIDAD INVALIDADA**.

Se ejecutaron ocho celdas reales de `RGB_GEO_BAND` sobre medios eAP, cajas GarlTTC y pesos
congelados: 1/4 objetos, retraso integral de cajas 0/50 ms, colas 2/64, 3.000 ofertas y 60 s
por celda. El `RESULTS.json` tiene SHA-256
`44b6ca830fbc5b262b06ff18fe2b7c2b1cc3542ddd440e1c00334d8429dfa2e4`.

La campaña comenzó en una ventana sin otro Python GPU, pero el entrenamiento ajeno PID 37868
entró 25 s después y antes de la primera celda. Los snapshots de todas las celdas lo registran.
Por ello `evidence_validity=INVALIDATED` y no se presenta como capacidad dedicada. Sí es una
medición íntegra y verificable del comportamiento **bajo esa carga concurrente**.

Resultado agregado: 24.000 ofertas, 1.886 respuestas completas válidas, 22.114 drops, ninguna
oferta completa antes de 20 ms y 3.236/3.236 predicciones producidas con paridad graph/eager
exacta. La paridad no implica cobertura de las ofertas descartadas.

Trazas portables: `rgb_gpu_long_contended_traces.zip`, SHA-256
`d202daf41921ee37c57ec635977df1188d5804f8681949ed5d62fc98e2e03134`.
