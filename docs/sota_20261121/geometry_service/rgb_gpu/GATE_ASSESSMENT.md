# Evaluación de la campaña RGB bajo carga

Esta evaluación se vincula al `RESULTS.json` SHA-256
`44b6ca830fbc5b262b06ff18fe2b7c2b1cc3542ddd440e1c00334d8429dfa2e4`.

- `capacity_decision = FAIL`
- `effective_decision = INVALIDATED`
- `evidence_validity = INVALIDATED`
- 0/8 celdas cumplen el criterio estricto.
- 24.000 ofertas; 1.886 válidas; 22.114 drops; 0 antes de 20 ms.
- Paridad: 3.236/3.236 predicciones producidas, error absoluto máximo 0.

El directorio de artefactos conserva el nombre histórico `rgb_gpu_long_dedicated`, pero no fue
una ejecución dedicada completa: el runner PID 15436 empezó sin otro Python GPU visible y el
workload ajeno PID 37868 entró antes de la primera celda. Todas las celdas guardan snapshots
de ambos procesos. La campaña sólo describe servicio bajo contención y no permite atribuir los
tiempos al sistema aislado.

El retraso de 50 ms es causal en ambas ramas de cajas: desplaza el pool geométrico y los relojes
de las observaciones RGB, eliminando recortes cuyas cajas aún no estarían disponibles. Los cuatro
objetos de cada oferta proceden de pistas distintas del mismo `(sequence, timestamp)`.

No se calcula una diferencia CPU–GPU: esta población RGB no se cotejó oferta por oferta con la
población del replay CPU, y además la campaña GPU estuvo sometida a carga concurrente.
