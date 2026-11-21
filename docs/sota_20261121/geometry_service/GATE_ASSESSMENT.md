# Decisión del gate estricto del servicio CPU

Esta evaluación post-run se deriva del bundle sellado
`artifacts/sota_20261121/geometry_service/long_cpu_clean`, cuyo `RESULTS.json` tiene SHA-256
`ad77962aba3599e2398472a5b67b8e90a7a5a80dd7f94104cd66d89758ed55a8`.

## Decisión

`strict_service_gate = FAIL`

- Cumplen el gate completo: 7 de 8 celdas.
- Ofertas programadas: 24.000.
- Respuestas completas válidas: 23.995.
- Respuestas completas antes de 20 ms: 23.992.
- Drops de cola: 5.
- Paridad de respuestas producidas: 59.980 de 59.980 predicciones comparadas pasan con error
  absoluto máximo 0. Las 20 predicciones ausentes corresponden a las 5 ofertas de cuatro
  objetos descartadas y no fueron comparadas; la paridad no implica cobertura completa.

La celda `supplied_050ms__objects_04__queue_02` incumple el criterio: 5 ofertas fueron
descartadas, 2.992 de 3.000 ofertas terminaron válidamente antes de 20 ms y el máximo desde
deadline fue 154,8274 ms. Por tanto, este ensayo **no demuestra ni garantiza capacidad
sostenida a 50 Hz para toda la matriz**. Sí aporta ocho ejecuciones reales de 60 s y permite
identificar las siete condiciones que cumplieron en esta única campaña.

El párrafo condicional del `REPORT.md` sellado que dice que no demuestra 60 s «cuando su
duración es menor» es una plantilla defectuosa: cada celda de este bundle sí ejecutó 3.000
ofertas durante 60 s. Esta corrección interpretativa no modifica los bytes del informe ni de
los resultados sellados.
