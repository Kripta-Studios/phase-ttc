# PHASE-TTC-Geo 23,47 — protocolo de reproducción A/B/C

Fecha de congelación: 2026-11-21, antes de inspeccionar las diferencias de B/C.
No se leen etiquetas privadas ni se contacta CodaBench.

## Alcance y tolerancias

- **A, paquete desde predicciones guardadas.** Éxito estricto: SHA-256 del ZIP igual a
  `8590f1151bb8b8db5baa5ae364675356a47ca93a3ee58139421f3d80d0f831df`.
- **B1, endpoints autenticados.** Las 5.805 consultas físicas se ensamblan desde los
  diagnósticos congelados, las cajas públicas y el checkpoint Geo congelado; se expanden
  a 6.762 tokens. Éxito estricto: cero diferencia TTC y ZIP idéntico. Este brazo no se
  denomina "desde medios".
- **B2, regeneración desde medios.** HDF5 y shards RGB públicos, fijados a la revisión
  `a814a946e60d37f7242358af5f344542177e50c5`, recorren los productores originales.
  Paridad de endpoint: máximo absoluto `1e-6`; equivalencia TTC: máximo absoluto `1e-4 s`.
  La cobertura física y por secuencia se informa literalmente. Un piloto no se extrapola
  a 5.805 consultas.
- **C, reajuste.** Se ajusta `RGB_GEO_BAND` desde el split TRAIN40 con semillas 7, 13 y 21,
  12.044 actualizaciones por semilla. Se informa dispersión entre semillas en Test12 y
  hashes de estados; no se calcula Overall privado.

## Diferencias y unidades

Toda deriva se publica en segundos TTC por token, con cuantiles, peor fila, secuencia,
faltantes, extras, cambios de signo y alias. Ninguna distancia entre dos predicciones se
interpreta como precisión ni se convierte en el Overall privado 23,47.

## Aislamiento

Las salidas nuevas viven sólo en `operational/geo_reproduction/runs/`; la documentación
nueva vive sólo en este directorio. Los candidatos y artefactos históricos son de sólo
lectura.
