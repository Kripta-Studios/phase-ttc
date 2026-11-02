# Auditoría del rango de entrenamiento — 2026-11-01

Fuente: `D:/PHASE_TTC_RANGE_20261101/train_long_range_v2/CENSUS.json`.
SHA-256: `8f10a2c416526cf5a895245e558b152aded0fef164c61395495d794b537c0094`.

Se auditaron 70,279 fotogramas TRAIN de 24 secuencias. No se leyeron etiquetas de test.
Máximo de profundidad óptica anotada: 249.892971 m; máximo de distancia horizontal ego: 253.595616 m.
No hay anotaciones TRAIN en 300–400 m ni por encima de 400 m.

## Anotaciones por clase

| Clase | Raw | Admitidas <60 m | <150 m | <400 m | Ganancia 150→400 m |
|---|---:|---:|---:|---:|---:|
| bicycle | 834 | 655 | 751 | 751 | 0 |
| bus | 5,137 | 1,958 | 4,603 | 4,782 | 179 |
| car | 304,200 | 181,954 | 289,665 | 290,702 | 1,037 |
| motorcycle | 22,732 | 10,338 | 20,796 | 21,828 | 1,032 |
| other_vehicle | 237 | 53 | 229 | 229 | 0 |
| pedestrian | 7,808 | 3,856 | 7,671 | 7,676 | 5 |
| tricycle | 2,268 | 1,200 | 2,153 | 2,172 | 19 |
| truck | 107,686 | 34,417 | 85,045 | 103,300 | 18,255 |

Ampliar de 150 m a todo el rango observado añade 20,527 anotaciones (5.00%); 18,255 son camiones (88.93% del incremento). Son apariciones correlacionadas, no objetos o escenas independientes.

## Distribución de profundidad y tamaño

| Profundidad óptica [m) | Raw | Admitidas por geometría |
|---|---:|---:|
| 0.1–60 | 252,419 | 234,431 |
| 60–100 | 106,747 | 105,439 |
| 100–150 | 71,198 | 71,043 |
| 150–200 | 16,481 | 16,470 |
| 200–300 | 4,057 | 4,057 |
| 300–400 | 0 | 0 |
| 400–inf | 0 | 0 |

Altura proyectada mediana, tras resize a 640×384:

| Profundidad [m) | Clase | Apariciones | Altura mediana [px] |
|---|---|---:|---:|
| 150–200 | bus | 163 | 19.335 |
| 150–200 | car | 1,037 | 10.844 |
| 150–200 | motorcycle | 1,018 | 8.745 |
| 150–200 | pedestrian | 5 | 10.285 |
| 150–200 | tricycle | 19 | 10.721 |
| 150–200 | truck | 14,228 | 18.411 |
| 200–300 | bus | 16 | 13.860 |
| 200–300 | motorcycle | 14 | 6.613 |
| 200–300 | truck | 4,027 | 14.863 |

Máximo de targets admitidos por imagen: 38; fotogramas con más de 50: 0.

## Comprobaciones y límites

- Exact raw/60/150 per-sequence counts, aggregate reasons and track counts.
- Dos tests cubren la comparación con el productor original (límites de profundidad y exclusiones) y el rechazo de geometría inválida.
- Los filtros reproducen la aproximación pinhole existente: centro dentro de imagen, esquinas delante de cámara y caja recortada mayor de un píxel. No certifican ausencia de oclusión ni detectabilidad visual.
- Los límites son profundidad óptica, con máximo exclusivo, no distancia euclídea.
- Incluir más camiones lejanos no aporta soporte a bicycle u other_vehicle.
- Esto mide targets disponibles; no demuestra una mejora de mAP.

Siguiente comparación de rango: cubrir todos los targets observados con un margen declarado (300 m), manteniendo población, presupuesto y evaluación de ocho clases. No hacen falta etiquetas sintéticas a 400 m.

Reproducción:

```powershell
python scripts/audit_detection_long_range.py --release RELEASE --dataset DATASET --reference-census CENSUS_60_150 --output NUEVO_DIRECTORIO
python scripts/build_detection_long_range_report.py --census NUEVO_DIRECTORIO/CENSUS.json --output INFORME.md
```
