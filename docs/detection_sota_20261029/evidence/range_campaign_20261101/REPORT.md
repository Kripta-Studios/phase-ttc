# Evidencia del ensayo de rango

Generado desde JSON original. Screening de una semilla en selección ya expuesta; ningún resultado de esta tabla implica promoción o excelencia.

## Panel de selección

| Brazo | Estado del run | Update evaluado | mAP raw | Segundos de validación |
|---|---|---:|---:|---:|
| screen_range60_v1 | COMPLETE | 0 | 0.06035564 | 257.64 |
| screen_range60_v1 | COMPLETE | 512 | 0.05824806 | 359.55 |
| screen_range60_v1 | COMPLETE | 1024 | 0.05238365 | 372.96 |
| screen_range60_v1 | COMPLETE | 1536 | 0.05331721 | 343.91 |
| screen_range60_v1 | COMPLETE | 2048 | 0.05707034 | 347.98 |
| screen_range150_v1 | COMPLETE | 0 | 0.06035564 | 337.64 |
| screen_range150_v1 | COMPLETE | 512 | 0.06205089 | 268.86 |
| screen_range150_v1 | COMPLETE | 1024 | 0.06586902 | 246.20 |
| screen_range150_v1 | COMPLETE | 1536 | 0.06294887 | 190.97 |
| screen_range150_v1 | COMPLETE | 2048 | 0.06445375 | 164.75 |

## Soporte del panel de selección

| Clase | Anotaciones raw |
|---|---:|
| car | 1999 |
| bicycle | 16 |
| truck | 782 |
| pedestrian | 231 |
| motorcycle | 624 |
| bus | 68 |
| tricycle | 42 |
| other_vehicle | 0 |

Las clases sin GT conservan AP=0 en la macro de ocho clases. Este panel no permite confirmar su calidad; se exige selección completa.

## Soporte raw TRAIN

Los identificadores se cuentan dentro de secuencia. No certifican adquisiciones independientes.

| Clase | Anotaciones | Instancias por secuencia | Secuencias |
|---|---:|---:|---:|
| bicycle | 834 | 86 | 12 |
| bus | 5137 | 134 | 20 |
| car | 304200 | 4622 | 24 |
| motorcycle | 22732 | 1453 | 24 |
| other_vehicle | 237 | 16 | 10 |
| pedestrian | 7808 | 469 | 20 |
| tricycle | 2268 | 87 | 15 |
| truck | 107686 | 1974 | 24 |

Unión clase/centro: PASS; colisiones: 0.

## Admisión de targets TRAIN

| Clase | Raw | Admitidos 60 m | Admitidos 150 m |
|---|---:|---:|---:|
| bicycle | 834 | 655 | 751 |
| bus | 5137 | 1958 | 4603 |
| car | 304200 | 181954 | 289665 |
| motorcycle | 22732 | 10338 | 20796 |
| other_vehicle | 237 | 53 | 229 |
| pedestrian | 7808 | 3856 | 7671 |
| tricycle | 2268 | 1200 | 2153 |
| truck | 107686 | 34417 | 85045 |

## Reproducción

```powershell
python scripts/build_detection_range_report.py `
  --campaign D:/PHASE_TTC_RANGE_20261101 `
  --output docs/detection_sota_20261029/evidence/range_campaign_20261101
```
