# Resultado del screening 60/150/300 m

Una semilla, 4.096 imágenes TRAIN y 512 de selección ya expuesta. Misma receta y población; cambia el rango. No es una prueba independiente.

| Actualización | mAP 60 m | mAP 150 m | mAP 300 m |
|---:|---:|---:|---:|
| 0 | 0.06035564 | 0.06035564 | 0.06035564 |
| 512 | 0.05824806 | 0.06205089 | 0.06067605 |
| 1024 | 0.05238365 | 0.06586902 | 0.06435242 |
| 1536 | 0.05331721 | 0.06294887 | 0.06015133 |
| 2048 | 0.05707034 | 0.06445375 | 0.06187839 |

Incumbente con ranking en el mismo panel: **0.06533347**. Mejor 300 m: update 1024, mAP **0.06435242**. Delta incumbente -0.00098106; delta control 150 m -0.00151660.

| Clase | GT panel | AP incumbente | AP 150 m seleccionado | AP 300 m seleccionado |
|---|---:|---:|---:|---:|
| car | 1999 | 0.36310627 | 0.34427324 | 0.33570919 |
| bicycle | 16 | 0.00102881 | 0.00000000 | 0.01230453 |
| truck | 782 | 0.04100870 | 0.04417464 | 0.04618799 |
| pedestrian | 231 | 0.00027640 | 0.00000000 | 0.00000083 |
| motorcycle | 624 | 0.04170331 | 0.03976179 | 0.03632443 |
| bus | 68 | 0.05598674 | 0.08226500 | 0.07375281 |
| tricycle | 42 | 0.01955757 | 0.01647746 | 0.01053957 |
| other_vehicle | 0 | 0.00000000 | 0.00000000 | 0.00000000 |

Las clases sin GT mantienen AP=0 en la macro. Este panel no confirma su calidad.

Decisión: conservar el incumbente y no asignar confirmación completa a este 300 m, al no superar en el panel al incumbente y al control pareado.
Esta es una decisión de recursos posterior al screening. El plan exige para promoción una ganancia completa de al menos 0,005 frente al incumbente y un control válido, con revisión de caídas por clase >0,01; esa prueba completa no se ha ejecutado para 300 m. No se descarta toda receta de mayor rango.

El ensayo KITTI registrado continúa por separado. Igualar nuevas exposiciones no iguala el historial previo de aprendizaje de los dos modelos.

Checkpoint conservado: `D:/PHASE_TTC_RANGE_20261101/long_range_campaign_v1/screen/checkpoints/selected.pt`. SHA-256 `73eb2c881e41508f4841842f1b0555a209198bf947e1849a520272a4fa1a993f`.
Se verificaron carga CPU, update seleccionado, contrato, fuentes archivadas, paridad inicial de predicciones y métricas, y recibo de finalización de la cola.
