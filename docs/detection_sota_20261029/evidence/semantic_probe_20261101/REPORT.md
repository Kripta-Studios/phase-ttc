# Comprobación semántica RGB en TRAIN

Se probó Faster R-CNN ResNet50-FPN v2 con sus pesos COCO_V1 públicos de [torchvision 0.26](https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html). No se entrenó ni se evaluó en selección o test. El modelo recibió únicamente RGB.

Población fijada antes de inferir: 8 imágenes con bicicleta y 8 sin anotación de bicicleta. Umbrales: score ≥ 0.5 e IoU ≥ 0.3 contra el rectángulo del cuboide 3D proyectado.

Coincidencias: **1/10** targets proyectables. Detecciones bicycle sobre los frames negativos: **0**. No son mAP eAP, recall general ni precisión de un detector 3D.

| Token TRAIN | GT proyectable | Bicycle detectadas | Coincidencias | Segundos CPU |
|---|---:|---:|---:|---:|
| 34K84TaN11:000000878_000358229 | 1 | 3 | 1 | 8.720 |
| 6WCDpro5rE:000000312_000747392 | 1 | 1 | 0 | 5.679 |
| 8SdieOgbva:000000752_000555784 | 1 | 0 | 0 | 6.725 |
| DGqicHUGWb:000001065_000398961 | 1 | 0 | 0 | 5.486 |
| Po6ePjWYAz:000000680_000727627 | 3 | 1 | 0 | 8.172 |
| nRcwUlHyAd:000000670_000138327 | 1 | 0 | 0 | 6.020 |
| qoohcdtLDH:000000723_000181251 | 1 | 0 | 0 | 5.620 |
| xkiVthsGwf:000001054_000633300 | 1 | 0 | 0 | 6.042 |
| GjMYZjmSQb:000000177_000701646 | 0 | 0 | 0 | 5.814 |
| 8SdieOgbva:000000525_000252372 | 0 | 0 | 0 | 5.293 |
| LMHPShQkaU:000000299_000880776 | 0 | 0 | 0 | 4.521 |
| pzJcg1OCfc:000001141_000484370 | 0 | 0 | 0 | 4.314 |
| 34K84TaN11:000000748_000156269 | 0 | 0 | 0 | 4.574 |
| 5ilM1PX2vz:000000495_000031899 | 0 | 0 | 0 | 5.988 |
| wE9MZUmiJP:000000607_000854160 | 0 | 0 | 0 | 5.136 |
| mHGFBekt7X:000000488_000202685 | 0 | 0 | 0 | 5.688 |

La inspección de los recortes encontró árboles/vallas y regiones oscuras en varios ejemplos. En otro, la caja bicycle visible es menor que el cuboide proyectado con el ciclista. Son observaciones cualitativas de estos ejemplos, no etiquetas de visibilidad ni un censo del dataset.

Decisión: no ampliar todavía esta receta a inferencia GPU completa. El resultado no justifica el coste y la comparación entre cajas visibles y cuboides amodales requiere cautela. No se cambian los umbrales del ensayo ni se interpreta el diagnóstico como descarte de todo preentrenamiento semántico.

COCO no aporta clases nativas tricycle u other_vehicle. No se certificó admisión de estos pesos en el benchmark ni su redistribución; el trabajo es local. La licencia del código y las identidades de pesos/versiones están en MODEL_INTAKE.json. No hay promoción ni envío oficial.

SHA-256 de pesos: `dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf`. Recortes diagnósticos locales: `D:/PHASE_TTC_SEMANTIC_20261101/visuals_v1/bicycle_crops.png`.
