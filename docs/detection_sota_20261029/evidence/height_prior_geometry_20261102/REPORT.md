# Sesgo del prior de profundidad por altura en TRAIN

Panel registrado: 4,096 imágenes TRAIN; 26,546 anotaciones originales y 25,424 targets admitidos a 300 m. No se leen etiquetas de selección ni test; no hay inferencia ni entrenamiento.

Se compara z = altura × fy / altura de la caja proyectada con la profundidad óptica del centro. Altura, yaw y bordes son targets perfectos: este diagnóstico mide el sesgo de esa fórmula, no el error final de un detector con residual aprendido.

La alternativa invierte exactamente la proyección pinhole de los ocho vértices. Cada borde no recortado produce una profundidad; se toma su mediana. Ambas fórmulas se comparan también sobre el mismo subconjunto válido. Los errores pequeños de la inversión incluyen el redondeo de targets a float32. Esto comprueba consistencia interna, no calibración física del sensor.

| Grupo | Targets | Sesgo altura (m) | MAE altura (m) | P90 abs. (m) | Targets con borde válido | MAE altura en ese subconjunto | MAE inversión (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 25424 | -4.844936 | 4.923050 | 8.266880 | 25424 | 4.923050 | 0.000002 |
| class/bicycle | 55 | -1.948236 | 1.948236 | 2.945478 | 55 | 1.948236 | 0.000001 |
| class/bus | 282 | -7.161930 | 7.161930 | 9.158426 | 282 | 7.161930 | 0.000002 |
| class/car | 17152 | -4.374057 | 4.478222 | 7.053372 | 17152 | 4.478222 | 0.000002 |
| class/motorcycle | 1303 | -2.772988 | 2.802484 | 4.645684 | 1303 | 2.802484 | 0.000002 |
| class/other_vehicle | 17 | -4.629466 | 4.629466 | 8.891380 | 17 | 4.629466 | 0.000002 |
| class/pedestrian | 474 | -1.391055 | 1.391055 | 2.315130 | 474 | 1.391055 | 0.000002 |
| class/tricycle | 124 | -2.474604 | 2.906661 | 5.053847 | 124 | 2.906661 | 0.000002 |
| class/truck | 6017 | -6.875333 | 6.893172 | 10.501053 | 6017 | 6.893172 | 0.000003 |
| depth/>=150 | 1221 | -9.544738 | 9.544738 | 12.387219 | 1221 | 9.544738 | 0.000006 |
| depth/[0,20) | 3863 | -1.810224 | 2.324329 | 3.173252 | 3863 | 2.324329 | 0.000000 |
| depth/[100,150) | 4203 | -7.365045 | 7.365045 | 10.011217 | 4203 | 7.365045 | 0.000004 |
| depth/[20,40) | 5858 | -3.511797 | 3.511797 | 4.116127 | 5858 | 3.511797 | 0.000001 |
| depth/[40,60) | 4184 | -4.410096 | 4.410096 | 5.523005 | 4184 | 4.410096 | 0.000001 |
| depth/[60,100) | 6095 | -5.668809 | 5.668809 | 7.576373 | 6095 | 5.668809 | 0.000003 |
| vertical_clipped/False | 23599 | -5.176324 | 5.176324 | 8.422363 | 23599 | 5.176324 | 0.000002 |
| vertical_clipped/True | 1825 | -0.559770 | 1.647981 | 2.774235 | 1825 | 1.647981 | 0.000000 |

Sensibilidad: cambio de profundidad al aumentar un píxel la altura proyectada, manteniendo tamaño y focal. No es una cota de precisión: un regresor puede predecir coordenadas subpíxel.

| Distancia óptica | Mediana de sensibilidad (m/píxel) |
|---|---:|
| >=150 | 9.555224 |
| [0,20) | 0.085975 |
| [100,150) | 6.796902 |
| [20,40) | 0.416408 |
| [40,60) | 1.238228 |
| [60,100) | 2.881225 |

No se ha cambiado ni promocionado ningún modelo. Sustituir directamente el prior de un checkpoint entrenado cambiaría su predicción y podría sumar una corrección que su residual ya aprendió. Antes de entrenar otra variante, hay que medir la inversión con tamaños, orientación y bordes predichos, separar bordes recortados y declarar cómo se tratará ese residual.

Tiempo del diagnóstico: 290.044 s. CUDA inicializada: False.

Filas comprimidas SHA-256: `b0787d89ebe9916b3c00780211ec79541aa561c5b9a9d33ee8053b2b7de32c5c`. Se conservan fuera de Git; contienen información derivada de las anotaciones.
