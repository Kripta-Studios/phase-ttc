# Diagnóstico de propuestas TRAIN y clases raras

Mismo checkpoint congelado, sin inferencia nueva. TRAIN: 4.096 imágenes, incluidas todas las imágenes TRAIN anotadas con bicycle u other_vehicle. Selección: 23.958 imágenes. Son poblaciones diferentes; sus diferencias no estiman por sí solas una brecha de generalización. Las vistas de un mismo objeto están correlacionadas.

Recall a 2 m con asignación de cardinalidad máxima en el frame completo. El matching ignora el orden de scores. Después se agrupan los GT por clase. No son AP ni recall operativo a un umbral de confianza.

| Clase | GT TRAIN | TRAIN clase correcta | TRAIN sin exigir clase | GT selección | Selección clase correcta | Selección sin exigir clase |
|---|---:|---:|---:|---:|---:|---:|
| car | 18108 | 0.604981 | 0.612547 | 90884 | 0.535826 | 0.557535 |
| bicycle | 834 | 0.509592 | 0.766187 | 789 | 0.076046 | 0.505703 |
| truck | 6500 | 0.314000 | 0.324462 | 38670 | 0.201991 | 0.244039 |
| pedestrian | 1068 | 0.441011 | 0.482210 | 14415 | 0.088796 | 0.238432 |
| motorcycle | 2528 | 0.523734 | 0.564478 | 39555 | 0.259638 | 0.314676 |
| bus | 343 | 0.364431 | 0.379009 | 3335 | 0.187406 | 0.248876 |
| tricycle | 273 | 0.439560 | 0.564103 | 2668 | 0.167916 | 0.431784 |
| other_vehicle | 237 | 0.105485 | 0.147679 | 153 | 0.026144 | 0.137255 |

Confusión sobre asociaciones agnósticas máximas a 4 m; un emparejamiento cercano no certifica identidad física del objeto.

| GT | Clase predicha | Asociaciones TRAIN | Asociaciones selección |
|---|---|---:|---:|
| bicycle | car | 11 | 18 |
| bicycle | bicycle | 410 | 63 |
| bicycle | truck | 8 | 4 |
| bicycle | pedestrian | 15 | 81 |
| bicycle | motorcycle | 269 | 396 |
| bicycle | bus | 7 | 5 |
| bicycle | tricycle | 3 | 11 |
| bicycle | other_vehicle | 1 | 6 |
| other_vehicle | car | 2 | 10 |
| other_vehicle | truck | 1 | 10 |
| other_vehicle | pedestrian | 11 | 5 |
| other_vehicle | motorcycle | 6 | 10 |
| other_vehicle | bus | 3 | 0 |
| other_vehicle | tricycle | 6 | 0 |
| other_vehicle | other_vehicle | 32 | 5 |
| pedestrian | car | 7 | 464 |
| pedestrian | bicycle | 4 | 161 |
| pedestrian | truck | 8 | 247 |
| pedestrian | pedestrian | 533 | 1775 |
| pedestrian | motorcycle | 76 | 3197 |
| pedestrian | bus | 1 | 26 |
| pedestrian | tricycle | 1 | 142 |
| pedestrian | other_vehicle | 6 | 284 |

IDs (secuencia, instance_id) del censo TRAIN completo, no solo este panel. No equivalen a escenas independientes.

| Clase | IDs raw | IDs admitidos 60 m | IDs admitidos 150 m |
|---|---:|---:|---:|
| bicycle | 86 | 69 | 77 |
| other_vehicle | 16 | 6 | 16 |
| pedestrian | 469 | 255 | 464 |

El diagnóstico no separa por sí solo capacidad, ambigüedad visual, ruido de anotación o desbalance. Las distancias usan GT únicamente para medir errores; no producen nuevas predicciones. No se ha entrenado una receta nueva ni se promueve un modelo. Se conservan los resultados negativos de R2-rare, de los rankers y del ensayo semántico CPU.
