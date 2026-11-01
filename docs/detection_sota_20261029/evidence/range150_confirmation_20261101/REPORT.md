# Confirmación del checkpoint de 150 m

Mismas 23.958 imágenes públicas, GT raw, ocho clases y perfil alineado. Las métricas se reconstruyeron desde los shards con paridad exacta.

| Variante | mAP completa |
|---|---:|
| Control 60 m, update cero | 0.05701726 |
| Incumbente con ranking | 0.06204229 |
| Candidato 150 m, update 1.024 | 0.06105242 |

Diferencia frente al control: +0.00403516; frente al incumbente: -0.00098987.
Umbral macro del plan original cumplido: **False**. No implica promoción automática ni excelencia.

| Clase | GT | Incumbente AP | Candidato AP | Diferencia |
|---|---:|---:|---:|---:|
| car | 90884 | 0.34789333 | 0.33641472 | -0.01147861 |
| bicycle | 789 | 0.00000000 | 0.00000000 | +0.00000000 |
| truck | 38670 | 0.04317640 | 0.04653466 | +0.00335826 |
| pedestrian | 14415 | 0.00048178 | 0.00022597 | -0.00025582 |
| motorcycle | 39555 | 0.04575330 | 0.04553864 | -0.00021466 |
| bus | 3335 | 0.03563948 | 0.03897482 | +0.00333534 |
| tricycle | 2668 | 0.02339401 | 0.02073053 | -0.00266348 |
| other_vehicle | 153 | 0.00000000 | 0.00000000 | +0.00000000 |

Caídas mayores de 0,01 AP: {"car": -0.011478611727870625}.

| Profundidad óptica | GT | Recall máximo 2 m anterior | Candidato |
|---|---:|---:|---:|
| >=150 | 6559 | 0.000152 | 0.001525 |
| [0,20) | 27580 | 0.767005 | 0.759862 |
| [100,150) | 27293 | 0.000000 | 0.073022 |
| [20,40) | 46355 | 0.655830 | 0.634559 |
| [40,60) | 33373 | 0.473527 | 0.361400 |
| [60,100) | 49309 | 0.037255 | 0.169665 |

El máximo usa asignación de cardinalidad máxima con clase correcta; no depende del orden de scores. El JSON conserva también greedy y umbrales 0,5/1/2/4 m. No es AP por rango.

| Secuencia | Incumbente mAP | Candidato mAP | Diferencia |
|---|---:|---:|---:|
| 2Y46OkK3ff | 0.06137883 | 0.06003595 | -0.00134288 |
| CzyQ8HKP6X | 0.07432400 | 0.07256517 | -0.00175884 |
| EllUE6bwVp | 0.06575204 | 0.06196825 | -0.00378379 |
| JaGC4Q9pgU | 0.05252650 | 0.05551429 | +0.00298779 |
| OBneIVg4Cw | 0.05044210 | 0.04434119 | -0.00610091 |
| WbCh1DRerJ | 0.04092031 | 0.04172738 | +0.00080707 |
| pBqGOb2vYq | 0.06763399 | 0.06239839 | -0.00523561 |
| vWp9XNYQ9V | 0.03924799 | 0.04024879 | +0.00100081 |

Media de diferencias por secuencia: -0.00167830. Intervalo percentil exploratorio 95 %: [-0.00371956, +0.00037686] con 2.000 remuestreos por secuencia, seed 7. Este estimando difiere de la mAP global agrupada. Ocho secuencias expuestas y sin independencia certificada no sostienen una afirmación de significancia o generalización independiente.

El control reutiliza predicciones verificadas de su update cero; la equivalencia cubre detecciones decodificadas, no logits auxiliares ni entrenamiento. No hay nuevo resultado oficial ni etiquetas privadas.
