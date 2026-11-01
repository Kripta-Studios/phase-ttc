# Ranking por clase entrenado exclusivamente en TRAIN

Tabla regenerada de JSON verificados. Todas las variantes conservan las ocho clases
en la media y usan las mismas 23.958 imágenes públicas de selección y GT raw.
La selección ya se ha consultado durante el desarrollo; no es test independiente.

| Variante | mAP raw alineada | Propuestas |
|---|---:|---:|
| baseline | 0.05701726 | 1197900 |
| ranked | 0.06204229 | 1197900 |
| scalar_uncertainty | 0.06053089 | 1197900 |
| evaluation_linear_16_to_8 | 0.05809659 | 9583200 |
| evaluation_mlp_16_to_64_gelu_to_8 | 0.05752680 | 9583200 |

| Clase | baseline | ranked | scalar_uncertainty | evaluation_linear_16_to_8 | evaluation_mlp_16_to_64_gelu_to_8 |
|---|---:|---:|---:|---:|---:|
| car | 0.31901477 | 0.34789333 | 0.33836097 | 0.34070463 | 0.33767059 |
| bicycle | 0.00000000 | 0.00000000 | 0.00000000 | 0.00167716 | 0.00009392 |
| truck | 0.03970005 | 0.04317640 | 0.04153224 | 0.03593649 | 0.03831016 |
| pedestrian | 0.00039233 | 0.00048178 | 0.00043715 | 0.00042153 | 0.00079397 |
| motorcycle | 0.04742940 | 0.04575330 | 0.04703056 | 0.04421119 | 0.04533990 |
| bus | 0.03405455 | 0.03563948 | 0.03562433 | 0.03246377 | 0.03087022 |
| tricycle | 0.01554696 | 0.02339401 | 0.02126190 | 0.00935797 | 0.00713561 |
| other_vehicle | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |

Estados: evaluation_linear_16_to_8=BELOW_INCUMBENT_NOT_PROMOTED, evaluation_mlp_16_to_64_gelu_to_8=BELOW_INCUMBENT_NOT_PROMOTED.

Las dos cabezas aprendieron con 4096 imágenes TRAIN, incluidas las 799 con
bicycle u other_vehicle. Ocho épocas fijas; sin selección de época ni ajuste
con etiquetas de selección. Aplican 400 hipótesis por imagen a las mismas
50 geometrías. No cambian centros, tamaños, orientación ni velocidad.

La pérdida TRAIN baja, pero eso no demuestra mejora de detección. El
resultado se decide con la macro de las ocho clases. La pequeña AP bicycle
no compensa las pérdidas en otras clases; other_vehicle sigue sin AP útil.

Ninguna fila certifica equivalencia con el scorer privado ni excelencia. La regla
de promoción exige +0,005 frente al incumbent ranked, revisar caídas >0,01 por
clase y validar un productor ejecutable sin etiquetas antes de empaquetar.
