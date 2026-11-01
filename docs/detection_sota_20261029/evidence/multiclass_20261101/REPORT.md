# Hipótesis de clase sobre geometría congelada

Tabla regenerada de JSON verificados. Todas las variantes conservan las ocho clases
en la media y usan las mismas 23.958 imágenes públicas de selección y GT raw.
La selección ya se ha consultado durante el desarrollo; no es test independiente.

| Variante | mAP raw alineada | Propuestas |
|---|---:|---:|
| baseline | 0.05701726 | 1197900 |
| ranked | 0.06204229 | 1197900 |
| scalar_uncertainty | 0.06053089 | 1197900 |
| v1 | 0.05797162 | 9583200 |
| top50_v1 | 0.05783289 | 1197900 |

| Clase | baseline | ranked | scalar_uncertainty | v1 | top50_v1 |
|---|---:|---:|---:|---:|---:|
| car | 0.31901477 | 0.34789333 | 0.33836097 | 0.31931732 | 0.31882180 |
| bicycle | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| truck | 0.03970005 | 0.04317640 | 0.04153224 | 0.04105984 | 0.04105432 |
| pedestrian | 0.00039233 | 0.00048178 | 0.00043715 | 0.00060610 | 0.00061086 |
| motorcycle | 0.04742940 | 0.04575330 | 0.04703056 | 0.04884902 | 0.04822697 |
| bus | 0.03405455 | 0.03563948 | 0.03562433 | 0.03702572 | 0.03703605 |
| tricycle | 0.01554696 | 0.02339401 | 0.02126190 | 0.01691496 | 0.01691314 |
| other_vehicle | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |

Estados: v1=EVALUATED_NOT_PROMOTED, top50_v1=EVALUATED_NOT_PROMOTED.

La expansión conserva los centros, tamaños, orientación y velocidad. No aumenta
la cobertura geométrica agnóstica de clase. Los logits alternativos no usan el
ranker TRAIN congelado: aplicarlo sin una receta nueva mezclaría semánticas.

Ninguna fila certifica equivalencia con el scorer privado ni excelencia. La regla
de promoción exige +0,005 frente al incumbent ranked, revisar caídas >0,01 por
clase y validar un productor ejecutable sin etiquetas antes de empaquetar.
