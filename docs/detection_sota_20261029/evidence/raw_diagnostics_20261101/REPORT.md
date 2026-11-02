# Diagnóstico del detector eAP sobre selección pública raw

Las sustituciones oracle usan GT público. No son modelos entrenados, mejoras
conseguidas ni cotas superiores generales. Mismo pool de propuestas; sin test privado.

Población: 23958 frames; 190469 GT; 1197900 propuestas. Paridad completa con el JSON alineado verificada.

## Sustituciones controladas

| Variante | mAP | ATE | ASE | AOE |
|---|---:|---:|---:|---:|
| ranked | 0.06204229 | 0.869129 | 0.483038 | 0.507858 |
| oracle_class | 0.09393767 | 0.925940 | 0.319222 | 0.513069 |
| oracle_depth | 0.14572786 | 0.318648 | 0.247626 | 0.476248 |
| oracle_size | 0.06204229 | 0.869129 | 0.407961 | 0.507858 |
| oracle_rank | 0.07309049 | 0.728955 | 0.495760 | 0.536651 |

Clase/tamaño: asignación agnóstica de cardinalidad máxima y mínima distancia XY <4 m.
Profundidad: asignación de misma clase por rayos de cámara normalizados con norma <0,02;
se conserva el rayo predicho y se sustituye su profundidad óptica. No se cambian cajas
sin asignación. Ranking: score=1/(1+distancia al GT más próximo de igual clase),
sin eliminar duplicados. Ninguna de estas variantes es inferencia desplegable.

## Clases

| Clase | GT | AP real | AP oracle profundidad | Recall 2m clase | Recall 2m agnóstico máximo |
|---|---:|---:|---:|---:|---:|
| car | 90884 | 0.347893 | 0.584498 | 0.535727 | 0.557535 |
| bicycle | 789 | 0.000000 | 0.000000 | 0.076046 | 0.505703 |
| truck | 38670 | 0.043176 | 0.187077 | 0.201991 | 0.244039 |
| pedestrian | 14415 | 0.000482 | 0.014488 | 0.087617 | 0.238432 |
| motorcycle | 39555 | 0.045753 | 0.178879 | 0.255265 | 0.314676 |
| bus | 3335 | 0.035639 | 0.107967 | 0.187406 | 0.248876 |
| tricycle | 2668 | 0.023394 | 0.092914 | 0.167916 | 0.431784 |
| other_vehicle | 153 | 0.000000 | 0.000000 | 0.026144 | 0.137255 |

## Profundidad óptica

| Estrato | GT | Recall clase 0,5m | 1m | 2m | 4m |
|---|---:|---:|---:|---:|---:|
| >=150 | 6559 | 0.000000 | 0.000000 | 0.000152 | 0.000457 |
| [0,20) | 27580 | 0.558738 | 0.696157 | 0.765410 | 0.817150 |
| [100,150) | 27293 | 0.000000 | 0.000000 | 0.000000 | 0.000220 |
| [20,40) | 46355 | 0.279474 | 0.465710 | 0.653155 | 0.796160 |
| [40,60) | 33373 | 0.139274 | 0.275612 | 0.472628 | 0.671711 |
| [60,100) | 49309 | 0.008274 | 0.018029 | 0.037235 | 0.080614 |

## Distancia radial ego XY

| Estrato | GT | Recall clase 0,5m | 1m | 2m | 4m |
|---|---:|---:|---:|---:|---:|
| >=150 | 7552 | 0.000000 | 0.000000 | 0.000132 | 0.000397 |
| [0,20) | 21686 | 0.598866 | 0.718528 | 0.775062 | 0.823158 |
| [100,150) | 29421 | 0.000000 | 0.000000 | 0.000000 | 0.000272 |
| [20,40) | 47035 | 0.307027 | 0.498076 | 0.675922 | 0.804847 |
| [40,60) | 34260 | 0.149621 | 0.294308 | 0.497023 | 0.694892 |
| [60,100) | 50515 | 0.017163 | 0.035296 | 0.066673 | 0.125092 |

## Altura proyectada en imagen original de 720 px

| Estrato | GT | Recall clase 0,5m | 1m | 2m | 4m |
|---|---:|---:|---:|---:|---:|
| >=128 | 42400 | 0.473184 | 0.634882 | 0.747995 | 0.831415 |
| [0,16) | 552 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [16,32) | 31002 | 0.000032 | 0.000065 | 0.000323 | 0.000871 |
| [32,64) | 61842 | 0.047638 | 0.093998 | 0.165729 | 0.257721 |
| [64,128) | 53894 | 0.192934 | 0.336308 | 0.501058 | 0.642020 |
| unprojectable | 779 | 0.016688 | 0.020539 | 0.024390 | 0.033376 |

## Dominio histórico

| Estrato | GT | Recall clase 0,5m | 1m | 2m | 4m |
|---|---:|---:|---:|---:|---:|
| camera_center_depth_outside_protocol | 83161 | 0.004906 | 0.010690 | 0.022090 | 0.047907 |
| eligible | 107308 | 0.307647 | 0.465818 | 0.625862 | 0.762851 |

## AP por secuencia

| Secuencia | Frames | GT | mAP |
|---|---:|---:|---:|
| 2Y46OkK3ff | 2995 | 20948 | 0.061379 |
| CzyQ8HKP6X | 2995 | 17109 | 0.074324 |
| EllUE6bwVp | 2995 | 26870 | 0.065752 |
| JaGC4Q9pgU | 2993 | 23908 | 0.052526 |
| OBneIVg4Cw | 2994 | 30475 | 0.050442 |
| WbCh1DRerJ | 2996 | 21254 | 0.040920 |
| pBqGOb2vYq | 2995 | 36131 | 0.067634 |
| vWp9XNYQ9V | 2995 | 13774 | 0.039248 |

## Límites

- Recall por rango tras matching global por frame; no es AP por rango.
- Cardinalidad máxima: cota solo para el pool y grafo de distancia fijos.
- Recall agnóstico comparte propuestas; no es una cota independiente por clase.
- Ocho secuencias expuestas a desarrollo; independencia no certificada.
- Errores geométricos condicionados a matches <2 m están en ERROR_DIAGNOSTICS.json;
  no sustituyen los errores TP y defaults del benchmark.
- Iluminación, visibilidad e identidades de adquisición no certificadas por estos shards.
- No hay nuevos pesos ni nuevo envío oficial.
