# Diagnóstico de predicciones en el panel fijo

Fuente: `D:/PHASE_TTC_COMPETITOR_20261101/monocop_screen_v1/evaluation/000512.json`; SHA-256 `d9357e7639c92e6c230f1b3eea9a60f833e37c998646cee987b952e6570b398b`.

512 imágenes de selección pública. Las sustituciones oracle usan GT; no son modelos entrenados ni mejoras obtenidas y no se incorporan a inferencia. La reevaluación independiente reproduce exactamente las métricas originales.

| Variante | mAP alineada |
|---|---:|
| candidate | 0.03013401 |
| incumbent | 0.06533347 |
| oracle_class | 0.08226232 |
| oracle_depth | 0.08025173 |
| oracle_rank | 0.04040682 |
| oracle_size | 0.03013401 |

| Clase | GT | AP candidato | AP incumbente | AP oracle clase | AP oracle profundidad |
|---|---:|---:|---:|---:|---:|
| car | 1999 | 0.23941202 | 0.36310627 | 0.24315210 | 0.60609725 |
| bicycle | 16 | 0.00000000 | 0.00102881 | 0.04879718 | 0.00000000 |
| truck | 782 | 0.00000000 | 0.04100870 | 0.00864612 | 0.01966384 |
| pedestrian | 231 | 0.00166008 | 0.00027640 | 0.02092020 | 0.01625274 |
| motorcycle | 624 | 0.00000000 | 0.04170331 | 0.08410442 | 0.00000000 |
| bus | 68 | 0.00000000 | 0.05598674 | 0.14679418 | 0.00000000 |
| tricycle | 42 | 0.00000000 | 0.01955757 | 0.10568432 | 0.00000000 |
| other_vehicle | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |

| Profundidad óptica GT | GT | Recall candidato 2 m | Recall incumbente 2 m | Recall candidato 4 m | Recall incumbente 4 m |
|---|---:|---:|---:|---:|---:|
| >=150 | 137 | 0.007299 | 0.000000 | 0.007299 | 0.000000 |
| [0,20) | 592 | 0.687500 | 0.778716 | 0.750000 | 0.819257 |
| [100,150) | 524 | 0.070611 | 0.000000 | 0.173664 | 0.003817 |
| [20,40) | 920 | 0.545652 | 0.684783 | 0.655435 | 0.823913 |
| [40,60) | 633 | 0.252765 | 0.489731 | 0.434439 | 0.685624 |
| [60,100) | 956 | 0.128661 | 0.040795 | 0.274059 | 0.084728 |

El matching se realiza en el frame completo antes de estratificar el GT. No se presenta AP por rango ni se trata cada frame correlacionado como una réplica independiente. Las clases con poco soporte requieren confirmación sobre más ejemplos; los cambios pequeños del panel no demuestran generalización.
