# Diagnóstico de predicciones en el panel fijo

Fuente: `D:/PHASE_TTC_COMPETITOR_20261101/monocop_screen_v1/evaluation/000000.json`; SHA-256 `f95590a728f869b22a903e4611bdeb9e73d2ad9c2b62564d2604a69a865b1173`.

512 imágenes de selección pública. Las sustituciones oracle usan GT; no son modelos entrenados ni mejoras obtenidas y no se incorporan a inferencia. La reevaluación independiente reproduce exactamente las métricas originales.

| Variante | mAP alineada |
|---|---:|
| candidate | 0.00560151 |
| incumbent | 0.06533347 |
| oracle_class | 0.02349410 |
| oracle_depth | 0.01458151 |
| oracle_rank | 0.01204061 |
| oracle_size | 0.00560151 |

| Clase | GT | AP candidato | AP incumbente | AP oracle clase | AP oracle profundidad |
|---|---:|---:|---:|---:|---:|
| car | 1999 | 0.04480934 | 0.36310627 | 0.04556832 | 0.11600779 |
| bicycle | 16 | 0.00000000 | 0.00102881 | 0.00092294 | 0.00000000 |
| truck | 782 | 0.00000000 | 0.04100870 | 0.02194213 | 0.00000000 |
| pedestrian | 231 | 0.00000278 | 0.00027640 | 0.00076001 | 0.00064426 |
| motorcycle | 624 | 0.00000000 | 0.04170331 | 0.08639338 | 0.00000000 |
| bus | 68 | 0.00000000 | 0.05598674 | 0.00000000 | 0.00000000 |
| tricycle | 42 | 0.00000000 | 0.01955757 | 0.03236605 | 0.00000000 |
| other_vehicle | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |

| Profundidad óptica GT | GT | Recall candidato 2 m | Recall incumbente 2 m | Recall candidato 4 m | Recall incumbente 4 m |
|---|---:|---:|---:|---:|---:|
| >=150 | 137 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [0,20) | 592 | 0.555743 | 0.778716 | 0.636824 | 0.819257 |
| [100,150) | 524 | 0.000000 | 0.000000 | 0.000000 | 0.003817 |
| [20,40) | 920 | 0.404348 | 0.684783 | 0.580435 | 0.823913 |
| [40,60) | 633 | 0.099526 | 0.489731 | 0.210111 | 0.685624 |
| [60,100) | 956 | 0.000000 | 0.040795 | 0.004184 | 0.084728 |

El matching se realiza en el frame completo antes de estratificar el GT. No se presenta AP por rango ni se trata cada frame correlacionado como una réplica independiente. Las clases con poco soporte requieren confirmación sobre más ejemplos; los cambios pequeños del panel no demuestran generalización.
