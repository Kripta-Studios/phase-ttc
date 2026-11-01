# Diagnóstico de predicciones en el panel fijo

Fuente: `D:/PHASE_TTC_RANGE_20261101/long_range_campaign_v1/screen/evaluation/000512.json`; SHA-256 `8c1c96035cf93b7024097275e8c89606ceca3401976691f32bf025d173a61201`.

512 imágenes de selección pública. Las sustituciones oracle usan GT; no son modelos entrenados ni mejoras obtenidas y no se incorporan a inferencia. La reevaluación independiente reproduce exactamente las métricas originales.

| Variante | mAP alineada |
|---|---:|
| candidate | 0.06067605 |
| incumbent | 0.06533347 |
| oracle_class | 0.11963540 |
| oracle_depth | 0.17605755 |
| oracle_rank | 0.08727315 |
| oracle_size | 0.06067605 |

| Clase | GT | AP candidato | AP incumbente | AP oracle clase | AP oracle profundidad |
|---|---:|---:|---:|---:|---:|
| car | 1999 | 0.34065405 | 0.36310627 | 0.34822113 | 0.74040970 |
| bicycle | 16 | 0.00000000 | 0.00102881 | 0.20064887 | 0.00000000 |
| truck | 782 | 0.04482265 | 0.04100870 | 0.04983312 | 0.27494158 |
| pedestrian | 231 | 0.00003098 | 0.00027640 | 0.01601388 | 0.01725058 |
| motorcycle | 624 | 0.04029907 | 0.04170331 | 0.08632595 | 0.17341967 |
| bus | 68 | 0.05233878 | 0.05598674 | 0.09647539 | 0.14159527 |
| tricycle | 42 | 0.00726286 | 0.01955757 | 0.15956488 | 0.06084362 |
| other_vehicle | 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |

| Profundidad óptica GT | GT | Recall candidato 2 m | Recall incumbente 2 m | Recall candidato 4 m | Recall incumbente 4 m |
|---|---:|---:|---:|---:|---:|
| >=150 | 137 | 0.007299 | 0.000000 | 0.021898 | 0.000000 |
| [0,20) | 592 | 0.748311 | 0.778716 | 0.797297 | 0.819257 |
| [100,150) | 524 | 0.049618 | 0.000000 | 0.158397 | 0.003817 |
| [20,40) | 920 | 0.681522 | 0.684783 | 0.785870 | 0.823913 |
| [40,60) | 633 | 0.364929 | 0.489731 | 0.571880 | 0.685624 |
| [60,100) | 956 | 0.161088 | 0.040795 | 0.315900 | 0.084728 |

El matching se realiza en el frame completo antes de estratificar el GT. No se presenta AP por rango ni se trata cada frame correlacionado como una réplica independiente. Las clases con poco soporte requieren confirmación sobre más ejemplos; los cambios pequeños del panel no demuestran generalización.
