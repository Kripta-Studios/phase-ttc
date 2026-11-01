# Resultado del screening de inicialización KITTI

Una semilla; mismo presupuesto adicional de 4.096 exposiciones eAP, 4.096 imágenes TRAIN y 512 de selección. No se iguala el historial anterior: Dfull ya vio 281.116 exposiciones eAP y el origen KITTI se entrenó para Car. Este resultado no descarta toda transferencia externa ni prueba convergencia con otra receta.

| Update | Dfull 300 m | KITTI |
|---:|---:|---:|
| 0 | 0.06035564 | 0.00776383 |
| 512 | 0.06067605 | 0.03174564 |
| 1024 | 0.06435242 | 0.03491650 |
| 1536 | 0.06015133 | 0.03658203 |
| 2048 | 0.06187839 | 0.03828135 |

Incumbente con ranking: **0.06533347**. Mejor KITTI: **0.03828135**, update 2048. Delta frente al incumbente -0.02705213; frente al control -0.02607107.

| Clase | GT panel | AP incumbente | AP control | AP KITTI |
|---|---:|---:|---:|---:|
| car | 1999 | 0.36310627 | 0.33570919 | 0.30625077 |
| bicycle | 16 | 0.00102881 | 0.01230453 | 0.00000000 |
| truck | 782 | 0.04100870 | 0.04618799 | 0.00000000 |
| pedestrian | 231 | 0.00027640 | 0.00000083 | 0.00000000 |
| motorcycle | 624 | 0.04170331 | 0.03632443 | 0.00000000 |
| bus | 68 | 0.05598674 | 0.07375281 | 0.00000000 |
| tricycle | 42 | 0.01955757 | 0.01053957 | 0.00000000 |
| other_vehicle | 0 | 0.00000000 | 0.00000000 | 0.00000000 |

| Clase KITTI seleccionado | Predicciones | TP a 2 m | FP a 2 m | Recall 2 m | Recall 4 m |
|---|---:|---:|---:|---:|---:|
| car | 6195 | 1142 | 5053 | 0.5713 | 0.6993 |
| bicycle | 0 | 0 | 0 | 0.0000 | 0.0000 |
| truck | 14776 | 180 | 14596 | 0.2302 | 0.3926 |
| pedestrian | 75 | 0 | 75 | 0.0000 | 0.0000 |
| motorcycle | 4552 | 35 | 4517 | 0.0561 | 0.1506 |
| bus | 0 | 0 | 0 | 0.0000 | 0.0000 |
| tricycle | 2 | 0 | 2 | 0.0000 | 0.0000 |
| other_vehicle | 0 | 0 | 0 | n/a | n/a |

AP cero no implica necesariamente cero aciertos. El perfil alineado usa puntos de recall superiores a 0,1, resta 0,1 a la precisión interpolada y recorta resultados negativos a cero. El JSON conserva TP, FP, recall y la precisión máxima utilizada por AP en cada umbral. Así se distingue ausencia de propuestas útiles de una curva con precisión insuficiente. No se cambia la métrica ni el ranking del ensayo.

Las clases sin GT conservan AP=0 en la macro. El panel ya está expuesto; no constituye test independiente.

Decisión: **NO_FULL_CONFIRMATION_PANEL_GATE_NOT_MET**. El criterio previo exige +0,005 frente a ambas referencias para considerar la confirmación completa. No se ejecuta selección completa ni se promociona este checkpoint mediante este informe.

Se verificaron fuentes ejecutadas, recibos de finalización, receta, población, perfil, carga CPU, finitud y selección del checkpoint.

Checkpoint conservado: `D:/PHASE_TTC_PRETRAINED_20261101/comparison_campaign_v1/screen/checkpoints/selected.pt`; SHA-256 `36efe9f80274616f51ab7d8e6a34afbec9cc393b1156ad5ab43a1dd84861fe8d`.

Próxima revisión: valorar el smoke técnico de MonoCoP multiclase ya preparado, con otro contrato y presupuesto local de 600 s. Este informe no inicia ni admite automáticamente esa ejecución.
