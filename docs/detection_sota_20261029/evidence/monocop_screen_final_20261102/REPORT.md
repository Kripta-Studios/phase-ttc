# Resultado del primer screening MonoCoP

Una semilla; 4.096 imágenes TRAIN, 4.096 exposiciones adicionales y 512 de selección. Cambian arquitectura y procedencia de pesos respecto a MonoDGP; no se iguala su entrenamiento previo ni se atribuye causalidad a un componente aislado. La selección pública ya fue utilizada en desarrollo. No es score oficial.

| Update | MonoCoP | Dfull 300 m | KITTI MonoDGP |
|---:|---:|---:|---:|
| 0 | 0.00560151 | 0.06035564 | 0.00776383 |
| 512 | 0.03013401 | 0.06067605 | 0.03174564 |
| 1024 | 0.02818374 | 0.06435242 | 0.03491650 |
| 1536 | 0.03277838 | 0.06015133 | 0.03658203 |
| 2048 | 0.03476030 | 0.06187839 | 0.03828135 |

Mejor MonoCoP: **0.03476030**, update 2048. Incumbente en el mismo panel: **0.06533347**. Delta incumbente -0.03057318; delta control -0.02959212.

| Clase | GT | AP incumbente | AP MonoCoP | TP 2 m | FP 2 m | Recall 2 m |
|---|---:|---:|---:|---:|---:|---:|
| car | 1999 | 0.36310627 | 0.27502097 | 1082 | 6741 | 0.5413 |
| bicycle | 16 | 0.00102881 | 0.00000000 | 2 | 203 | 0.1250 |
| truck | 782 | 0.04100870 | 0.00075301 | 142 | 13068 | 0.1816 |
| pedestrian | 231 | 0.00027640 | 0.00230842 | 32 | 1016 | 0.1385 |
| motorcycle | 624 | 0.04170331 | 0.00000000 | 35 | 3161 | 0.0561 |
| bus | 68 | 0.05598674 | 0.00000000 | 0 | 81 | 0.0000 |
| tricycle | 42 | 0.01955757 | 0.00000000 | 0 | 37 | 0.0000 |
| other_vehicle | 0 | 0.00000000 | 0.00000000 | 0 | 0 | n/a |

Decisión registrada: **NO_FULL_CONFIRMATION_PANEL_GATE_NOT_MET**. No se ha medido selección completa ni se promociona este candidato.

Tiempo comunicado por el productor: 3396.91 s; peak CUDA asignado: 2.725 GiB.

| Etapa | Media por actualización (s) |
|---|---:|
| prepare_transfer_seconds | 0.013669 |
| forward_seconds | 0.532855 |
| loss_matching_seconds | 0.183905 |
| backward_seconds | 0.411752 |
| optimizer_seconds | 0.056470 |

Los tiempos por actualización no incluyen validación ni checkpoints; el tiempo de ejecución sí los incluye. Un resultado negativo con este presupuesto no prueba convergencia ni descarta recetas más largas. No se añaden épocas o semillas automáticamente.

Checkpoint SHA-256: `a608055453bca574f63d295de38d007061339321549742a62f58339128e723fb`.

Hubo un reinicio del host. Se preservó el STATUS dañado y se verificaron en CPU el modelo y el optimizador de ambos checkpoints recuperados. Reanudación desde 1792; 66 updates registrados se perdieron y se repitieron con los mismos tokens. Losses repetidas bit a bit: False.

Exposiciones lógicas: 4096; ejecutadas como mínimo incluyendo trabajo perdido: 4228. Se reservaron 3120 s del presupuesto original para todo el trabajo anterior al reinicio. Por ello, elapsed_seconds incluye una reserva conservadora; no es una medición exacta del tiempo de cómputo. No se amplió el límite de 7.200 s ni el de 2.048 updates lógicos.
