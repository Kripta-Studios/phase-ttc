# MonoCoP multiclase: transferencia CPU verificada

Estado **PASS_CPU_TRANSFER_ONLY**. Pesos públicos fijados, cero pasos de optimizador y ninguna métrica eAP medida. No hay promoción ni otro entrenamiento GPU admitido por este resultado.

Checkpoint original: `f574a65bfd22ebcbb80b77fd18a09b4812d89ae5cdc21e6a556a081b7b755321`; 694 tensores. Pesos puros: `344222be2ed6413294593986bba84fc3d4bf17548dd0688d6022c3535d7d7c06`. La conversión no reutiliza el optimizador y conserva exactamente todos los tensores.

Se mapearon Pedestrian→pedestrian, Car→car y Cyclist→bike. Las otras cinco filas se inicializaron por separado. Cyclist/bike es una hipótesis de transferencia, no una equivalencia certificada de las anotaciones. Los bins de 300 m se conservaron y su clasificador se reinicializó; el embedding conserva las filas físicas 0–60 y repite la última por encima.

Se copiaron 683 tensores completos, se adaptaron 11 y se verificaron 30 grupos de aliases. Repetir la transferencia conserva pesos exactos aunque cambie el RNG exterior, y no modifica el estado aleatorio CPU del llamante.

| Etapa | CPU v1, s | CPU v2, s |
|---|---:|---:|
| construction | 31.432 | 6.508 |
| transfer_contract | 5.359 | 3.766 |
| eight_class_backward | 6.951 | 6.346 |
| empty_target_backward | 6.264 | 6.044 |
| checkpoint_replay | 3.368 | 3.294 |
| real_train_frame_backward | 40.330 | 37.851 |

v1 pasó las pruebas de ejecución, pero Pyright rechazó el acceso al generador CPU por un símbolo no exportado. v2 usa Generator/get_state/set_rng_state públicos. Todos los resultados numéricos de las etapas y el hash del checkpoint inicializado son **idénticos**. No hubo reentrenamiento ni ajuste según métricas de selección.

Pasaron forward/backward con ocho clases y targets vacíos, replay de inferencia sin etiquetas y un frame TRAIN real a 640×384. Los parámetros y buffers quedaron intactos tras las comprobaciones.

Frame `xkiVthsGwf:000001113_000434196`: 7 targets y profundidad máxima 249.153183 m. El decoder emitió 50/50 cajas con geometría válida. Esto no mide recall, AP ni utilidad de esas cajas. El tiempo de esta etapa incluye lectura, preparación, backward y otra inferencia; no es latencia aislada ni una estimación de coste GPU.

Checkpoint inicializado: `e0de1b40d0b0316eaaf88d8133b333a00b825baf080fc19604b55d51c321b80a`. Es un punto de partida reproducible, no un candidato entrenado en eAP.

Se conservan v1, v2, fuentes ejecutadas y los recibos de procedencia. El checkpoint upstream no incluye hashes de su receta original; la asociación a las tres clases se apoya en la configuración y el log publicados. No se certifican aquí licencia de pesos ni admisión de datos externos en una evaluación oficial.

Siguiente decisión: terminar KITTI y revisar su resultado. Después, si procede, registrar un smoke GPU acotado para aprendizaje, paridad y coste; el screening de calidad requiere su propio control y presupuesto.
