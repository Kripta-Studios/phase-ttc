# MonoCoP: admisión técnica CPU

Estado **PASS_CPU_PORT_ONLY**; 42,599,880 parámetros, 42,571,287 entrenables. Dos hilos CPU; tiempo total 56.126 s. CUDA no se inicializó.

Inicialización aleatoria, cero pasos de optimizador. No es una evaluación de detección ni una admisión para entrenar en GPU.

| Etapa | Tiempo CPU, s |
|---|---:|
| construction | 28.314 |
| attention_reference | 0.187 |
| pyramid_resize_parity | 0.261 |
| depth_bins | 0.002 |
| eight_class_backward | 5.684 |
| empty_target_backward | 4.237 |
| focal_contract | 1.720 |
| checkpoint_replay | 1.005 |
| real_train_frame_backward | 14.696 |

Atención fp64 frente a la referencia PyTorch del autor: error forward máximo 8.882e-16; errores de primeros gradientes [0.0, 0.0, 0.0]. La extensión CUDA compilada no se comparó.

Se ejecutó el forward/backward completo con ocho clases sintéticas y con targets vacíos: 550 consultas y ocho logits por consulta durante entrenamiento. En evaluación se comprobaron 50 consultas, `targets=None` y replay exacto tras guardar, perturbar y restaurar los pesos.

Sensibilidad geométrica: error máximo frente al cambio esperado por fy 5.960e-07; cambiar solamente fx conservó las salidas exactamente. El fixture de 96×160 tiene focales distintas. La pirámide se ajustó al tamaño receptor después del fallo v1; conserva valores y gradientes del aumento ×2 para niveles compatibles de una entrada 640×384.

Frame TRAIN real: `xkiVthsGwf:000001113_000434196`, 7 targets admitidos y profundidad máxima 249.153183 m. La etapa incluye lectura, preparación, forward/backward a 640×384, otra inferencia sin etiquetas y decodificación; su tiempo no es una medida aislada de latencia ni una estimación de velocidad GPU.

Con pesos aleatorios se emitieron 0 cajas válidas de 50 consultas. Las cajas con tamaño o profundidad no positivos fueron rechazadas por el decoder. El port aún necesita pesos admitidos, aprendizaje y evaluación de ocho clases.

Los bins comparten el rango de 300 m entre predictor y pérdida. Se comprobaron las fronteras de rango, profundidades lejanas y entradas no finitas. Se preserva el intento v1 fallido, su código y su traceback.

Siguiente paso técnico: inspección y transferencia trazable del checkpoint multiclase oficial. La eventual comparación GPU requiere un presupuesto y coste medidos, y la decisión posterior al ensayo KITTI en curso.

Referencias de API: [grid_sample de PyTorch 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.grid_sample.html). Los resultados numéricos de esta página proceden del JSON del ensayo local.
