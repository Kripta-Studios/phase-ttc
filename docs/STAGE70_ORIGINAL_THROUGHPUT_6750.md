# Diagnóstico de velocidad del original: DIRECT outer1, checkpoint 6750

Fuente verificada: `ORIGINAL_THROUGHPUT_AT_6750.json`, SHA256 `66cdaa43ede6f7d9394f3772dddba4ee9b313b591ab65ca7c01c5a120bd0f832`.
Se leyó el estado de un checkpoint TRAIN inmutable para extraer tiempos;
no se evaluaron predicciones ni resultados exteriores. El proceso original
36072 seguía activo al observarlo. No se modificó ningún archivo del original.

- Últimos 1000 updates: media 4.078 s,
  mediana 4.118 s, p95 4.565 s.
- Cuatro intervalos de 250 updates: 4084.953 s de reloj frente a
  4078.047 s dentro del temporizador de updates.
  Sólo 6.905 s (0.169 %) quedan fuera del temporizador,
  incluyendo checkpoints y otras operaciones entre updates.
- Muestreo de proceso durante 10 s: 1.695
  núcleos equivalentes de CPU; lectura lógica 70675.5
  B/s y escritura 22034.4 B/s.
  Estas cifras no miden por separado fallos de página de memmaps ni latencia de fsync.
- Una consulta puntual nvidia-smi mostró GPU 4 %, 3263/12227 MiB y 36,70 W.
  No es una media de utilización ni permite atribuir por sí sola todo el tiempo.

La evidencia descarta los checkpoints periódicos como causa principal en estos
intervalos. Hacerlos asíncronos tendría un beneficio máximo muy pequeño incluso
si eliminara todo el tiempo entre updates, y añadiría riesgo de publicación.

El código congelado prepara cada muestra en CPU, aplica thinning binomial
float64 y normalización, y hace ocho transferencias/microbatches consecutivos.
Cada microbatch publica su traza con fsync; cada update publica dos estados del
ledger. Todo eso está dentro del temporizador observado. La combinación de
CPU activa y GPU puntualmente baja es compatible con alimentación serial y
sincronizaciones, pero la telemetría disponible no separa preparación, kernels,
transferencias y latencia de los registros. No atribuyo porcentajes inventados
ni afirmo haber demostrado un único cuello de botella.

El único ajuste a estudiar con este diagnóstico es prefetch acotado de entradas
CPU→GPU, permitido por document11. Debe conservar tokens y orden, generadores,
augmentations, batch16/micro2, escalado de loss y precisión. Antes de usarlo
requiere reserva física, paridad de valores/gradientes y recuperación exacta;
para el original exige además una enmienda de ejecución visible. No se ha
implementado ni activado como consecuencia de estas observaciones. No propongo
cambiar precisión, receta, resolución ni reducir la persistencia del ledger.

Este resultado corresponde a DIRECT en este host/carga concurrente; no estima
la velocidad de PPS, pretexto o comparadores. La siguiente comprobación que
permitiría decidir un ajuste sería separar los tiempos de preparación y GPU
en un smoke TRAIN no seleccionable ya autorizado, sin modificar el fit activo.
