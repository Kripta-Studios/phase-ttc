# Prueba técnica GPU de MonoCoP: preparación condicional

El port y la transferencia multiclase pasaron las pruebas CPU. Este documento
fija la siguiente prueba antes de observar sus resultados. No admite su ejecución
mientras siga activa la cola KITTI, ni autoriza un screening de calidad.

El punto de partida es el checkpoint inicializado CPU, SHA-256
`e0de1b40d0b0316eaaf88d8133b333a00b825baf080fc19604b55d51c321b80a`.
Contiene pesos de MonoCoP_All con mapeo de tres clases, otras cinco filas nuevas,
rango coherente de 300 m, 50 consultas, 11 grupos y tres capas decoder. No ha
recibido pasos de optimizador en eAP. No se añade una cabeza de velocidad.

Condición previa: cerrar y revisar KITTI, verificar sus recibos y la ausencia
de sus procesos escritores, y registrar una admisión local específica. La CLI
de validación CPU no permite iniciar la GPU. No se amplían presupuestos o se
lanzan otros modelos automáticamente al pasar el smoke.

La prueba usa las mismas dos imágenes TRAIN y 16 de selección del smoke KITTI.
Una imagen TRAIN contiene un target a 249,153 m. Se verifica separación por
secuencia, hashes de RGB/etiquetas y metadatos. La selección se evalúa contra
todas las anotaciones crudas con el perfil alineado, sin filtrarlas a 300 m.

Receta técnica: seed 7, 640×384, FP32, batch 2, 32 actualizaciones, AdamW,
LR 0,00005, multiplicador de backbone 0,1, weight decay 0,0001, clip de gradiente
1,0 y warmup/cosine idéntico al smoke anterior. Sin augmentation ni ranker.
El forward recibe `targets=None`; las etiquetas solo entran en la pérdida.

Etapas y gates:

1. Verificar contratos y procedencia antes de reservar GPU. Construir sin red y
   cargar estrictamente el checkpoint CPU, incluida la configuración efectiva.
2. Comprobar atención PyTorch CPU/GPU, forward y primeros gradientes, con las
   mismas entradas fp32/fp64 y tolerancia absoluta/relativa 2e-5. No se compara
   con la extensión CUDA compilada de los autores.
3. Comparar todas las salidas del modelo completo, en evaluación sin targets,
   sobre el frame TRAIN lejano, CPU contra GPU: atol=1e-4, rtol=1e-4. Registrar
   los errores máximos por salida. Restablecer las semillas antes de entrenar
   para que las pruebas numéricas no alteren el RNG del entrenamiento.
4. Evaluar en 0/16/32; conservar checkpoints atómicos de inicialización, intermedio
   y último. Registrar cada pérdida, norma de gradiente, tokens, tiempos y estado.
   Exigir valores finitos y media de pérdida de las últimas ocho actualizaciones
   menor que la de las primeras ocho. Esto comprueba aprendizaje del batch, no
   generalización, ni aprendizaje útil de cada una de las ocho clases.
5. Restaurar el último checkpoint y repetir las 16 imágenes: predicciones y
   métricas deben coincidir exactamente. Comprobar los bins del target lejano.
   El gate no depende de subir mAP en las 16 imágenes.

Presupuesto local: 600 segundos comprobados entre etapas y actualizaciones,
incluidos preparación, validaciones y replay; una operación en curso puede
rebasar ese límite. Máximo 9 GiB de memoria CUDA asignada medida. No lanzar otro
proceso GPU, gasto cloud, semillas extra, selección completa o envío oficial.
Ante interrupción o presupuesto agotado, guardar el progreso y no continuar
automáticamente; un resume conserva optimizador, scheduler, sampler, RNG y el
tiempo consumido. Un STOP también conserva un checkpoint.

Se medirán preparación/transferencia, forward, pérdida/matching, backward,
optimizer, validación, escritura y tiempo de ciclo completo. Separar primeros
pasos de régimen caliente. Dos imágenes repetidas no caracterizan por sí solas
la lectura de todo TRAIN; no inferir una ETA exacta para otro experimento a partir
de ese tiempo. La revisión posterior decidirá receta, control, coste y criterio
de promoción del eventual screening de ocho clases.
