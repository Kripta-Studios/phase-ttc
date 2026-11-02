# Primer screening MonoCoP de ocho clases

Se admite un ensayo local tras superar el gate técnico de MonoCoP. El primer
arranque falló antes de GPU por una ruta Windows; el primer entrenamiento guardó
su checkpoint con un nombre erróneo. El segundo conservó los estados 0/16/32 y
aprendió el batch. Su verificador confundió tuplas del perfil con listas JSON.
La auditoría independiente cargó el último checkpoint y reprodujo exactamente
predicciones y métricas en las 16 imágenes, sin actualizaciones adicionales.
Los tres fallos y la recuperación se conservan en
[la evidencia](evidence/monocop_gpu_execution_20261101/ARCHIVE.json).

Hipótesis: otra arquitectura monocular y la supervisión externa de coche, peatón
y ciclista pueden mejorar la detección y geometría eAP. Es una comparación de
candidatos: cambian conjuntamente arquitectura e historial de preentrenamiento.
No permite atribuir una diferencia solo al backbone o a una clase preentrenada.
La correspondencia Cyclist→bicycle tampoco garantiza equivalencia de dominio.

Se parte del checkpoint de transferencia CPU anterior a cualquier entrenamiento
eAP. No se reutilizan los pesos adaptados a las dos imágenes del smoke. Las otras
cinco clases y el clasificador de profundidad se inicializaron según el recibo
de transferencia. Rango 300 m, ocho clases y GT de evaluación crudo sin filtro
de rango; no hay una cabeza de velocidad ni capacidad demostrada a 400 m.

Población fija: las mismas 4.096 imágenes TRAIN y 512 de selección usadas en
los controles de rango e inicialización KITTI. Particiones por secuencia;
calibración y test quedan fuera. Esta selección pública ya se ha utilizado en
desarrollo, por lo que no constituye una prueba independiente.

Receta: seed 7, 640×384, batch 2, FP32, TF32 desactivado, 2.048 actualizaciones
(4.096 exposiciones), AdamW, LR 0,00005, multiplicador backbone 0,1, weight decay
0,0001, clip 1, warmup/cosine. Sin augmentations ni ranker. Las anotaciones entran
en la pérdida, nunca en el forward. Se conserva la caché compacta ya probada,
con preparación anticipada de un batch y 4 GiB de presupuesto de imágenes CPU.
Cada inserción exige equivalencia exacta con el preprocesamiento original.

Los controles anteriores comparten población, presupuesto adicional y receta
de optimización; no igualan preentrenamiento ni arquitectura. Sus flags cuDNN
TF32 no estaban registrados igual: no se presupone equivalencia numérica ni se
atribuye una diferencia de tiempo exclusivamente a la arquitectura.

Métrica primaria: mAP alineada macro de ocho clases, en 0/512/1.024/1.536/2.048.
Se selecciona el mayor valor; en empate se conserva el checkpoint anterior.
Se registra AP, TP, FP y recall por clase desde los JSON originales.

Regla de avance: al terminar el presupuesto, exigir al menos +0,005 mAP frente
al incumbente con ranking y al mejor control Dfull 300 m del mismo panel.
Solo entonces se prepara confirmación de ese checkpoint en las 23.958 imágenes
de selección completa. Allí debe mantenerse +0,005 frente al incumbente y
revisarse cada caída de AP por clase superior a 0,01. El control 300 m no tuvo
confirmación completa; no se fingirá una cifra completa para él. La regla es
operacional, no una prueba de significancia. No hay promoción automática.

Límites: 7.200 segundos locales, un único escritor GPU y máximo 9 GiB de memoria
CUDA asignada. El tiempo incluye construcción, entrenamiento, evaluaciones y
checkpoints; una operación iniciada puede rebasar el límite. NaN/OOM, cambios
de fuentes o contratos detienen el proceso. STOP y presupuesto conservan el
último estado; resume mantiene optimizador, scheduler, sampler, RNG y tiempo
consumido. No hay semillas, épocas, gasto cloud ni envío oficial automáticos.

El ritmo se medirá en el panel real, separando preparación, forward,
matching/pérdida, backward, optimizer, validación y escritura. Los tiempos del
smoke repetido no constituyen una ETA válida de acceso a todo el dataset.
