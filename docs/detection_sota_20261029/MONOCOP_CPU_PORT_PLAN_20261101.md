# Preparación CPU de MonoCoP

El ensayo KITTI sigue activo. Se adelanta únicamente la preparación técnica CPU
del competidor condicional identificado en la [revisión de fuentes](MONOCOP_FEASIBILITY_20261101.md).
La decisión sobre otro entrenamiento GPU sigue pendiente del resultado KITTI y
de una medición de coste. No se altera su productor, receta, cola ni presupuesto.

Esta etapa usa código del commit `7d3932444f7ea94be564ae61a0b334fc6673213b`,
verificado contra el recibo de descarga, en una copia nueva. No descarga pesos:
la red se construye con inicialización aleatoria. No mide AP ni hace pasos de
optimizador. Conserva la arquitectura R50-vd, tres capas encoder/decoder,
50 consultas por grupo y 11 grupos de entrenamiento.

Adaptaciones explícitas: ocho clases; carga de pesos separada del constructor;
atención PyTorch existente; dispositivos derivados de tensores; rango compartido
de 300 m entre predictor y DDNLoss; escala de mapas de profundidad derivada de su
shape; `fy` para la altura proyectada; paso explícito del número de grupos al
modelo, transformer y criterio; manejo de targets vacíos en la pérdida de tamaño.
El código original y el parche quedan separados y con hashes. No se atribuirá
una eventual diferencia frente a MonoDGP exclusivamente al encadenamiento.

Comprobaciones previstas antes de cualquier admisión GPU:

1. Construcción CPU sin red y sin inicializar CUDA, con las ocho salidas de clase.
2. Paridad forward y de primeros gradientes de la atención frente a la referencia
   PyTorch incluida por MonoCoP, con entradas fp64 pequeñas. No equivale a probar
   la extensión CUDA compilada.
3. Forward/backward completos con ocho clases sintéticas y con targets vacíos;
   pérdidas, salidas y gradientes finitos.
4. Sensibilidad geométrica del modelo real: variar `fy` debe cambiar únicamente
   el término esperado de profundidad; variar solo `fx` no debe cambiar esa
   salida. Se usan focales distintas y tamaño de mapa no KITTI.
5. Bins coherentes en fronteras y targets lejanos; replay exacto de pesos guardados
   en modo evaluación. La prueba usa `targets=None` en inferencia.
6. Un frame TRAIN real del smoke registrado, a 640×384, con un target admitido
   por encima de 200 m: forward, pérdida y backward finitos en CPU.

Límite: dos hilos CPU y 600 segundos por intento de comprobación, revisados entre
etapas; una operación en curso puede rebasarlo. Los fallos quedan registrados.
No se lanzan barridos de arquitectura, pesos, umbrales, GPU ni gasto cloud.
El éxito de esta etapa solo acredita compatibilidad de ejecución y contratos.

## Corrección tras el primer intento

`monocop_cpu_smoke_v1` construyó la red, verificó la atención y sus primeros
gradientes frente a la referencia PyTorch y comprobó los bins. Falló en el
forward de 96×160: la pirámide duplicaba un mapa de ancho 3 hasta 6 para sumarlo
a otro de ancho 5. El fallo y sus fuentes se conservan.

El segundo port usa el tamaño explícito del mapa receptor, manteniendo
interpolación bilinear y `align_corners=True`. La repetición conserva el fixture
de 96×160 y añade paridad exacta de valores y gradientes con el aumento original
×2 en las dimensiones de la pirámide correspondiente a 640×384. No cambian las
formas de los pesos ni se inicia entrenamiento GPU.
