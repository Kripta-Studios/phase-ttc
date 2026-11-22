# PHASE-TTC · Cómo continuar sin regiones suministradas

Corte del 22 de noviembre de 2026. Este plan parte del piloto de TTC por objeto descrito en [Hallazgos y límites](HALLAZGOS_Y_LIMITES.md) y en su [análisis de pérdidas](per_object_events/LOSS_ANALYSIS.md). El benchmark con regiones tiene su propio plan en [Cómo continuar en el benchmark](COMO_CONTINUAR_BENCHMARK.md).

## Qué dice realmente el piloto

La cadena automática dio un TTC bien asociado en 3 de 165 observaciones. El desglose indica dónde está el problema:

| Etapa | Pasan | Lectura |
|---|---:|---|
| Hay eventos dentro del objeto | 165/165 | La señal existe |
| Propuesta con IoU ≥ 0,5 | 16/165 | **Aquí se pierde casi todo** |
| Pista que admite TTC | 3/165 | Consecuencia de lo anterior |
| Mismo seguidor con cajas de referencia | 160/165 | El seguidor funciona si la caja es buena |
| Cabezas con pistas anotadas | 165/165, error medio 1,6–1,9 s | La estimación de TTC funciona |

Tres límites de esta lectura:

- **El detector del piloto no aprende.** Es un umbral de actividad sobre celdas de 8 × 8 píxeles con cierre morfológico. El 3/165 mide ese detector, no lo que puede dar un detector entrenado.
- **El piloto es pequeño.** 12 secuencias y 8 instantes puntuados por secuencia. Sirve para localizar el fallo, no para estimar tasas.
- **Las cabezas nunca vieron cajas imperfectas.** Se entrenaron con regiones anotadas, que son proyecciones exactas de cuboides 3D.

## Orden de trabajo recomendado

### 1. Fijar la evaluación antes de tocar el detector

- Partir TRAIN40 por secuencias completas: un bloque para desarrollar el detector y otro, sellado, que no se mira hasta el final. Las 12 secuencias del piloto pasan al bloque de desarrollo.
- Puntuar secuencias enteras, no ocho instantes.
- Métricas fijas, contando todo objeto de referencia aunque no reciba salida: cobertura de localización a IoU 0,5 y 0,3, cambios de identidad, cobertura de TTC, error de las salidas emitidas y tiempo desde el sensor hasta la respuesta.
- Sellar las predicciones antes de abrir las anotaciones, como en el piloto.

### 2. Medir cuánta precisión de caja necesitan las cabezas

Experimento barato y previo a cualquier detector: perturbar las cajas anotadas con niveles crecientes de desplazamiento, cambio de tamaño y pérdida de instantes, y medir el error de TTC de las dos cabezas congeladas.

- Da el IoU mínimo que el detector debe alcanzar para que el TTC sea útil.
- Dice si la escala de contenido de eventos tolera mejor una caja demasiado grande que la geometría de bordes. De las 64 propuestas grandes del piloto, 50 cubrían al menos el 80 % del objeto: si la escala de contenido aguanta ese caso, parte del problema se relaja.

### 3. Detector aprendido (camino principal)

El piloto propone agrupar por movimiento local coherente. Vale como control clásico a igual presupuesto de regiones, pero el techo esperable es bajo. La vía con más recorrido es entrenar un detector:

- **Etiquetas disponibles.** Los cuboides 3D de eAP proyectados dan cajas exactas en todos los instantes anotados de TRAIN40. Usarlos como objetivo de entrenamiento del detector, nunca como entrada en evaluación.
- **Punto de partida.** Existe un detector monocular piloto en `phase-ttc-detection-local` con su adaptador causal y su replay ([auditoría](geometry_service/DETECTOR_AUDIT.md)). Se entrenó con dos secuencias y tarda unos 47 ms por pasada: sirve de base de código, no de modelo.
- **Variantes a comparar**, con el mismo presupuesto de regiones por instante: RGB, eventos, y eventos + RGB. La de solo eventos es la que responde a la pregunta original del proyecto.
- **Disciplina de pliegues.** El detector que puntúa una secuencia no puede haberla visto. Mismo esquema de cinco bloques de ocho secuencias.

### 4. Reentrenar las cabezas con cajas de detector

- Ajustar las cabezas sobre las cajas fuera de pliegue del detector, o con las perturbaciones del paso 2 como aumento, para que aprendan a desconfiar de bordes ruidosos.
- Mantener como entrada la escala de contenido de eventos, que no depende de los bordes.
- Comparar tres condiciones en las mismas secuencias: cabezas congeladas + detector, cabezas reentrenadas + detector, y cabezas con pistas anotadas como techo.

### 5. Evaluación final y comparación justa

- Una sola evaluación en el bloque sellado, con todo congelado por huella.
- **Comparador:** Garl-TTC alimentado con las mismas cajas del detector. Es la única comparación entre iguales en este régimen; no existe clasificación pública sin regiones.
- **Latencia de la cadena completa**, desde la disponibilidad del sensor hasta el TTC por objeto, que hoy no está medida.
- Para afirmar generalización hace falta la cohorte independiente del [protocolo](independent_cohort/PROTOCOL.md), igual que en el plan del benchmark.

## Criterios para seguir o parar

| Resultado | Decisión |
|---|---|
| Paso 2: las cabezas toleran cajas con IoU ≈ 0,5 | Seguir con el paso 3 tal cual |
| Paso 2: el error se dispara con poco ruido | Adelantar el paso 4 y entrenar cabezas con ruido antes de invertir en el detector |
| Paso 3: cobertura a IoU 0,5 por debajo del 50 % en desarrollo | No pasar al bloque sellado; revisar el detector |
| Paso 5: cobertura de TTC alta pero error muy por encima del techo con pistas anotadas | El límite es la calidad de caja, no la cobertura |

## Qué se podrá afirmar

Ahora: con regiones suministradas el sistema lidera el benchmark, y sin ellas la cadena de eventos del piloto no localiza los objetos. Tras este plan, como mucho: un sistema completo de detección, seguimiento y TTC evaluado en secuencias selladas de eAP y comparado con Garl-TTC sobre las mismas cajas. No será un resultado de benchmark oficial, porque el test privado solo se puntúa con sus propias regiones.
