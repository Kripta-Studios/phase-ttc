# PHASE-TTC · Cómo continuar en el benchmark con regiones suministradas

Corte del 22 de noviembre de 2026. Este plan parte de [Trabajo realizado](TRABAJO_REALIZADO.md) y de [Hallazgos y límites](HALLAZGOS_Y_LIMITES.md). Cubre el benchmark GarlTTC, donde cada consulta trae su región. El caso sin regiones tiene su propio plan en [Cómo continuar sin cajas](COMO_CONTINUAR_SIN_CAJAS.md).

## Dónde estamos

| Entrada | Overall MiD en test | Qué usa |
|---|---:|---|
| Garl-TTC, referencia publicada | 45,03 | eventos + RGB, dos observaciones |
| PHASE-TTC original (`1016512`) | 35,80 | eventos + RGB, 650 ms |
| Solo cajas (`1063261`) | 27,98 | historia de cajas, 600 ms |
| RGB + geometría (`1058935`) | **23,47** | RGB + historia de cajas |

El primer puesto es real y está verificado. Lo que todavía no sostiene es la frase «estado del arte» sin matices, por tres motivos:

1. **La ventaja es sobre todo historia de cajas.** Solo cajas ya da 27,98. Las cajas de TRAIN40 son proyecciones exactas de cuboides 3D anotados, así que su expansión es una señal casi limpia de aproximación.
2. **Falta la comparación con la misma información.** Garl de eventos + cajas empata con PHASE RGB + cajas en desarrollo (0,70 MiD, intervalo con cero), con Garl dentro de muestra.
3. **No hay segundo conjunto independiente.** FCWD está consumido como desarrollo y TSTTC no es accesible.

## Orden de trabajo recomendado

### 1. Cerrar la comparación exterior emparejada (prioridad máxima)

Es lo único que convierte el 23,47 en una afirmación de método y no de información. La receta está en [`matched_outer/NEXT_RUN.json`](matched_outer/NEXT_RUN.json).

- Completar las cuatro integraciones pendientes que lista ese fichero: caché rotatoria por pliegue, productores conscientes del pliegue, ajuste completo del productor RGB de PHASE y el apilado común.
- Entrenar por pliegue exterior los dos productores (PHASE RGB y Garl RGB / RGB+eventos), con semillas 7, 13 y 21, y la misma cabeza de cajas encima de ambos.
- Requisitos: ventana de GPU dedicada y unos 57 GB de espacio temporal. No mirar ninguna precisión exterior hasta cerrar los cinco pliegues.
- Resultado esperado: el contraste `PHASE_RGB_BOX − GARL_RGB_BOX` con intervalo por secuencias. Se publica salga como salga.

### 2. Preparar un único candidato nuevo para el test

Hay tres mejoras medidas en desarrollo que aún no se han enviado:

| Mejora | Efecto en pliegues frente a RGB + geometría |
|---|---|
| Geometría robusta + escala de contenido de eventos sobre RGB | −2,42 [−2,90; −1,93] |
| Puerta de fiabilidad en cajas truncadas | −0,57 [−0,77; −0,38] sobre todas las filas; −2,81 en truncadas |
| Promedio de semillas | sin medir como brazo; las colas cambian entre semillas |

- Combinarlas en un solo brazo y medirlo con los pliegues exteriores completos del paso 1, no con los de cabeza. Así el intervalo no hereda productores que vieron el bloque evaluado.
- Regla previa: enviar solo si el extremo superior del intervalo frente a RGB + geometría queda por debajo de −2,0 MiD. El brazo sin puerta se quedó en −1,93; no rebajar el umbral después de verlo.
- Un envío, con recibo previo según [ADR-0047](../decisions/ADR-0047-garlttc-submission-discipline.md). Sería el cuarto de la cuenta: cada envío adicional gasta independencia del test.
- Antes de empaquetar, fijar el camino de entrada. En la reconstrucción desde imágenes solo 5 de 12 consultas cumplieron la tolerancia ([reproducción](geo_reproduction/REPORT.md)); el paquete debe salir de descriptores congelados con huella.

### 3. Ideas de modelo por probar, solo en pliegues

Son hipótesis, ninguna medida todavía:

- **Historia corta.** 18.672 de 88.744 filas tienen menos de 600 ms de cajas. Entrenar la cabeza con historia enmascarada al azar para que no degrade ahí.
- **Segundo orden.** Añadir la curvatura del logaritmo del tamaño (aceleración relativa), que la pendiente lineal ignora.
- **Banda crucial.** Es la mitad del peso de la métrica y donde está el 90 % de la diferencia entre pliegues y test. Probar un experto dedicado a TTC menor de 3 s.
- **Saturación real.** Las 46 filas saturadas de eAP siguen sin resolverse. Hacen falta más filas reales, no más ajuste sobre FCWD.

Cada idea entra como brazo con protocolo previo y se descarta si su intervalo incluye el cero.

### 4. Conseguir una cohorte independiente

Sin ella no hay afirmación de generalización. El [protocolo de cohorte](independent_cohort/PROTOCOL.md) ya fija las condiciones.

- Boreas queda descartado como cohorte: una sola sesión física.
- Candidatos a investigar: nuScenes y Argoverse 2, con TTC de referencia derivado de LiDAR y registros de conducción completos reservados. Sería una evaluación de transferencia definida por el proyecto, no un benchmark oficial.
- Congelar modelo, población y construcción de la verdad antes de abrir ninguna etiqueta.

### 5. Bajar la latencia del modelo ganador

RGB + geometría tarda 28,8 ms de mediana en cómputo aislado y no entregó ninguna respuesta en plazo bajo GPU compartida. El 75 % del tiempo es la rama RGB.

- Repetir el ensayo de servicio en GPU dedicada, que es la medida que falta.
- Optimizar la rama RGB (media precisión, compilación del grafo, o destilarla a un productor más pequeño) y comprobar que el MiD en pliegues no cambia.
- Solo cajas ya cumple los 20 ms en CPU (23.992 de 24.000 ofertas): es la opción a presentar cuando el plazo importe.

## Qué se podrá afirmar

| Si se cumple | Afirmación permitida |
|---|---|
| Estado actual | Primer puesto en GarlTTC Test con regiones suministradas, a fecha de la clasificación archivada |
| Paso 1 favorable | PHASE supera a Garl-TTC con la misma historia de cajas en pliegues exteriores completos |
| Paso 1 nulo o adverso | La ventaja procede de usar la historia de cajas, no de la arquitectura |
| Paso 4 favorable | El resultado se transfiere a sesiones ajenas al diseño |
| Paso 5 favorable | Cumple el plazo de 20 ms en el hardware medido |
