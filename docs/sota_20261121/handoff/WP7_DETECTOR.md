# WP7 · Detector y seguidor entrenados

Antes de empezar, leer el [traspaso](README.md), el [informe del piloto](../per_object_events/REPORT.md), su [análisis de pérdidas](../per_object_events/LOSS_ANALYSIS.md) y el resultado de [WP6](WP6_TOLERANCIA_A_CAJAS.md).

## Pregunta

¿Puede un detector entrenado localizar y seguir los objetos de eAP con la calidad que las cabezas de TTC necesitan, sin recibir ninguna región anotada en inferencia?

## Qué se sabe

- El detector del piloto es un umbral de actividad de eventos sobre celdas de 8 × 8 píxeles con cierre morfológico. Localizó 16 de 165 observaciones a IoU 0,5.
- Todas las regiones anotadas contenían eventos: la señal existe.
- El seguidor del proyecto, con cajas de referencia y sin identidades, quedó listo para emitir en 160 de 165 observaciones. El seguimiento no es el primer problema.
- Las cajas de TRAIN40 son proyecciones exactas de los cuboides 3D anotados ([`benchmark_scope`](../benchmark_scope/REPORT.md)): hay etiquetas de detección para todos los instantes anotados.
- Existe un detector monocular piloto en el árbol `phase-ttc-detection-local` (rama `dev/eap-detection-local-20261029`), con adaptador causal en `operational/detection_phase_service/` y replay en `scripts/replay_detection_phase_service.py`. Se entrenó con dos secuencias, tarda unos 47 ms por pasada y no es comparable ([auditoría](../geometry_service/DETECTOR_AUDIT.md)). Sirve como base de código.

## Paso 0 · Evaluación fijada antes de entrenar

Escribir `docs/sota_20261121/detector/PROTOCOL.md` con:

- **Partición por secuencias completas de TRAIN40.** Un bloque de desarrollo (pliegues para entrenar y validar el detector) y un **bloque sellado** de 8 secuencias que no se abre hasta [WP8](WP8_CABEZAS_Y_EVALUACION_FINAL.md). Elegir el bloque sellado por una regla independiente del contenido (por ejemplo, orden por huella del identificador). Las 12 secuencias del piloto van al bloque de desarrollo.
- **Población:** todos los instantes anotados de cada secuencia, no ocho por secuencia.
- **Métricas**, contando todo objeto de referencia aunque no reciba salida:
  - cobertura de localización a IoU 0,5 y 0,3, y precisión con un presupuesto fijo de regiones por instante (32, como el piloto);
  - cambios de identidad y fracción de pista seguida;
  - fracción de observaciones con historia suficiente para estimar TTC (600 ms);
  - tiempo por instante.
- Emparejamiento uno a uno por IoU, sin usar el valor del TTC.
- El aviso de que las regiones sin pareja pueden ser objetos reales sin anotación de TTC: la precisión es una cota inferior.

## Paso 1 · Control clásico

Implementar la agrupación causal por movimiento local coherente sobre varias ventanas, propuesta en el análisis de pérdidas, con el mismo presupuesto de regiones. Es el control frente al que se mide el detector entrenado, y es barato.

## Paso 2 · Detector entrenado

Comparar tres entradas con la misma arquitectura base y el mismo presupuesto:

| Variante | Entrada | Responde a |
|---|---|---|
| RGB | fotograma actual | Techo práctico con el sensor convencional |
| Eventos | representación de eventos causal (recuento o vóxeles de la ventana previa) | La pregunta original del proyecto |
| Eventos + RGB | ambas | Si los eventos añaden algo a la localización |

Indicaciones:
- Partir de un detector 2D preentrenado y ligero. Registrar con qué datos se preentrenó: si incluyen un conjunto candidato de [WP4](WP4_COHORTE_INDEPENDIENTE.md), ese conjunto deja de ser independiente para esta cadena.
- Objetivo: las cajas proyectadas de los cuboides, recortadas al sensor. Para eventos, proyectar en el plano del sensor de eventos y comprobar la alineación en una muestra visual.
- Entrenar por pliegues dentro del bloque de desarrollo, de modo que haya cajas de detector **fuera de pliegue** para todas sus secuencias: WP8 las necesita para reentrenar las cabezas.
- Atender a lo que diga WP6: si las cabezas son sensibles al ruido por fotograma, añadir suavizado temporal causal de las cajas en el seguidor y medirlo como brazo.
- Presupuesto: memoria de GPU de 12 GiB, y disco limitado. Si hace falta una caché de representaciones de eventos, estimar su tamaño antes y pedir espacio al propietario si no cabe.

## Paso 3 · Seguimiento

Reutilizar el seguidor del proyecto. Medir el efecto de sus parámetros (umbral de asociación, confirmación) solo en el bloque de desarrollo.

## Regla de decisión

- El detector entrenado debe superar al control clásico en cobertura a IoU 0,5 con intervalo por secuencias que excluya el cero.
- Para pasar a WP8: cobertura a IoU 0,5 de al menos el 50 % de las observaciones en el bloque de desarrollo, fuera de pliegue, o el umbral que WP6 haya fijado como utilizable. Por debajo, no se abre el bloque sellado.

## Reglas específicas

- Las cajas anotadas son objetivo de entrenamiento y referencia de evaluación, nunca entrada en inferencia.
- El detector que puntúa una secuencia no la ha visto.
- El bloque sellado no se usa para nada en este paquete, ni siquiera para mirar imágenes.

## Entregables

`PROTOCOL.md`, `REPORT.md`, `RESULTS.json`, lista de secuencias de cada bloque con su huella, cajas de detector fuera de pliegue para el bloque de desarrollo (en `artifacts/`, con huella en el informe), pesos con registro de secuencias vistas, código en `operational/detector/` con pruebas (partición disjunta, causalidad de la representación de eventos, emparejamiento).
