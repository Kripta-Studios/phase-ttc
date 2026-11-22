# WP8 · Cabezas con cajas de detector y evaluación sellada

Antes de empezar, leer el [traspaso](README.md) y los resultados de [WP6](WP6_TOLERANCIA_A_CAJAS.md) y [WP7](WP7_DETECTOR.md).

## Pregunta

Con detector y seguidor propios, ¿qué cobertura y qué error de TTC por objeto da la cadena completa en secuencias que nadie ha mirado, y cómo queda frente a Garl-TTC con las mismas cajas?

## Requisitos de entrada

- WP7 cerrado con su regla cumplida.
- Cajas de detector fuera de pliegue para todas las secuencias del bloque de desarrollo.
- Bloque sellado de 8 secuencias intacto.

## Parte A · Reentrenar las cabezas (solo bloque de desarrollo)

Brazos, todos con la misma arquitectura de cabeza, pérdida por banda y pliegues:

| Brazo | Cajas de entrenamiento | Cajas de evaluación |
|---|---|---|
| A0 | anotadas | de detector (cabezas actuales, sin tocar) |
| A1 | anotadas con las perturbaciones de WP6 | de detector |
| A2 | de detector, fuera de pliegue | de detector |
| A3 | mezcla de anotadas y de detector | de detector |
| Techo | anotadas | anotadas |

- Asignar a cada pista del detector el TTC de la referencia emparejada por IoU para entrenar; las pistas sin pareja no entrenan.
- Mantener la escala de contenido de eventos como entrada en una variante de cada brazo: no depende de los bordes de la caja.
- La cabeza y el detector que puntúan una secuencia no la han visto.
- Elegir **un** brazo primario por una regla escrita antes de medir (menor Overall MiD entre las salidas emitidas, con empate a favor de la mayor cobertura).

## Parte B · Comparador

Garl-TTC visual y Garl-TTC completo alimentados con **las mismas cajas del detector**. Usar el modelo publicado y declarar que vio TRAIN40 en su entrenamiento; si [WP1](WP1_COMPARACION_EXTERIOR.md) dejó productores de Garl por pliegue, usar esos y decirlo.

## Parte C · Evaluación sellada, una sola vez

1. Congelar por SHA-256 detector, seguidor, cabeza primaria, comparador, evaluador y política de retrasos. Registrar las huellas en `seal.json` y pedir al orquestador el commit **antes** de puntuar.
2. Generar predicciones para las 8 secuencias selladas y añadir su huella al sello.
3. Abrir las anotaciones y puntuar una vez.

Métricas, contando todo objeto de referencia:
- cobertura de localización a IoU 0,5 y 0,3;
- cambios de identidad;
- cobertura de TTC (observaciones con salida asociada correctamente);
- Overall MiD, error relativo y error absoluto de las salidas emitidas, por banda;
- las mismas métricas del techo con pistas anotadas, sobre las mismas observaciones;
- una métrica conjunta declarada de antemano que penalice lo no emitido (por ejemplo, error con un valor de castigo fijo para las observaciones sin salida), para que cobertura y error no se puedan intercambiar a conveniencia.

Intervalos por remuestreo de las 8 secuencias, advirtiendo de que son pocos remuestreos distintos.

## Parte D · Latencia de la cadena completa

Medir desde la disponibilidad del sensor hasta el TTC por objeto: representación, detector, seguidor, entradas de la cabeza y cabeza. Mismas convenciones que [WP5](WP5_LATENCIA.md), con la GPU dedicada.

## Regla de decisión

| Resultado en el bloque sellado | Lectura |
|---|---|
| Cobertura de TTC ≥ 50 % y error de lo emitido dentro de 1,5 veces el techo | Cadena utilizable; proponer evaluación en la cohorte de [WP4](WP4_COHORTE_INDEPENDIENTE.md) |
| Cobertura alta y error muy por encima del techo | El límite es la calidad de caja |
| Cobertura baja | El límite sigue siendo la localización; volver a WP7 con secuencias nuevas, no con las selladas |
| Mejor que Garl-TTC con las mismas cajas, con intervalo que excluya el cero | Ventaja de PHASE en el régimen sin regiones, en eAP |

## Lo que este paquete no da

- No es un resultado del benchmark oficial: el test privado solo se puntúa con sus propias regiones.
- No es generalización: las secuencias selladas son del mismo conjunto y de las mismas campañas de captura que el entrenamiento. Para eso hace falta WP4.
- Una vez abierto, el bloque sellado pasa a ser desarrollo. No hay segunda evaluación.

## Entregables

`docs/sota_20261121/end_to_end/PROTOCOL.md`, `seal.json`, `REPORT.md`, `RESULTS.json`, predicciones selladas, tabla por etapa de dónde se pierden las observaciones (como la del piloto), código en `operational/end_to_end/` con pruebas.
