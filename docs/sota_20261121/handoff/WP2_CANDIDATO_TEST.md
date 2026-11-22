# WP2 · Un candidato nuevo para el test privado

Antes de empezar, leer el [traspaso](README.md) y [ADR-0047](../../decisions/ADR-0047-garlttc-submission-discipline.md).

## Pregunta

¿Hay una variante que mejore de forma fiable el 23,47 Overall MiD y merezca el cuarto envío de la cuenta?

## Qué hay medido y sin enviar

| Ingrediente | Efecto en pliegues frente a RGB + geometría (41,27) | Fuente |
|---|---|---|
| Geometría robusta + escala de contenido de eventos, sobre RGB (`RGB_ROBUST_CS`) | −2,42 [−2,90; −1,93] | [`saturation_scale`](../../sota_20261118/saturation_scale/REPORT.md) |
| Puerta de fiabilidad para cajas truncadas | −0,57 [−0,77; −0,38] global; −2,81 en truncadas | [`reliability_gate`](../reliability_gate/REPORT.md) |
| Promedio de semillas | sin medir como brazo | — |
| Ideas de [WP3](WP3_IDEAS_DE_MODELO.md) | pendientes | — |

`RGB_ROBUST_CS` no cumplió su regla por 0,07 puntos (se pedía un extremo superior por debajo de −2,0) y no se empaquetó. Esa decisión se respeta: no se envía ese brazo tal cual ni se rebaja el umbral.

## Advertencias que condicionan el diseño

- **Los pliegues actuales son de cabeza, no completos.** Los productores RGB vieron las secuencias evaluadas. El candidato debe medirse con los productores por pliegue de [WP1](WP1_COMPARACION_EXTERIOR.md). Si WP1 no está disponible, se puede medir con pliegues de cabeza, pero el informe debe decirlo y la regla se endurece (ver abajo).
- **Los niveles de pliegue y de test no coinciden** (85 frente a 35,8 para el sistema original; 41,3 frente a 23,5 para RGB + geometría). Las diferencias pueden no transferirse en tamaño. Ver [`level_gap`](../../sota_20261118/level_gap/REPORT.md).
- **La escala de contenido usa eventos.** El candidato vuelve a ser de eventos + RGB + cajas; el formulario de CodaBench debe decirlo.
- **Camino de entrada.** Al reconstruir desde imágenes, solo 5 de 12 consultas reprodujeron el TTC dentro de tolerancia ([`geo_reproduction`](../geo_reproduction/REPORT.md)). El paquete debe generarse desde descriptores congelados con huella, y el recibo debe registrar esas huellas.
- **Soporte en test.** Comprobar, sin etiquetas, que los campos de entrada del candidato caen dentro del soporte de entrenamiento en las consultas del test, incluida la validez de la escala de contenido. Informar el porcentaje fuera de soporte.

## Tareas

1. Escribir `docs/sota_20261121/test_candidate/PROTOCOL.md` con **un** brazo primario y como mucho dos de sensibilidad, definidos antes de calcular nada:
   - Primario propuesto: RGB + geometría robusta + escala de contenido + puerta de fiabilidad, predicción promedio de las semillas 7, 13 y 21.
   - Incorporar una idea de WP3 solo si ya cerró con intervalo que excluye el cero.
2. Medir en TRAIN40 fuera de pliegue (5 bloques de 8 secuencias) frente a `RGB_GEO_BAND`, con remuestreo pareado de secuencias, por banda y por estrato (libres, truncadas, saturadas reales, saturación sintética).
3. Proyectar el nivel en test con los métodos de `level_gap` y dejar escrito el rango esperado.
4. Diagnóstico sin etiquetas en el test: soporte de campos y desacuerdo con el paquete de 23,47.
5. Si se cumple la regla, generar el paquete, validarlo con el comprobador de formato (6.762 identificadores en orden oficial, cero fallos) y crear el recibo:
   ```sh
   python scripts/archive_submission.py <paquete.zip> --variant <nombre> --hypothesis "<texto>" \
     --commit <HEAD> --producer-path <fichero> --recipe <PROTOCOL.md> --seed 7
   ```
   El commit del recibo lo hace el orquestador. **Nadie sube el paquete salvo el propietario.**

## Regla de decisión

- Con productores por pliegue (WP1): empaquetar solo si el extremo superior del intervalo frente a `RGB_GEO_BAND` queda por debajo de **−2,0 MiD** y ninguna banda empeora con intervalo que excluya el cero.
- Solo con pliegues de cabeza: el umbral pasa a **−3,0 MiD**.
- Si no se cumple: no hay paquete, y el informe lo dice.

## Entregables

- `PROTOCOL.md`, `REPORT.md`, `RESULTS.json`, tabla por banda y estrato.
- Si procede: el ZIP, su recibo, la fila en el [registro de envíos](../../sota_20261118/SUBMISSION_LEDGER.md) como «preparado, no subido» y los valores del formulario (nombre del método, usa eventos, usa imagen).
- Mensaje para el propietario con el rango esperado y la advertencia de que el ranking muestra una sola entrada por participante.
