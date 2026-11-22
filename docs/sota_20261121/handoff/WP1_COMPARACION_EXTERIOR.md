# WP1 · Comparación exterior emparejada con Garl-TTC

Antes de empezar, leer el [traspaso](README.md) y cumplir sus reglas.

## Pregunta

Con la misma historia de cajas y la misma cabeza, ¿el productor RGB de PHASE-TTC es mejor que el de Garl-TTC cuando **ninguno** ha visto las secuencias que se evalúan?

## Por qué importa

El 23,47 del test se debe en gran parte a la historia de cajas: solo cajas da 27,98. La única comparación hecha con cajas para ambos da empate (0,70 MiD [−0,31; 1,75]) y con Garl-TTC dentro de muestra. Sin este paquete no se puede decir que PHASE supere a Garl-TTC como método.

## Qué existe

- Protocolo congelado: [`matched_outer/PROTOCOL.md`](../matched_outer/PROTOCOL.md) y `PROTOCOL.json` (SHA-256 `c5a6260e…76508`). **No se modifica.** Si hace falta cambiar algo, se escribe una enmienda fechada antes de ver ninguna precisión.
- Informe parcial: [`matched_outer/REPORT.md`](../matched_outer/REPORT.md). Cerrados los brazos de solo cajas (46,09 ± 0,10 en cinco pliegues y tres semillas).
- Receta de reanudación: [`matched_outer/NEXT_RUN.json`](../matched_outer/NEXT_RUN.json), con las cuatro integraciones pendientes y los comandos.
- Código: `operational/matched_outer/`, `scripts/train_garl_release_cache.py`, `scripts/materialize_garlttc_rotating_cache.py`.
- Pruebas de humo ya hechas en CPU: una actualización real de Garl visual, Garl RGB+eventos y del productor `R_A5` de PHASE sobre dos filas.

## Tareas

1. **Caché por pliegue.** En `scripts/materialize_garlttc_rotating_cache.py` propagar `mask_reader=None` a `_materialize_row` y generar manifiestos sin pérdida por pliegue exterior e interior, indexados por protocolo, pliegue y huella del productor.
2. **Productores conscientes del pliegue.** En `operational/matched_outer/pilot.py` sustituir el consumo de las características PHASE20 precalculadas por salidas de productores entrenados solo con las secuencias permitidas. Las características existentes no sirven: sus productores vieron el bloque exterior.
3. **Ajuste completo del productor RGB de PHASE.** Extender `phase_rgb_smoke.py` a la fuente `IndependentRGBProducerSource` completa, con geometría de caja y caché del maestro DINOv3, y entrenar las 18 épocas fijadas.
4. **Garl-TTC por pliegue.** Entrenar `visual_only` y `rgbe_late_fusion` con `train_garl_release_cache.py`, 50 épocas, según los comandos de `NEXT_RUN.json`.
5. **Apilado común.** Crear `operational/matched_outer/stack.py`: la cabeza de cajas se entrena con las tablas fuera de pliegue de los cuatro pliegues interiores y se evalúa una vez en las ocho secuencias exteriores. Debe rechazar cualquier caché cuya huella de productor o clave de pliegue no coincida.
6. **Rotación.** Interiores 0 a 3 más reajuste, semillas 7, 13 y 21, y después los cinco pliegues exteriores.
7. **Informe.** Contraste primario `PHASE_RGB_BOX − GARL_RGB_BOX` con intervalo por remuestreo de secuencias, por banda y global, más `GARL_RGBE_BOX`.

## Antes de entrenar: presupuesto

Este paquete es el más caro. No lanzar el entrenamiento completo sin haber medido y escrito en el informe:

- Espacio: la extrapolación es de 56,9 GB para la caché completa. Al corte hay 57 GB libres en `C:` y 31 GB en `D:`. **Si no cabe con margen, parar y pedirlo al propietario.** Alternativa a evaluar: caché rotatoria que solo materializa un pliegue cada vez.
- Tiempo: medir una época real de cada productor en GPU y proyectar la rotación completa (5 exteriores × 5 ajustes × 3 semillas × 3 productores). Si supera lo razonable en esta máquina, proponer al propietario un recorte declarado de antemano (por ejemplo, una semilla en la rotación completa y tres solo en el contraste primario) o una GPU alquilada. No recortar después de ver resultados.
- La GPU debe estar dedicada: no solapar con WP5.

## Reglas específicas

- No mirar ninguna precisión exterior hasta que los cinco pliegues estén cerrados.
- Cada peso guarda la lista de secuencias vistas y excluidas; el validador existente debe pasar para todos.
- La comparación es entre productores con la misma cabeza, pérdida y cajas. No se ajusta nada a favor de un brazo.

## Regla de decisión (ya fijada en el protocolo)

Se informa el contraste con su intervalo. Tres lecturas posibles, todas publicables:

| Resultado | Lectura |
|---|---|
| Intervalo por debajo de cero | PHASE RGB supera a Garl RGB con la misma información |
| Intervalo que incluye cero | No se separan; la ventaja en test es de información |
| Intervalo por encima de cero | Garl-TTC es mejor productor; PHASE gana por usar la historia de cajas |

## Entregables

- `docs/sota_20261121/matched_outer/REPORT.md` actualizado, `RESULTS.json`, índice de evidencia con huellas de cachés, pesos y predicciones.
- Tablas fuera de pliegue completas de los tres productores: las necesita [WP2](WP2_CANDIDATO_TEST.md).
- Pruebas unitarias del apilado y del validador de exclusión.
